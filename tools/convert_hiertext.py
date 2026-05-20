import json
import random
import sys
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from tqdm import tqdm

try:
    from adet.utils.curve_utils import BezierCurve
    from adet.utils.polygon_utils import make_valid_poly
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from adet.utils.curve_utils import BezierCurve
    from adet.utils.polygon_utils import make_valid_poly

CTLABELS = [chr(i) for i in range(32, 127)]
CHAR_TO_ID = {c: i for i, c in enumerate(CTLABELS)}
assert len(CTLABELS) == 95

VOC_SIZE = 96
BLANK_TOKEN = VOC_SIZE - 1
PAD_TOKEN = VOC_SIZE
MAX_LEN = 100

CATEGORIES = [{
    "supercategory": "beverage",
    "id": 1,
    "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
    "name": "text"
}]


def text_to_rec(text, max_len=MAX_LEN):
    rec = [CHAR_TO_ID[c] for c in str(text).strip() if c in CHAR_TO_ID][:max_len]
    if len(rec) < max_len:
        rec.extend([PAD_TOKEN] * (max_len - len(rec)))
    return rec


def _vertices_to_pixels(verts, img_w, img_h):
    pts = np.empty((len(verts), 2), dtype=np.float32)
    max_xy = 0.0
    for i, v in enumerate(verts):
        if isinstance(v, dict):
            x, y = float(v["x"]), float(v["y"])
        else:
            x, y = float(v[0]), float(v[1])
        pts[i, 0] = x
        pts[i, 1] = y
        max_xy = max(max_xy, abs(x), abs(y))
    if max_xy <= 1.5 and img_w > 1 and img_h > 1:
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        return pts, "normalized"
    return pts, "absolute"


def _find_quad_corners(pts):
    x, y = pts[:, 0], pts[:, 1]
    return int(np.argmin(x + y)), int(np.argmax(x - y)), int(np.argmax(x + y)), int(np.argmin(x - y))


def _contour_slice(pts, start_idx, end_idx):
    n = len(pts)
    if start_idx == end_idx:
        return np.vstack([pts[start_idx], pts[start_idx]])
    out_idx = []
    i = start_idx
    while True:
        out_idx.append(i)
        if i == end_idx:
            break
        i = (i + 1) % n
        if len(out_idx) > n + 1:
            break
    return pts[out_idx]


def _split_polygon_top_bottom(poly_xy):
    poly = make_valid_poly(poly_xy.tolist())
    xs, ys = poly.exterior.xy
    pts = np.stack([xs, ys], axis=1)[:-1]
    n = len(pts)
    if n <= 2:
        return pts, pts

    tl_idx, tr_idx, br_idx, bl_idx = _find_quad_corners(pts)
    top_chain = _contour_slice(pts, tr_idx, tl_idx)[::-1]
    bottom_chain = _contour_slice(pts, bl_idx, br_idx)[::-1]

    if len({tl_idx, tr_idx, br_idx, bl_idx}) == 4 and len(top_chain) >= 2 and len(bottom_chain) >= 2:
        return top_chain, bottom_chain

    left_idx = int(np.argmin(pts[:, 0]))
    right_idx = int(np.argmax(pts[:, 0]))
    if left_idx < right_idx:
        c1 = pts[left_idx:right_idx + 1]
        idx2 = list(range(right_idx + 1, n)) + list(range(0, left_idx))
    else:
        c1 = np.vstack([pts[left_idx:], pts[:right_idx + 1]])
        idx2 = list(range(right_idx + 1, left_idx))
    c2 = pts[idx2] if len(idx2) >= 2 else pts[[left_idx, right_idx]]
    top_chain, bottom_chain = (c1, c2) if c1[:, 1].mean() <= c2[:, 1].mean() else (c2, c1)

    if len(top_chain) >= 2 and top_chain[0, 0] > top_chain[-1, 0]:
        top_chain = top_chain[::-1]
    if len(bottom_chain) >= 2 and bottom_chain[0, 0] < bottom_chain[-1, 0]:
        bottom_chain = bottom_chain[::-1]
    return top_chain, bottom_chain


def _resample_chain(chain_xy, num_samples=20):
    chain_xy = np.asarray(chain_xy, dtype=np.float32)
    if len(chain_xy) < 2:
        return np.repeat(chain_xy[:1], num_samples, axis=0)
    deltas = np.diff(chain_xy, axis=0)
    seg_len = np.sqrt((deltas ** 2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(seg_len)])
    if t[-1] == 0:
        return np.repeat(chain_xy[:1], num_samples, axis=0)
    t /= t[-1]
    u = np.linspace(0.0, 1.0, num_samples)
    return np.stack([np.interp(u, t, chain_xy[:, 0]), np.interp(u, t, chain_xy[:, 1])], axis=1)


def _line_bezier(chain_xy):
    p0, p3 = chain_xy[0], chain_xy[-1]
    v = (p3 - p0) / 3.0
    return np.stack([p0, p0 + v, p0 + 2.0 * v, p3], axis=0)


def _bezier_cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return np.array([
        u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
        u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1],
    ], dtype=np.float32)


def _sample_bezier_ctrl(ctrl, n=40):
    p0, p1, p2, p3 = ctrl
    return np.stack([_bezier_cubic(p0, p1, p2, p3, t) for t in np.linspace(0.0, 1.0, n)], axis=0)


def _bezier_to_ordered_ring(bezier_pts, n_samples=60):
    bp = np.asarray(bezier_pts, dtype=np.float32)
    if bp.shape[0] != 16:
        return None
    top = _sample_bezier_ctrl([bp[0:2], bp[2:4], bp[4:6], bp[6:8]], n=n_samples)
    bot = _sample_bezier_ctrl([bp[8:10], bp[10:12], bp[12:14], bp[14:16]], n=n_samples)
    return np.vstack([top, bot])


def _bezier_to_adet_eval_ring(bezier_pts, n_samples=25):
    bp = np.asarray(bezier_pts, dtype=np.float32)
    if bp.shape[0] != 16:
        return None
    curve_params = bp.reshape(-1, 2).reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    u = np.linspace(0.0, 1.0, n_samples)
    boundary = (
        np.outer((1 - u) ** 3, curve_params[:, 0])
        + np.outer(3 * u * ((1 - u) ** 2), curve_params[:, 1])
        + np.outer(3 * (u ** 2) * (1 - u), curve_params[:, 2])
        + np.outer(u ** 3, curve_params[:, 3])
    )
    bd = np.hstack([boundary[:, :2], boundary[:, 2:][::-1, :]])
    top, bottom_reversed = np.hsplit(bd, 2)
    return np.vstack([top, bottom_reversed[::-1]])


def _ring_has_self_intersection(bezier_pts, n_samples=60):
    for ring in [_bezier_to_ordered_ring(bezier_pts, n_samples), _bezier_to_adet_eval_ring(bezier_pts, 25)]:
        if ring is None:
            return True
        try:
            poly = Polygon(ring)
        except Exception:
            return True
        if not poly.is_empty and not poly.is_valid:
            reason = explain_validity(poly)
            # Esclude solo "Too few points" che non è una self-intersection
            #print(reason)
            return True
    return False


def _bbox_to_bezier(bbox_xywh):
    x_min, y_min, w, h = float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])
    x_max, y_max = x_min + w, y_min + h
    # Margine minimo per evitare Ring Self-intersection sugli angoli condivisi
    eps = min(w, h) * 1e-4

    top_p0 = np.array([x_min,       y_min + eps], dtype=np.float32)
    top_p1 = np.array([(2*x_min + x_max) / 3, y_min], dtype=np.float32)
    top_p2 = np.array([(x_min + 2*x_max) / 3, y_min], dtype=np.float32)
    top_p3 = np.array([x_max,       y_min + eps], dtype=np.float32)

    bot_p0 = np.array([x_max,       y_max - eps], dtype=np.float32)
    bot_p1 = np.array([(2*x_max + x_min) / 3, y_max], dtype=np.float32)
    bot_p2 = np.array([(x_max + 2*x_min) / 3, y_max], dtype=np.float32)
    bot_p3 = np.array([x_min,       y_max - eps], dtype=np.float32)

    return np.concatenate([top_p0, top_p1, top_p2, top_p3,
                           bot_p0, bot_p1, bot_p2, bot_p3]).astype(float).tolist()


def _fit_cubic_bezier(chain_xy):
    chain_xy = np.asarray(chain_xy, dtype=np.float32)
    n = chain_xy.shape[0]
    if n == 0:
        p = np.array([0.0, 0.0], dtype=np.float32)
        return np.stack([p, p, p, p], axis=0)
    if n == 1:
        return np.stack([chain_xy[0]] * 4, axis=0)
    if np.sqrt((np.diff(chain_xy, axis=0) ** 2).sum(axis=1)).sum() < 1e-6:
        return _line_bezier(chain_xy)
    try:
        bez = BezierCurve(order=3, num_sample_points=n)
        flat_cp = bez.get_middle_control_points(chain_xy[:, 0], chain_xy[:, 1])
        return np.array(flat_cp, dtype=np.float32).reshape(4, 2)
    except Exception:
        return _line_bezier(chain_xy)


def _poly_to_bezier(poly_xy, bbox_xywh, stats):
    """
    Converte poly_xy in bezier_pts (16 valori).
    stats: dict con chiavi 'self_intersection' e 'catastrophic' per i contatori.
    """
    poly_xy = np.asarray(poly_xy, dtype=np.float32)
    if len(poly_xy) == 0:
        return [0.0] * 16
    if len(poly_xy) == 1:
        return np.tile(poly_xy[0], 8).astype(float).tolist()

    top_chain, bottom_chain = _split_polygon_top_bottom(poly_xy)

    top_orig_start, top_orig_end = top_chain[0].copy(), top_chain[-1].copy()
    bot_orig_start, bot_orig_end = bottom_chain[0].copy(), bottom_chain[-1].copy()

    top_cp = _fit_cubic_bezier(_resample_chain(top_chain, num_samples=20))
    bot_cp = _fit_cubic_bezier(_resample_chain(bottom_chain, num_samples=20))

    top_cp[0], top_cp[3] = top_orig_start, top_orig_end
    bot_cp[0], bot_cp[3] = bot_orig_start, bot_orig_end

    x_min, x_max = float(poly_xy[:, 0].min()), float(poly_xy[:, 0].max())
    y_min, y_max = float(poly_xy[:, 1].min()), float(poly_xy[:, 1].max())
    top_cp[:, 0] = np.clip(top_cp[:, 0], x_min, x_max)
    top_cp[:, 1] = np.clip(top_cp[:, 1], y_min, y_max)
    bot_cp[:, 0] = np.clip(bot_cp[:, 0], x_min, x_max)
    bot_cp[:, 1] = np.clip(bot_cp[:, 1], y_min, y_max)

    bezier = np.concatenate([top_cp.reshape(-1), bot_cp.reshape(-1)]).astype(float).tolist()

    if _ring_has_self_intersection(bezier, n_samples=60):
        stats['self_intersection'] += 1
        bezier = _bbox_to_bezier(bbox_xywh)
        if _ring_has_self_intersection(bezier, n_samples=60):
            stats['catastrophic'] += 1

    return bezier


def _read_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "annotations" in data:
            return data["annotations"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _clean_ann(ann):
    return {k: v for k, v in ann.items() if not k.startswith("_")}


def _clean_with_text(ann):
    d = _clean_ann(ann)
    d["text"] = ann["_text"]
    return d


def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix=".jpg"):
    samples = _read_jsonl(jsonl_path)

    images, annotations_all = [], []
    ann_id = 1
    n_normalized = n_absolute = 0
    stats = {'self_intersection': 0, 'catastrophic': 0}

    # Conta le word totali per la progress bar
    total_words = sum(
        1
        for s in samples
        for p in s.get("paragraphs", [])
        for ln in p.get("lines", [])
        for w in ln.get("words", [])
        if len(w.get("vertices", [])) >= 3
    )

    desc = os.path.basename(jsonl_path)
    with tqdm(total=total_words, desc=desc, unit="word") as pbar:

        for img_id, sample in enumerate(samples):
            img_w = sample.get("image_width", 0)
            img_h = sample.get("image_height", 0)
            images.append({
                "width": img_w, "date_captured": "", "license": 0,
                "flickr_url": "", "file_name": sample["image_id"] + img_suffix,
                "id": img_id, "coco_url": "", "height": img_h,
            })

            for para in sample.get("paragraphs", []):
                for ln in para.get("lines", []):
                    for word in ln.get("words", []):
                        verts = word.get("vertices", [])
                        if len(verts) < 3:
                            continue

                        poly_xy, mode = _vertices_to_pixels(verts, img_w, img_h)
                        if len(poly_xy) < 3:
                            continue
                        if mode == "normalized":
                            n_normalized += 1
                        else:
                            n_absolute += 1

                        poly_xy[:, 0] = np.clip(poly_xy[:, 0], 0, img_w - 1)
                        poly_xy[:, 1] = np.clip(poly_xy[:, 1], 0, img_h - 1)

                        mins, maxs = poly_xy.min(axis=0), poly_xy.max(axis=0)
                        x_min, y_min = float(mins[0]), float(mins[1])
                        x_max, y_max = float(maxs[0]), float(maxs[1])
                        w = max(1.0, x_max - x_min)
                        h = max(1.0, y_max - y_min)

                        text_norm = str(word.get("text", "")).strip()
                        rec = text_to_rec(text_norm)
                        ignore = 1 if (not word.get("legible", True) or all(t == PAD_TOKEN for t in rec)) else 0

                        annotations_all.append({
                            "image_id": img_id,
                            "area": float(w * h),
                            "category_id": 1,
                            "iscrowd": 0,
                            "id": ann_id,
                            "bezier_pts": _poly_to_bezier(poly_xy, [x_min, y_min, w, h], stats),
                            "rec": rec,
                            "bbox": [x_min, y_min, w, h],
                            "_ignore": ignore,
                            "_text": text_norm,
                        })
                        ann_id += 1

                        pbar.set_postfix(SI=stats['self_intersection'],
                                         catastrophic=stats['catastrophic'],
                                         refresh=False)
                        pbar.update(1)

    annotations_supervised = [_clean_ann(ann) for ann in annotations_all if ann["_ignore"] == 0]
    annotations_gt = [_clean_with_text(ann) for ann in annotations_all]

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations_supervised, "categories": CATEGORIES}, f)
    with open(out_gt_source_path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations_gt, "categories": CATEGORIES}, f)

    n_ignored = len(annotations_all) - len(annotations_supervised)
    tqdm.write(f"[{desc}] img={len(images)}  ann={len(annotations_supervised)}  "
               f"ignored={n_ignored}  norm={n_normalized}  abs={n_absolute}  "
               f"SI={stats['self_intersection']}  catastrophic={stats['catastrophic']}")

    return images, annotations_all, annotations_supervised


def make_semi_splits(images, annotations_all, label_ratio, out_dir, split_name="train_96voc", seed=42):
    random.seed(seed)
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)
    n_label = max(1, int(len(img_ids) * label_ratio))
    labeled_ids = set(img_ids[:n_label])
    unlabeled_ids = set(img_ids[n_label:])

    ann_by_img = {}
    for ann in annotations_all:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def build_split(ids, include_text):
        split_imgs = [img for img in images if img["id"] in ids]
        split_anns = []
        for img in split_imgs:
            for ann in ann_by_img.get(img["id"], []):
                if ann["_ignore"] == 0:
                    d = _clean_ann(ann)
                    if include_text:
                        d["text"] = ann["_text"]
                    split_anns.append(d)
        return {"images": split_imgs, "annotations": split_anns, "categories": CATEGORIES}

    ratio_str = str(int(label_ratio * 100)) if label_ratio * 100 >= 1 else str(label_ratio * 100)
    labeled_path = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    labeled_data = build_split(labeled_ids, include_text=True)
    unlabeled_data = build_split(unlabeled_ids, include_text=False)

    with open(labeled_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f)
    with open(unlabeled_path, "w", encoding="utf-8") as f:
        json.dump(unlabeled_data, f)

    tqdm.write(f"  labeled   -> {labeled_path}  ({len(labeled_ids)} img, {len(labeled_data['annotations'])} ann)")
    tqdm.write(f"  unlabeled -> {unlabeled_path}  ({len(unlabeled_ids)} img, {len(unlabeled_data['annotations'])} ann)")


if __name__ == "__main__":
    BASE = "datasets/hiertext"

    tqdm.write("\n[1/3] Conversione validation -> test.json + test_gt_source.json")
    convert(
        jsonl_path=f"{BASE}/validation.jsonl",
        out_json_path=f"{BASE}/test.json",
        out_gt_source_path=f"{BASE}/test_gt_source.json",
    )

    tqdm.write("\n[2/3] Conversione train -> train_96voc.json + train_gt_source.json")
    images, annotations_all, annotations_supervised = convert(
        jsonl_path=f"{BASE}/train.jsonl",
        out_json_path=f"{BASE}/train_96voc.json",
        out_gt_source_path=f"{BASE}/train_gt_source.json",
    )

    tqdm.write("\n[3/3] Generazione split semi-supervised")
    for ratio in [0.005, 0.010, 0.020, 0.050, 0.10]:
        tqdm.write(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)