import json
import math
import random
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# BezierCurve from adet/utils/curve_utils
# (imported here so convert_hiertext can be run standalone from repo root)
# ---------------------------------------------------------------------------
try:
    from adet.utils.curve_utils import BezierCurve
    from adet.utils.polygon_utils import simplify_polygon, make_valid_poly
except ImportError:
    # Fallback: insert repo root in path and try again
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from adet.utils.curve_utils import BezierCurve
    from adet.utils.polygon_utils import simplify_polygon, make_valid_poly

# 96-voc convention (same as CTW1500 / SemiETS):
# ASCII printable characters: ' ' (32) ... '~' (126)  -> 95 symbols
# CTLABELS[i] = chr(i + 32)  for i in 0..94
# blank/unknown for CTC decoding: 95 (= VOC_SIZE - 1)
# dataset padding sentinel used in rec arrays: 96 (= VOC_SIZE)
CTLABELS = [chr(i) for i in range(32, 127)]  # 95 printable ASCII chars
assert len(CTLABELS) == 95

VOC_SIZE = 96
BLANK_TOKEN = VOC_SIZE - 1   # 95
PAD_TOKEN = VOC_SIZE          # 96

# CTW1500 usa rec di lunghezza 100 (verificato sul file di riferimento)
MAX_LEN = 100

# Struttura categories identica a CTW1500
CATEGORIES = [{
    "supercategory": "beverage",
    "id": 1,
    "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
    "name": "text"
}]


def text_to_rec(text, max_len=MAX_LEN):
    """Encode text using DeepSolo/SemiETS 96-voc format."""
    rec = []
    for c in str(text).strip():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:max_len]
    rec += [PAD_TOKEN] * (max_len - len(rec))
    assert len(rec) == max_len
    assert all((0 <= t < BLANK_TOKEN) or (t == PAD_TOKEN) for t in rec), \
        f"invalid rec tokens for DeepSolo 96-voc: {rec}"
    return rec


# ---------------------------------------------------------------------------
# Geometry helpers: vertices -> pixels, polygon -> Bezier
# ---------------------------------------------------------------------------

def _vertices_to_pixels(verts, img_w, img_h):
    """Converte i vertices di HierText in coordinate pixel assolute.

    HierText puo' usare due convenzioni:
      - assolute:  {"x": 123, "y": 45}
      - normalizzate: {"x": 0.1,  "y": 0.2}   (0..1)

    Questa funzione ritorna:
      pts : lista [(x,y), ...] in pixel
      mode: 'absolute' oppure 'normalized'
    """
    pts = []
    max_xy = 0.0
    for v in verts:
        if isinstance(v, dict):
            x, y = float(v["x"]), float(v["y"])
        else:
            x, y = float(v[0]), float(v[1])
        pts.append([x, y])
        max_xy = max(max_xy, abs(x), abs(y))

    mode = "absolute"
    if max_xy <= 1.5 and img_w > 1 and img_h > 1:  # vertices normalizzati
        mode = "normalized"
        for p in pts:
            p[0] *= img_w
            p[1] *= img_h
    return pts, mode


def _split_polygon_top_bottom(poly_xy):
    """Divide un poligono chiuso in catena superiore e inferiore.

    poly_xy: array-like (N, 2) in pixel.
    Restituisce: top_chain, bottom_chain (entrambi np.ndarray (M,2), M>=2).

    Strategia: usa make_valid_poly per correggere self-intersection e orientazione,
    poi prende i vertici dalla boundary principale e li spezza tra xmin e xmax.
    La catena con y_media minore e' considerata "top".

    I vertici usati per top e bottom sono, per quanto possibile, disgiunti:
    gli estremi comuni (sinistra/destra) vengono rimossi dalla catena bottom,
    eccetto nei casi degeneri in cui non resterebbero abbastanza punti.
    """
    poly = make_valid_poly(poly_xy.tolist())
    xs, ys = poly.exterior.xy
    pts = np.stack([xs, ys], axis=1)[:-1]  # rimuovi duplicato finale

    if len(pts) <= 2:
        return pts, pts

    left_idx = int(np.argmin(pts[:, 0]))
    right_idx = int(np.argmax(pts[:, 0]))

    if left_idx < right_idx:
        chain1 = pts[left_idx:right_idx + 1]
        chain2 = np.vstack([pts[right_idx:], pts[:left_idx + 1]])
    else:
        chain1 = np.vstack([pts[left_idx:], pts[:right_idx + 1]])
        chain2 = pts[right_idx:left_idx + 1]

    if chain1[:, 1].mean() <= chain2[:, 1].mean():
        top_chain, bottom_chain = chain1, chain2
    else:
        top_chain, bottom_chain = chain2, chain1

    # Rendi i vertici delle due catene disgiunti rimuovendo gli estremi
    # condivisi dalla catena bottom (salvo casi degeneri)
    top_start = top_chain[0]
    top_end = top_chain[-1]

    mask = []
    for p in bottom_chain:
        if np.allclose(p, top_start) or np.allclose(p, top_end):
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask, dtype=bool)
    bottom_pruned = bottom_chain[mask]
    if len(bottom_pruned) >= 2:
        bottom_chain = bottom_pruned

    return top_chain, bottom_chain


def _resample_chain(chain_xy, num_samples=20):
    """Resampling uniforme lungo la lunghezza della polilinea."""
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
    xs = np.interp(u, t, chain_xy[:, 0])
    ys = np.interp(u, t, chain_xy[:, 1])
    return np.stack([xs, ys], axis=1)


def _fit_cubic_bezier(chain_xy):
    """Fit robusto di una cubica di Bezier (4 ctrl points) ai punti chain_xy.

    In casi degeneri (tutti i punti coincidenti o quasi) fa fallback a una
    semplice retta dai primi agli ultimi punti, evitando errori di SVD.
    """
    chain_xy = np.asarray(chain_xy, dtype=np.float32)
    n = chain_xy.shape[0]
    if n == 0:
        p = np.array([0.0, 0.0], dtype=np.float32)
        return np.stack([p, p, p, p], axis=0)
    if n == 1:
        p = chain_xy[0]
        return np.stack([p, p, p, p], axis=0)

    x = chain_xy[:, 0]
    y = chain_xy[:, 1]

    # Controlla degenerazione: lunghezze segmento quasi nulle
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.sqrt(dx ** 2 + dy ** 2)
    if dt.sum() < 1e-6:
        # Tutti i punti quasi coincidenti -> retta degenerata
        p0 = chain_xy[0]
        p3 = chain_xy[-1]
        v = (p3 - p0) / 3.0
        p1 = p0 + v
        p2 = p0 + 2.0 * v
        return np.stack([p0, p1, p2, p3], axis=0)

    try:
        bez = BezierCurve(order=3, num_sample_points=n)
        flat_cp = bez.get_middle_control_points(x, y)
        cp = np.array(flat_cp, dtype=np.float32).reshape(4, 2)
        return cp
    except Exception:
        # Fallback robusto in caso di problemi numerici/SVD
        p0 = chain_xy[0]
        p3 = chain_xy[-1]
        v = (p3 - p0) / 3.0
        p1 = p0 + v
        p2 = p0 + 2.0 * v
        return np.stack([p0, p1, p2, p3], axis=0)


def _poly_to_bezier(poly_xy):
    """Converte un poligono HierText in 2 curve di Bezier (top+bottom).

    poly_xy: lista/array (N,2) in pixel assoluti.
    Restituisce una lista di 16 float:
      [TOP_p0, TOP_p1, TOP_p2, TOP_p3, BOT_p0, BOT_p1, BOT_p2, BOT_p3]
    con 4 punti di controllo per la parte alta e 4 per la parte bassa.

    Primo/ultimo punto di controllo coincidono sempre con i vertici estremi
    della catena superiore/inferiore (dopo lo split disgiunto). I due punti
    centrali sono stimati via least-squares usando BezierCurve.
    """
    poly_xy = np.asarray(poly_xy, dtype=np.float32)
    if len(poly_xy) < 2:
        p = poly_xy[0] if len(poly_xy) else np.array([0.0, 0.0], dtype=np.float32)
        return [float(p[0])] * 8 + [float(p[1])] * 8

    top_chain, bottom_chain = _split_polygon_top_bottom(poly_xy)
    top_chain_s = _resample_chain(top_chain, num_samples=20)
    bottom_chain_s = _resample_chain(bottom_chain, num_samples=20)

    # fit robusto
    top_cp = _fit_cubic_bezier(top_chain_s)
    bot_cp_free = _fit_cubic_bezier(bottom_chain_s)

    # forza il primo/ultimo ctrl point a coincidere con l'estremo della catena
    # TOP: sinistra -> destra
    top_p0 = top_chain_s[0]
    top_p3 = top_chain_s[-1]
    top_cp[0] = top_p0
    top_cp[3] = top_p3

    # BOT: destra -> sinistra (per compatibilita' con create_gt_hiertext)
    bot_chain_s_rev = bottom_chain_s[::-1]
    bot_p0 = bot_chain_s_rev[0]
    bot_p3 = bot_chain_s_rev[-1]
    bot_cp = bot_cp_free.copy()
    bot_cp[0] = bot_p0
    bot_cp[3] = bot_p3

    # clamp per sicurezza dentro il bounding box del poligono
    x_min, x_max = float(poly_xy[:, 0].min()), float(poly_xy[:, 0].max())
    y_min, y_max = float(poly_xy[:, 1].min()), float(poly_xy[:, 1].max())
    for cp in (top_cp, bot_cp):
        cp[:, 0] = np.clip(cp[:, 0], x_min, x_max)
        cp[:, 1] = np.clip(cp[:, 1], y_min, y_max)

    bez = np.concatenate([top_cp.reshape(-1), bot_cp.reshape(-1)]).astype(float)
    assert bez.shape[0] == 16
    return bez.tolist()


# ---------------------------------------------------------------------------
# JSON reading
# ---------------------------------------------------------------------------

def _read_jsonl(jsonl_path):
    """Legge un file HierText .jsonl (JSON unico o JSON Lines)."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "annotations" in data:
            return data["annotations"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    samples = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    """Converti HierText nel formato COCO-like usato da DeepSolo/SemiETS."""
    samples = _read_jsonl(jsonl_path)

    images, annotations_all = [], []
    ann_id = 1
    n_normalized = 0
    n_absolute = 0
    bbox_samples = []  # per sanity check

    for img_id, sample in enumerate(samples):
        fname = sample['image_id'] + img_suffix
        img_w = sample.get("image_width", 0)
        img_h = sample.get("image_height", 0)

        images.append({
            "width": img_w,
            "date_captured": "",
            "license": 0,
            "flickr_url": "",
            "file_name": fname,
            "id": img_id,
            "coco_url": "",
            "height": img_h,
        })

        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if len(verts) < 3:
                        continue

                    poly_xy, mode = _vertices_to_pixels(verts, img_w, img_h)
                    poly_xy = np.asarray(poly_xy, dtype=np.float32)
                    if len(poly_xy) < 3:
                        continue

                    if mode == "normalized":
                        n_normalized += 1
                    else:
                        n_absolute += 1

                    poly_xy[:, 0] = np.clip(poly_xy[:, 0], 0, img_w - 1)
                    poly_xy[:, 1] = np.clip(poly_xy[:, 1], 0, img_h - 1)

                    x_min = float(np.min(poly_xy[:, 0]))
                    y_min = float(np.min(poly_xy[:, 1]))
                    x_max = float(np.max(poly_xy[:, 0]))
                    y_max = float(np.max(poly_xy[:, 1]))
                    w = max(1.0, x_max - x_min)
                    h = max(1.0, y_max - y_min)
                    bbox = [x_min, y_min, w, h]
                    area = float(w * h)
                    bbox_samples.append((w, h))

                    bezier = _poly_to_bezier(poly_xy)

                    text_orig = str(word.get("text", ""))
                    legible = word.get("legible", True)
                    text_norm = text_orig.strip()
                    rec = text_to_rec(text_norm)

                    has_real_text = any(t != PAD_TOKEN for t in rec)
                    ignore = 1 if (not legible) or (not has_real_text) else 0

                    ann = {
                        "image_id": img_id,
                        "area": area,
                        "category_id": 1,
                        "iscrowd": 0,
                        "id": ann_id,
                        "bezier_pts": bezier,
                        "rec": rec,
                        "bbox": bbox,
                        "_ignore": ignore,
                        "_text": text_norm,
                    }
                    annotations_all.append(ann)
                    ann_id += 1

    print(f"  vertices normalizzati: {n_normalized}")
    print(f"  vertices assoluti:    {n_absolute}")

    def _clean(ann):
        return {k: v for k, v in ann.items() if not k.startswith("_")}

    def _clean_with_text(ann):
        d = _clean(ann)
        d["text"] = ann["_text"]
        return d

    annotations_supervised = [_clean(ann) for ann in annotations_all if ann["_ignore"] == 0]
    annotations_gt = [_clean_with_text(ann) for ann in annotations_all]

    coco_supervised = {
        "images": images,
        "annotations": annotations_supervised,
        "categories": CATEGORIES,
    }
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_supervised, f)

    coco_gt_source = {
        "images": images,
        "annotations": annotations_gt,
        "categories": CATEGORIES,
    }
    with open(out_gt_source_path, 'w', encoding='utf-8') as f:
        json.dump(coco_gt_source, f)

    n_ignored = len(annotations_all) - len(annotations_supervised)
    print(f"Scritto {out_json_path}")
    print(f"  -> {len(images)} immagini, {len(annotations_supervised)} ann valide, {n_ignored} ignored")
    print(f"Scritto {out_gt_source_path}")
    print(f"  -> {len(annotations_all)} annotazioni totali (ignored incluse)")

    return images, annotations_all, annotations_supervised


# ---------------------------------------------------------------------------
# Semi-supervised splits (labeled / unlabeled)
# ---------------------------------------------------------------------------

def make_semi_splits(images, annotations_all, label_ratio, out_dir,
                     split_name="train_96voc", seed=42):
    random.seed(seed)
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)
    n_label = max(1, int(len(img_ids) * label_ratio))
    labeled_ids = set(img_ids[:n_label])
    unlabeled_ids = set(img_ids[n_label:])

    ann_by_img = {}
    for ann in annotations_all:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def _clean(ann):
        return {k: v for k, v in ann.items() if not k.startswith("_")}

    def build_split(ids, include_text: bool):
        split_imgs = [img for img in images if img["id"] in ids]
        split_anns = []
        for img in split_imgs:
            for ann in ann_by_img.get(img["id"], []):
                if ann["_ignore"] == 0:
                    d = _clean(ann)
                    if include_text:
                        d["text"] = ann["_text"]
                    split_anns.append(d)
        return {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": CATEGORIES,
        }

    if label_ratio * 100 >= 1:
        ratio_str = str(int(label_ratio * 100))
    else:
        ratio_str = str(label_ratio * 100)

    labeled_path = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    labeled_data = build_split(labeled_ids, include_text=True)
    unlabeled_data = build_split(unlabeled_ids, include_text=False)

    with open(labeled_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f)
    with open(unlabeled_path, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_data, f)

    print(f"  labeled   -> {labeled_path}")
    print(f"              {len(labeled_ids)} img, {len(labeled_data['annotations'])} ann")
    print(f"  unlabeled -> {unlabeled_path}")
    print(f"              {len(unlabeled_ids)} img, {len(unlabeled_data['annotations'])} ann")


if __name__ == "__main__":
    BASE = "datasets/hiertext"

    print("\n[1/3] Conversione validation -> test.json + test_gt_source.json")
    convert(
        jsonl_path=f"{BASE}/validation.jsonl",
        out_json_path=f"{BASE}/test.json",
        out_gt_source_path=f"{BASE}/test_gt_source.json",
    )

    print("\n[2/3] Conversione train -> train_96voc.json + train_gt_source.json")
    images, annotations_all, annotations_supervised = convert(
        jsonl_path=f"{BASE}/train.jsonl",
        out_json_path=f"{BASE}/train_96voc.json",
        out_gt_source_path=f"{BASE}/train_gt_source.json",
    )

    print("\n[3/3] Generazione split semi-supervised")
    for ratio in [0.005, 0.010, 0.020, 0.050, 0.10]:
        print(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)

    print("\n[VERIFICA] Controllo coerenza file generati...")
    import numpy as np

    checks = [
        (f"{BASE}/train_96voc.json", False, "train_96voc (supervised)"),
        (f"{BASE}/train_96voc_10_labeled.json", True, "10% labeled"),
        (f"{BASE}/train_96voc_10_unlabeled.json", False, "10% unlabeled"),
    ]
    for path, expect_text, label in checks:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        anns = d["annotations"]
        recs = np.array([a["rec"] for a in anns]) if anns else np.zeros((0, MAX_LEN), dtype=np.int64)
        bad = int(((recs == BLANK_TOKEN).sum() if recs.size else 0))
        over = int(((recs > PAD_TOKEN).sum() if recs.size else 0))
        has_text = all("text" in a for a in anns) if anns else True
        has_ignore = any("ignore" in a for a in anns) if anns else False
        has_seg = any("segmentation" in a for a in anns) if anns else False
        all_pad_rows = int(np.sum(np.all(recs == PAD_TOKEN, axis=1))) if recs.size else 0
        wrong_rec_len = sum(1 for a in anns if len(a["rec"]) != MAX_LEN)

        bboxes = np.array([a["bbox"] for a in anns]) if anns else np.zeros((0, 4))
        bad_bbox = int(np.sum((bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0))) if bboxes.size else 0
        neg_bbox = int(np.sum((bboxes[:, 0] < 0) | (bboxes[:, 1] < 0))) if bboxes.size else 0
        mean_w = float(np.mean(bboxes[:, 2])) if bboxes.size else 0.0
        mean_h = float(np.mean(bboxes[:, 3])) if bboxes.size else 0.0

        bez_lens = [len(a["bezier_pts"]) for a in anns]) if anns else []
        wrong_bez = sum(1 for l in bez_lens if l != 16)

        status = []
        if bad > 0:
            status.append(f"ERRORE: trovati token blank=95 nei GT ({bad})")
        if over > 0:
            status.append(f"ERRORE: {over} token rec > {PAD_TOKEN}")
        if has_ignore:
            status.append("ERRORE: campo 'ignore' presente")
        if has_seg:
            status.append("ERRORE: campo 'segmentation' presente (non deve esserci)")
        if expect_text and not has_text:
            status.append("ERRORE: 'text' mancante")
        if not expect_text and has_text:
            status.append("WARN: 'text' presente (inatteso)")
        if all_pad_rows > 0:
            status.append(f"ERRORE: {all_pad_rows} istanze con rec tutto padding")
        if bad_bbox > 0:
            status.append(f"ERRORE: {bad_bbox} bbox con w<=0 o h<=0")
        if neg_bbox > 0:
            status.append(f"ERRORE: {neg_bbox} bbox con x_min<0 o y_min<0 (clip mancante)")
        if wrong_rec_len > 0:
            status.append(f"ERRORE: {wrong_rec_len} rec con lunghezza != {MAX_LEN}")
        if wrong_bez > 0:
            status.append(f"ERRORE: {wrong_bez} bezier_pts con lunghezza != 16")

        result = "OK" if not status else " | ".join(status)
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  rec_len={MAX_LEN}  bbox_mean=[w={mean_w:.1f}, h={mean_h:.1f}]  -> {result}")
