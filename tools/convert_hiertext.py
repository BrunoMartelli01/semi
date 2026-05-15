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
    """
    Encode text using DeepSolo/SemiETS 96-voc format.

    Mapping:
      ASCII printable chars (space=32 ... ~=126) -> 0..94
      blank/unknown is reserved at 95 and is NOT written into GT rec
      padding sentinel is 96

    Characters outside the printable ASCII range are ignored.
    Output length is fixed to max_len with PAD_TOKEN=96.
    """
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



def _read_jsonl(jsonl_path):
    """
    Legge un file HierText .jsonl.

    HierText usa un formato ibrido: il file contiene UN singolo oggetto JSON
    con chiave 'annotations' che contiene la lista di tutte le immagini.
    In alternativa, supporta anche il formato JSON Lines puro (un oggetto per riga).
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    # Prova prima come singolo oggetto JSON (formato HierText standard)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "annotations" in data:
            return data["annotations"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: formato JSON Lines (una riga = un oggetto)
    samples = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    """
    Convert HierText annotations to the COCO-like format expected by DeepSolo/SemiETS,
    identico allo schema di datasets/ctw1500/train_96voc.json.

    Formato output per ogni annotazione:
      {
        "image_id": <int>,
        "area": <float>,
        "category_id": 1,
        "iscrowd": 0,
        "id": <int>,
        "bezier_pts": [x0,y0, x1,y1, ..., x7,y7],   # 8 punti = 16 valori
        "rec": [int * 100],                           # lunghezza 100, PAD=96
        "bbox": [x_min, y_min, width, height]         # XYWH_ABS, clippato a [0, img_wh]
      }


    Campi NON presenti (rimossi rispetto alla versione precedente):
      - "segmentation"  (non esiste in CTW1500)
      - "text"          (non esiste in CTW1500 supervised)
      - "ignore"        (non esiste in CTW1500)

    Struttura "images" identica a CTW1500:
      {width, date_captured, license, flickr_url, file_name, id, coco_url, height}
    """
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

        # Struttura images identica a CTW1500
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

                    # --- converti vertices in pixel assoluti ---


                    # 96-voc rec con MAX_LEN=100 (come CTW1500)
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
                        # campi interni (rimossi al momento della scrittura)
                        "_ignore": ignore,
                        "_text": text_norm,
                    }
                    annotations_all.append(ann)
                    ann_id += 1





    def _clean(ann):
        return {k: v for k, v in ann.items() if not k.startswith("_")}

    def _clean_with_text(ann):
        d = _clean(ann)
        d["text"] = ann["_text"]
        return d

    annotations_supervised = [
        _clean(ann) for ann in annotations_all if ann["_ignore"] == 0
    ]
    annotations_gt = [_clean_with_text(ann) for ann in annotations_all]

    # --- supervised (no text, no ignore, no segmentation) ---
    coco_supervised = {
        "images": images,
        "annotations": annotations_supervised,
        "categories": CATEGORIES,
    }
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_supervised, f)

    # --- gt_source: tutte le annotazioni + campo text per evaluation ---
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

        bez_lens = [len(a["bezier_pts"]) for a in anns] if anns else []
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
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  "
              f"rec_len={MAX_LEN}  bbox_mean=[w={mean_w:.1f}, h={mean_h:.1f}]  -> {result}")
