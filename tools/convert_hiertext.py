import json
import random

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


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _vertices_to_pixels(verts, img_w, img_h):
    """
    HierText vertices possono essere:
      - normalizzati in [0, 1]  -> moltiplicare per img_w / img_h
      - gia' in pixel assoluti  -> usare direttamente

    La distinzione e' semplice: se tutti i valori x e y sono <= 1.0
    (con un piccolo margine per floating point) le coordinate sono
    normalizzate; altrimenti sono gia' in pixel.

    Restituisce una lista di [x_px, y_px] in pixel assoluti float.
    """
    if isinstance(verts[0], dict):
        pts_raw = [[v["x"], v["y"]] for v in verts]
    else:
        pts_raw = [[v[0], v[1]] for v in verts]

    xs_raw = [p[0] for p in pts_raw]
    ys_raw = [p[1] for p in pts_raw]

    # Se tutti i valori sono in [0, 1] le coordinate sono normalizzate
    is_normalized = (max(xs_raw) <= 1.0 + 1e-6) and (max(ys_raw) <= 1.0 + 1e-6)

    if is_normalized:
        pts = [[p[0] * img_w, p[1] * img_h] for p in pts_raw]
    else:
        pts = pts_raw  # gia' in pixel

    return pts, is_normalized


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
        "bbox": [x_min, y_min, width, height]         # XYWH_ABS
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
                    pts, was_normalized = _vertices_to_pixels(verts, img_w, img_h)
                    if was_normalized:
                        n_normalized += 1
                    else:
                        n_absolute += 1

                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)
                    # XYWH_ABS: width e height sono dimensioni, NON coordinate
                    w_box = x_max - x_min
                    h_box = y_max - y_min

                    # bezier_pts: 8 punti cubici di Bezier (top + bottom),
                    # 16 valori float, identico a CTW1500
                    p0 = pts[0]
                    p1 = pts[1] if len(pts) > 1 else pts[0]
                    p3 = pts[-1]
                    p2 = pts[-2] if len(pts) > 2 else pts[-1]
                    top = [p0, lerp(p0, p1, 1/3), lerp(p0, p1, 2/3), p1]
                    bot = [p3, lerp(p3, p2, 1/3), lerp(p3, p2, 2/3), p2]
                    bezier = [coord for p in top + bot for coord in p]

                    # 96-voc rec con MAX_LEN=100 (come CTW1500)
                    text_orig = str(word.get("text", ""))
                    legible = word.get("legible", True)
                    text_norm = text_orig.strip()
                    rec = text_to_rec(text_norm)

                    has_real_text = any(t != PAD_TOKEN for t in rec)
                    ignore = 1 if (not legible) or (not has_real_text) else 0

                    if w_box <= 0 or h_box <= 0:
                        ignore = 1

                    area = round(w_box * h_box, 1) if w_box > 0 and h_box > 0 else 0.0

                    # Annotazione nel formato CTW1500 esatto.
                    # I campi con prefisso _ sono interni e vengono rimossi
                    # prima della serializzazione finale.
                    ann = {
                        "image_id": img_id,
                        "area": area,
                        "category_id": 1,
                        "iscrowd": 0,
                        "id": ann_id,
                        "bezier_pts": bezier,
                        "rec": rec,
                        "bbox": [round(x_min, 2), round(y_min, 2),
                                 round(w_box, 2), round(h_box, 2)],
                        # campi interni (rimossi al momento della scrittura)
                        "_ignore": ignore,
                        "_text": text_norm,
                    }
                    annotations_all.append(ann)
                    ann_id += 1

                    if len(bbox_samples) < 5:
                        bbox_samples.append({
                            "img": fname, "img_wh": (img_w, img_h),
                            "was_normalized": was_normalized,
                            "bbox_xywh": [round(x_min, 2), round(y_min, 2),
                                          round(w_box, 2), round(h_box, 2)],
                        })

    # --- sanity check a schermo ---
    print(f"\n[SANITY CHECK bbox] vertices normalizzati={n_normalized}, gia' in pixel={n_absolute}")
    print("Prime 5 bbox (formato XYWH salvato nel JSON):")
    for s in bbox_samples:
        print(f"  img={s['img']} ({s['img_wh'][0]}x{s['img_wh'][1]}) "
              f"normalized={s['was_normalized']} bbox_xywh={s['bbox_xywh']}")
    print()

    # --- helper per pulizia finale delle annotazioni ---
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

    # Restituisce anche le annotazioni grezze (con campi interni) per uso in make_semi_splits
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

    ratio_str = (str(int(label_ratio * 100))
                 if label_ratio * 100 == int(label_ratio * 100)
                 else str(label_ratio))

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
    for ratio in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        print(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)

    print("\n[VERIFICA] Controllo coerenza file generati...")
    import numpy as np

    checks = [
        (f"{BASE}/train_96voc.json", True, "train_96voc (supervised)"),
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
        mean_w = float(np.mean(bboxes[:, 2])) if bboxes.size else 0.0
        mean_h = float(np.mean(bboxes[:, 3])) if bboxes.size else 0.0

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
        if wrong_rec_len > 0:
            status.append(f"ERRORE: {wrong_rec_len} rec con lunghezza != {MAX_LEN}")
        if mean_w > 300 or mean_h > 300:
            status.append(f"WARN: bbox media molto grande (mean_w={mean_w:.1f}, mean_h={mean_h:.1f})")

        result = "OK" if not status else " | ".join(status)
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  "
              f"rec_len={MAX_LEN}  bbox_mean=[w={mean_w:.1f}, h={mean_h:.1f}]  -> {result}")
