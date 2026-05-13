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
MAX_LEN = 25


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


def _make_clean_ann(ann, keep_text: bool):
    strip = {"ignore"} if keep_text else {"ignore", "text"}
    return {k: v for k, v in ann.items() if k not in strip}


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


def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    """
    Convert HierText annotations to the COCO-like format expected by DeepSolo/SemiETS.

    Le bbox vengono salvate in formato XYWH_ABS (x_min, y_min, width, height)
    in pixel assoluti, coerente con bbox_mode=BoxMode.XYWH_ABS usato in
    adet/data/datasets/text.py.

    Output rec format follows SemiETS 96-voc (CTW1500) exactly:
    - VOC_SIZE = 96
    - characters occupy ids 0..94  (ASCII printable: space=0, ...~=94)
    - id 95 is reserved for CTC blank/unknown during decoding
    - GT rec sequences are padded with 96 to fixed length 25
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images, annotations_all = [], []
    ann_id = 1
    n_normalized = 0
    n_absolute = 0
    bbox_samples = []  # per sanity check

    for img_id, sample in enumerate(data["annotations"]):
        fname = sample['image_id'] + img_suffix
        img_w = sample.get("image_width", 0)
        img_h = sample.get("image_height", 0)
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": img_w,
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
                    poly = [coord for p in pts for coord in p]

                    p0 = pts[0]
                    p1 = pts[1] if len(pts) > 1 else pts[0]
                    p3 = pts[-1]
                    p2 = pts[-2] if len(pts) > 2 else pts[-1]
                    top = [p0, lerp(p0, p1, 1/3), lerp(p0, p1, 2/3), p1]
                    bot = [p3, lerp(p3, p2, 1/3), lerp(p3, p2, 2/3), p2]
                    bezier = [coord for p in top + bot for coord in p]

                    # 96-voc: preserve original case (no .lower())
                    text_orig = str(word.get("text", ""))
                    legible = word.get("legible", True)
                    text_norm = text_orig.strip()
                    rec = text_to_rec(text_norm)

                    # instances con nessun testo reale dopo encoding vengono ignorate
                    has_real_text = any(t != PAD_TOKEN for t in rec)
                    ignore = 1 if (not legible) or (not has_real_text) else 0

                    # bbox sanity check: w e h devono essere > 0
                    if w_box <= 0 or h_box <= 0:
                        ignore = 1

                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        # XYWH_ABS: [x_min, y_min, width, height] in pixel assoluti
                        "bbox": [x_min, y_min, w_box, h_box],
                        "area": w_box * h_box,
                        "segmentation": [poly],
                        "bezier_pts": bezier,
                        "rec": rec,
                        "text": text_norm,
                        "iscrowd": 0,
                        "ignore": ignore,
                    }
                    annotations_all.append(ann)
                    ann_id += 1

                    if len(bbox_samples) < 5:
                        bbox_samples.append({
                            "img": fname, "img_wh": (img_w, img_h),
                            "was_normalized": was_normalized,
                            "bbox_xywh": [round(x_min,2), round(y_min,2), round(w_box,2), round(h_box,2)],
                            "bbox_xyxy_check": [round(x_min,2), round(y_min,2), round(x_max,2), round(y_max,2)],
                        })

    # --- sanity check a schermo ---
    print(f"\n[SANITY CHECK bbox] vertices normalizzati={n_normalized}, gia' in pixel={n_absolute}")
    print("Prime 5 bbox (formato XYWH salvato nel JSON):")
    for s in bbox_samples:
        print(f"  img={s['img']} ({s['img_wh'][0]}x{s['img_wh'][1]}) "
              f"normalized={s['was_normalized']} "
              f"bbox_xywh={s['bbox_xywh']} "
              f"-> xyxy_atteso={s['bbox_xyxy_check']}")
    print()

    coco_gt_source = {
        "images": images,
        "annotations": annotations_all,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_gt_source_path, 'w', encoding='utf-8') as f:
        json.dump(coco_gt_source, f)

    annotations_supervised = [
        _make_clean_ann(a, keep_text=True)
        for a in annotations_all if a["ignore"] == 0
    ]
    coco_supervised = {
        "images": images,
        "annotations": annotations_supervised,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_supervised, f)

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

    def build_split(ids, keep_text: bool):
        split_imgs = [img for img in images if img["id"] in ids]
        split_anns = [
            _make_clean_ann(ann, keep_text=keep_text)
            for img in split_imgs
            for ann in ann_by_img.get(img["id"], [])
            if ann["ignore"] == 0
        ]
        return {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": [{"id": 1, "name": "text"}]
        }

    ratio_str = str(int(label_ratio * 100)) if label_ratio * 100 == int(label_ratio * 100) else str(label_ratio)

    labeled_path = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    labeled_data = build_split(labeled_ids, keep_text=True)
    unlabeled_data = build_split(unlabeled_ids, keep_text=False)

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
    for ratio in [0.05, 0.10, 0.20]:
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
        all_pad_rows = int(np.sum(np.all(recs == PAD_TOKEN, axis=1))) if recs.size else 0

        # Verifica bbox: w e h devono essere > 0 per ogni annotazione
        bboxes = np.array([a["bbox"] for a in anns]) if anns else np.zeros((0, 4))
        bad_bbox = int(np.sum((bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0))) if bboxes.size else 0
        # Verifica che le bbox siano in XYWH e non XYXY (w e h devono essere << img size)
        # Una bbox in XYXY avrebbe x2 > x1 con valori tipicamente > 100px
        # Una bbox in XYWH avrebbe w = x2-x1 (dimensione reale, piu' piccola)
        # Avvisiamo se la bbox media sembra troppo grande (possibile errore XYXY)
        if bboxes.size:
            mean_w = float(np.mean(bboxes[:, 2]))
            mean_h = float(np.mean(bboxes[:, 3]))
        else:
            mean_w = mean_h = 0.0

        status = []
        if bad > 0:
            status.append(f"ERRORE: trovati token blank=95 nei GT ({bad})")
        if over > 0:
            status.append(f"ERRORE: {over} token rec > {PAD_TOKEN}")
        if has_ignore:
            status.append("ERRORE: campo 'ignore' presente")
        if expect_text and not has_text:
            status.append("ERRORE: 'text' mancante")
        if not expect_text and has_text:
            status.append("WARN: 'text' presente (inatteso)")
        if all_pad_rows > 0:
            status.append(f"ERRORE: {all_pad_rows} istanze con rec tutto padding")
        if bad_bbox > 0:
            status.append(f"ERRORE: {bad_bbox} bbox con w<=0 o h<=0")
        if mean_w > 300 or mean_h > 300:
            status.append(f"WARN: bbox media molto grande (mean_w={mean_w:.1f}, mean_h={mean_h:.1f}) -- possibile XYXY invece di XYWH?")

        result = "OK" if not status else " | ".join(status)
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  "
              f"bbox_mean=[w={mean_w:.1f}, h={mean_h:.1f}]  -> {result}")
