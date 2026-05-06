import json
import random

CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
            't','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
VOC_SIZE = 37
PAD_TOKEN = VOC_SIZE - 1   # 36 = EOS/blank  ← MUST be VOC_SIZE-1, never VOC_SIZE

def text_to_rec(text):
    MAX_LEN = 25
    rec = []
    for c in text.lower():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:MAX_LEN]
    rec += [PAD_TOKEN] * (MAX_LEN - len(rec))   # padding with 36, not 37
    assert all(0 <= t <= PAD_TOKEN for t in rec), \
        f"rec token out of range [0,{PAD_TOKEN}]: {rec}"
    return rec

def lerp(a, b, t):
    return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]

def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    with open(jsonl_path) as f:
        data = json.load(f)

    images, annotations_all = [], []
    ann_id = 1

    for img_id, sample in enumerate(data["annotations"]):
        fname = sample['image_id'] + img_suffix
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": sample.get("image_width", 0),
            "height": sample.get("image_height", 0),
        })

        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if len(verts) < 3:
                        continue

                    if isinstance(verts[0], dict):
                        pts = [[v["x"], v["y"]] for v in verts]
                    else:
                        pts = [[v[0], v[1]] for v in verts]

                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x_min, y_min = min(xs), min(ys)
                    w_box = max(xs) - x_min
                    h_box = max(ys) - y_min
                    poly = [coord for p in pts for coord in p]

                    p0, p1 = pts[0], pts[1] if len(pts) > 1 else pts[0]
                    p3, p2 = pts[-1], pts[-2] if len(pts) > 2 else pts[-1]
                    top = [p0, lerp(p0,p1,1/3), lerp(p0,p1,2/3), p1]
                    bot = [p3, lerp(p3,p2,1/3), lerp(p3,p2,2/3), p2]
                    bezier = [coord for p in top+bot for coord in p]

                    text_orig = word.get("text", "")
                    legible   = word.get("legible", True)
                    ignore    = 1 if not legible or text_orig == "" else 0
                    rec       = text_to_rec(text_orig)

                    annotations_all.append({
                        "id":           ann_id,
                        "image_id":     img_id,
                        "category_id":  1,
                        "bbox":         [x_min, y_min, w_box, h_box],
                        "area":         w_box * h_box,
                        "segmentation": [poly],
                        "bezier_pts":   bezier,
                        "rec":          rec,
                        "text":         text_orig.lower().strip(),
                        "iscrowd":      0,
                        "ignore":       ignore,
                    })
                    ann_id += 1

    # --- file GT source: tutte le annotazioni (ignored incluse) ---
    coco_full = {
        "images": images,
        "annotations": annotations_all,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_gt_source_path, "w") as f:
        json.dump(coco_full, f)

    # --- file loader: solo annotazioni valide, senza campi ignore/text ---
    annotations_clean = []
    for a in annotations_all:
        if a["ignore"] == 1:
            continue
        entry = {k: v for k, v in a.items() if k not in ("ignore", "text")}
        annotations_clean.append(entry)

    coco_clean = {
        "images": images,
        "annotations": annotations_clean,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_json_path, "w") as f:
        json.dump(coco_clean, f)

    print(f"Scritto {out_json_path}  →  {len(images)} immagini, {len(annotations_clean)} annotazioni valide")
    print(f"Scritto {out_gt_source_path}  →  {len(annotations_all)} annotazioni totali ({len(annotations_all)-len(annotations_clean)} ignored)")
    return images, annotations_all, annotations_clean


def make_semi_splits(images, annotations_all, label_ratio, out_dir,
                     split_name="train_37voc", seed=42):
    """
    Genera i JSON labeled/unlabeled per il training semi-supervised.
    label_ratio: frazione di immagini labeled (es. 0.10 per 10%)
    """
    random.seed(seed)
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)
    n_label = max(1, int(len(img_ids) * label_ratio))
    labeled_ids   = set(img_ids[:n_label])
    unlabeled_ids = set(img_ids[n_label:])

    ann_by_img = {}
    for ann in annotations_all:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def build_split(ids):
        split_imgs = [img for img in images if img["id"] in ids]
        split_anns = []
        for img in split_imgs:
            for ann in ann_by_img.get(img["id"], []):
                if ann["ignore"] == 1:
                    continue
                entry = {k: v for k, v in ann.items() if k not in ("ignore", "text")}
                split_anns.append(entry)
        return {"images": split_imgs, "annotations": split_anns,
                "categories": [{"id": 1, "name": "text"}]}

    ratio_str = str(int(label_ratio * 100)) if label_ratio * 100 == int(label_ratio * 100) \
                else str(label_ratio)

    labeled_path   = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    with open(labeled_path, "w") as f:
        json.dump(build_split(labeled_ids), f)
    with open(unlabeled_path, "w") as f:
        json.dump(build_split(unlabeled_ids), f)

    print(f"  labeled   → {labeled_path}  ({len(labeled_ids)} img)")
    print(f"  unlabeled → {unlabeled_path}  ({len(unlabeled_ids)} img)")


if __name__ == "__main__":
    BASE = "datasets/hiertext"

    # ── 1. TEST / VALIDATION ────────────────────────────────────────────────
    print("\n[1/3] Conversione validation → test.json")
    convert(
        jsonl_path         = f"{BASE}/validation.jsonl",
        out_json_path      = f"{BASE}/test.json",
        out_gt_source_path = f"{BASE}/test_gt_source.json",
    )

    # ── 2. TRAINING COMPLETO ────────────────────────────────────────────────
    print("\n[2/3] Conversione train → train_37voc.json")
    images, annotations_all, annotations_clean = convert(
        jsonl_path         = f"{BASE}/train.jsonl",
        out_json_path      = f"{BASE}/train_37voc.json",
        out_gt_source_path = f"{BASE}/train_gt_source.json",
    )

    # ── 3. SPLIT SEMI-SUPERVISED ────────────────────────────────────────────
    print("\n[3/3] Generazione split semi-supervised")
    for ratio in [0.05, 0.10, 0.20]:
        print(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)

    # ── VERIFICA FINALE rec ─────────────────────────────────────────────────
    print("\n[VERIFICA] Controllo token rec in train_37voc.json ...")
    import numpy as np
    with open(f"{BASE}/train_37voc.json") as f:
        d = json.load(f)
    recs = np.array([a["rec"] for a in d["annotations"]])
    bad = (recs > PAD_TOKEN).sum()
    print(f"  Token > {PAD_TOKEN} (devono essere 0): {bad}")
    assert bad == 0, "ERRORE: ci sono ancora token fuori range!"
    print("  ✅ Tutti i token rec sono nel range corretto [0, 36]")
