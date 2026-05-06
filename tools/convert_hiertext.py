import json

CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
            't','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
VOC_SIZE = 37

def text_to_rec(text):
    MAX_LEN = 25
    rec = []
    for c in text.lower():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:MAX_LEN]
    rec += [VOC_SIZE - 1] * (MAX_LEN - len(rec))
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


if __name__ == "__main__":
    convert(
        jsonl_path        = "datasets/hiertext/validation.jsonl",
        out_json_path     = "datasets/hiertext/test.json",
        out_gt_source_path= "datasets/hiertext/test_gt_source.json",
    )