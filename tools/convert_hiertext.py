import json
import random

CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
            't','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
# Vocabolario: 36 caratteri (indici 0-35)
#   a-z  -> 0-25
#   0-9  -> 26-35
# Il padding/blank token è 37 (= VOC_SIZE), NON 36.
# I modelli SemiETS/ABCNet si aspettano rec in [0,35] + pad=37 di lunghezza 25.
VOC_SIZE  = 36          # numero di caratteri reali nel vocabolario
PAD_TOKEN = 37          # token di padding (usato per riempire fino a MAX_LEN=25)

def text_to_rec(text, max_len=25):
    """
    Converte una stringa in una lista di interi lunga max_len.

    Logica:
      - ogni carattere alfanumerico (a-z, 0-9) viene mappato al suo indice in CTLABELS
      - i caratteri non presenti in CTLABELS vengono ignorati
      - la lista viene troncata a max_len e riempita con PAD_TOKEN (37)

    Esempio:
      'avon' -> [0, 21, 14, 13, 37, 37, ..., 37]  (25 elementi)
    """
    rec = []
    for c in str(text).lower():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:max_len]
    rec += [PAD_TOKEN] * (max_len - len(rec))
    assert all(0 <= t <= PAD_TOKEN for t in rec), \
        f"rec token out of range [0,{PAD_TOKEN}]: {rec}"
    return rec


def lerp(a, b, t):
    return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]


def _make_clean_ann(ann, keep_text: bool):
    """
    Restituisce una copia dell'annotazione pronta per il file loader:
      - rimuove sempre 'ignore'
      - rimuove 'text' solo se keep_text=False (file unlabeled)
    """
    strip = {"ignore"} if keep_text else {"ignore", "text"}
    return {k: v for k, v in ann.items() if k not in strip}


def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    """
    Converte un file HierText JSONL in due file COCO:
      - out_gt_source_path : tutte le annotazioni con tutti i campi
                             (inclusi 'text' e 'ignore') - usato per eval/debug
      - out_json_path      : solo annotazioni valide (ignore==0),
                             con 'text' mantenuto (file di training supervisionato)

    Formato rec generato:
      lista di 25 interi in [0,35] per i caratteri + 37 come padding.
      Esempio: 'hello' -> [7,4,11,11,14,37,37,...,37]
    """
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
                        "iscrowd":      0,
                        "bbox":         [x_min, y_min, w_box, h_box],
                        "area":         w_box * h_box,
                        "segmentation": [poly],
                        "bezier_pts":   bezier,
                        "rec":          rec,
                        "text":         text_orig.lower().strip(),
                        "ignore":       ignore,
                    })
                    ann_id += 1

    # --- gt_source: tutte le annotazioni, tutti i campi (eval/debug) ---
    coco_gt_source = {
        "images": images,
        "annotations": annotations_all,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_gt_source_path, "w") as f:
        json.dump(coco_gt_source, f)

    # --- file loader supervisionato: valide + text mantenuto ---
    annotations_supervised = [
        _make_clean_ann(a, keep_text=True)
        for a in annotations_all if a["ignore"] == 0
    ]
    coco_supervised = {
        "images": images,
        "annotations": annotations_supervised,
        "categories": [{"id": 1, "name": "text"}]
    }
    with open(out_json_path, "w") as f:
        json.dump(coco_supervised, f)

    n_ignored = len(annotations_all) - len(annotations_supervised)
    print(f"Scritto {out_json_path}")
    print(f"  -> {len(images)} immagini, {len(annotations_supervised)} ann valide, {n_ignored} ignored")
    print(f"Scritto {out_gt_source_path}")
    print(f"  -> {len(annotations_all)} annotazioni totali (ignored incluse)")
    return images, annotations_all, annotations_supervised


def make_semi_splits(images, annotations_all, label_ratio, out_dir,
                     split_name="train_37voc", seed=42):
    """
    Genera i JSON labeled/unlabeled per il training semi-supervised.
    label_ratio: frazione di immagini labeled (es. 0.10 per 10%)

    Regola campi per tipo di file:
      labeled   -> mantiene 'text' (supervisione esplicita, ground truth disponibile)
      unlabeled -> rimuove 'text' (non disponibile nello scenario semi-supervised)
    Entrambi: rimuovono 'ignore', escludono annotazioni con ignore==1.
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

    ratio_str = str(int(label_ratio * 100)) if label_ratio * 100 == int(label_ratio * 100) \
                else str(label_ratio)

    labeled_path   = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    labeled_data   = build_split(labeled_ids,   keep_text=True)
    unlabeled_data = build_split(unlabeled_ids, keep_text=False)

    with open(labeled_path, "w") as f:
        json.dump(labeled_data, f)
    with open(unlabeled_path, "w") as f:
        json.dump(unlabeled_data, f)

    print(f"  labeled   -> {labeled_path}")
    print(f"              {len(labeled_ids)} img, {len(labeled_data['annotations'])} ann")
    print(f"  unlabeled -> {unlabeled_path}")
    print(f"              {len(unlabeled_ids)} img, {len(unlabeled_data['annotations'])} ann")


if __name__ == "__main__":
    BASE = "datasets/hiertext"

    # -- 1. TEST / VALIDATION --------------------------------------------------
    print("\n[1/3] Conversione validation -> test.json + test_gt_source.json")
    convert(
        jsonl_path         = f"{BASE}/validation.jsonl",
        out_json_path      = f"{BASE}/test.json",
        out_gt_source_path = f"{BASE}/test_gt_source.json",
    )

    # -- 2. TRAINING COMPLETO --------------------------------------------------
    print("\n[2/3] Conversione train -> train_37voc.json + train_gt_source.json")
    images, annotations_all, annotations_supervised = convert(
        jsonl_path         = f"{BASE}/train.jsonl",
        out_json_path      = f"{BASE}/train_37voc.json",
        out_gt_source_path = f"{BASE}/train_gt_source.json",
    )

    # -- 3. SPLIT SEMI-SUPERVISED ----------------------------------------------
    print("\n[3/3] Generazione split semi-supervised")
    for ratio in [0.05, 0.10, 0.20]:
        print(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)

    # -- VERIFICA FINALE -------------------------------------------------------
    print("\n[VERIFICA] Controllo coerenza file generati...")
    import numpy as np

    checks = [
        (f"{BASE}/train_37voc.json",             True,  "train_37voc (supervised)"),
        (f"{BASE}/train_37voc_10_labeled.json",  True,  "10% labeled"),
        (f"{BASE}/train_37voc_10_unlabeled.json",False, "10% unlabeled"),
    ]
    for path, expect_text, label in checks:
        with open(path) as f:
            d = json.load(f)
        anns = d["annotations"]
        recs  = np.array([a["rec"] for a in anns])
        bad   = int((recs > PAD_TOKEN).sum())
        has_text  = all("text" in a for a in anns)
        has_ignore = any("ignore" in a for a in anns)
        status = []
        if bad > 0:              status.append(f"ERRORE: {bad} token rec > {PAD_TOKEN}")
        if has_ignore:           status.append("ERRORE: campo 'ignore' presente")
        if expect_text and not has_text:  status.append("ERRORE: 'text' mancante")
        if not expect_text and has_text:  status.append("WARN: 'text' presente (inatteso)")
        result = "OK" if not status else " | ".join(status)
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  -> {result}")
