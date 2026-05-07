"""
debug_hiertext.py
=================
Script di debug per la conversione HierText -> COCO.

Per ogni immagine campione mostra affiancate:
  - PRIMA : bounding box estratte dal JSON sorgente HierText (JSONL)
  - DOPO  : bounding box presenti nel JSON COCO convertito

In entrambi i casi ogni box viene annotata con il testo che dovrebbe contenere.

I campioni vengono scelti tra le immagini presenti nel file --coco (JSON
convertito), in modo da garantire che esistano sia le annotazioni DOPO che
la corrispondenza con il JSONL originale.

Uso:
    python tools/debug_hiertext.py \
        --jsonl  datasets/hiertext/validation.jsonl \
        --coco   datasets/hiertext/test_gt_source.json \
        --images datasets/hiertext/images/validation \
        --out    debug_output \
        --n      5

Se --images non e' disponibile le visualizzazioni vengono prodotte su sfondo
grigio usando le dimensioni presenti nel JSON.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

# -------------------------------------------------
#  Colori
# -------------------------------------------------
COLOR_BEFORE = (0, 200, 0)    # verde   - boxes PRIMA
COLOR_AFTER  = (0, 100, 255)  # arancio - boxes DOPO
TEXT_COLOR   = (255, 255, 255)
TEXT_BG      = (0, 0, 0)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.4
FONT_THICK   = 1


# -------------------------------------------------
#  Helpers visivi
# -------------------------------------------------

def load_image(img_path: str, width: int, height: int) -> np.ndarray:
    """Carica l'immagine dal disco, oppure crea un canvas grigio."""
    if img_path and os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img
    h = max(height, 1) if height > 0 else 600
    w = max(width,  1) if width  > 0 else 800
    return np.full((h, w, 3), 80, dtype=np.uint8)


def draw_text_with_bg(img, text, x, y, color=TEXT_COLOR, bg=TEXT_BG):
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    x = max(0, min(x, img.shape[1] - tw - 2))
    y = max(th + baseline, min(y, img.shape[0] - baseline))
    cv2.rectangle(img, (x - 1, y - th - baseline - 1),
                  (x + tw + 1, y + baseline + 1), bg, -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, FONT_THICK,
                cv2.LINE_AA)


def draw_poly_box(img, pts, color, text=""):
    if len(pts) < 2:
        return
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=1)
    if text:
        x = int(pts_arr[:, 0].min())
        y = int(pts_arr[:, 1].min()) - 3
        draw_text_with_bg(img, text[:30], x, y, color=color)


def draw_bbox(img, bbox, color, text=""):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    if text:
        draw_text_with_bg(img, text[:30], x, max(0, y - 3), color=color)


def label_panel(img, title):
    bar = np.zeros((22, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (5, 15), FONT, 0.5, (220, 220, 220), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


# -------------------------------------------------
#  Parsing JSONL HierText
# -------------------------------------------------

def parse_jsonl(jsonl_path: str) -> dict:
    """
    Restituisce dict  image_id_str -> {
        'image_id', 'image_width', 'image_height',
        'words': [{'vertices': [[x,y],...], 'text': str, 'legible': bool}]
    }
    Supporta sia JSON con chiave 'annotations' che JSONL puro.
    """
    with open(jsonl_path, "r") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        entries = data["annotations"] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    samples = {}
    for sample in entries:
        img_id = sample.get("image_id", "")
        words = []
        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if verts and isinstance(verts[0], dict):
                        pts = [[v["x"], v["y"]] for v in verts]
                    else:
                        pts = [[v[0], v[1]] for v in verts]
                    words.append({
                        "vertices": pts,
                        "text":     word.get("text", ""),
                        "legible":  word.get("legible", True),
                    })
        samples[img_id] = {
            "image_id":     img_id,
            "image_width":  sample.get("image_width", 0),
            "image_height": sample.get("image_height", 0),
            "words":        words,
        }
    return samples


# -------------------------------------------------
#  Parsing COCO JSON convertito
# -------------------------------------------------

def parse_coco(coco_path: str):
    """
    Restituisce:
      - images_meta : dict  coco_id(int) -> {file_name, width, height}
      - anns_by_img : dict  coco_id(int) -> [ann, ...]
    """
    with open(coco_path) as f:
        data = json.load(f)

    images_meta = {img["id"]: img for img in data.get("images", [])}

    anns_by_img = {}
    for ann in data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    return images_meta, anns_by_img


# -------------------------------------------------
#  Costruzione pannelli
# -------------------------------------------------

def build_before_panel(img, words):
    out = img.copy()
    for w in words:
        text = w["text"] if w["legible"] else "[illegible]"
        draw_poly_box(out, w["vertices"], COLOR_BEFORE, text)
    return label_panel(out, "PRIMA  (HierText JSONL)")


def build_after_panel(img, anns):
    out = img.copy()
    for ann in anns:
        text = ann.get("text", "")
        seg  = ann.get("segmentation", [[]])
        seg  = seg[0] if seg else []
        if seg and len(seg) >= 6:
            pts = [[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)]
            draw_poly_box(out, pts, COLOR_AFTER, text)
        else:
            draw_bbox(out, ann["bbox"], COLOR_AFTER, text)
    return label_panel(out, "DOPO   (COCO JSON convertito)")


# -------------------------------------------------
#  Ricerca immagine su disco
# -------------------------------------------------

def find_image_path(images_dir: str, file_name: str) -> str:
    """
    Cerca l'immagine provando in ordine:
      1. images_dir / file_name           (nome esatto dal COCO)
      2. images_dir / basename(file_name) (solo il nome, senza subdir)
      3. images_dir / stem + .jpg/.png    (cambia estensione)
    """
    if not images_dir:
        return ""

    candidates = [
        os.path.join(images_dir, file_name),
        os.path.join(images_dir, os.path.basename(file_name)),
    ]
    stem = os.path.splitext(os.path.basename(file_name))[0]
    for ext in (".jpg", ".jpeg", ".png"):
        candidates.append(os.path.join(images_dir, stem + ext))

    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


# -------------------------------------------------
#  Mappa file_name COCO -> image_id JSONL
# -------------------------------------------------

def coco_fname_to_jsonl_id(file_name: str) -> str:
    """
    Il file_name nel COCO e' qualcosa come '0a1b2c3d.jpg'.
    L'image_id nel JSONL e' lo stem senza estensione: '0a1b2c3d'.
    """
    return os.path.splitext(os.path.basename(file_name))[0]


# -------------------------------------------------
#  Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug HierText box conversion")
    parser.add_argument("--jsonl",  required=True,
                        help="Path al file HierText JSONL (es. validation.jsonl)")
    parser.add_argument("--coco",   required=True,
                        help="Path al JSON COCO convertito (es. test_gt_source.json)")
    parser.add_argument("--images", default="",
                        help="Cartella con le immagini (opzionale)")
    parser.add_argument("--out",    default="debug_output",
                        help="Cartella di output")
    parser.add_argument("--n",      type=int, default=5,
                        help="Numero di campioni da visualizzare")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Caricamento JSONL: {args.jsonl}")
    before_data = parse_jsonl(args.jsonl)
    print(f"      {len(before_data)} immagini nel JSONL")

    print(f"[2/3] Caricamento COCO:  {args.coco}")
    coco_meta, coco_anns = parse_coco(args.coco)
    print(f"      {len(coco_meta)} immagini nel COCO")

    # ---- campiona dagli image_id del COCO ----
    # Considera solo le entry COCO che hanno anche corrispondenza nel JSONL
    coco_ids_all = list(coco_meta.keys())
    coco_ids_valid = [
        cid for cid in coco_ids_all
        if coco_fname_to_jsonl_id(coco_meta[cid]["file_name"]) in before_data
    ]
    if not coco_ids_valid:
        print("WARN: nessuna corrispondenza trovata tra COCO e JSONL. "
              "Uso tutti gli id COCO.")
        coco_ids_valid = coco_ids_all

    random.seed(args.seed)
    sample_coco_ids = random.sample(coco_ids_valid, min(args.n, len(coco_ids_valid)))
    print(f"[3/3] Generazione {len(sample_coco_ids)} visualizzazioni -> {args.out}/")

    for coco_id in sample_coco_ids:
        meta        = coco_meta[coco_id]
        file_name   = meta["file_name"]
        jsonl_id    = coco_fname_to_jsonl_id(file_name)
        after_anns  = coco_anns.get(coco_id, [])

        # dimensioni: preferisce quelle nel COCO, poi nel JSONL
        w_img = meta.get("width",  0)
        h_img = meta.get("height", 0)
        if (w_img == 0 or h_img == 0) and jsonl_id in before_data:
            w_img = before_data[jsonl_id]["image_width"]
            h_img = before_data[jsonl_id]["image_height"]

        # carica immagine usando il file_name dal COCO
        img_path = find_image_path(args.images, file_name)
        base_img = load_image(img_path, w_img, h_img)

        # words dal JSONL
        words = before_data[jsonl_id]["words"] if jsonl_id in before_data else []

        # costruisci pannelli
        panel_before = build_before_panel(base_img, words)
        panel_after  = build_after_panel(base_img, after_anns)

        # allinea altezze
        h_b = panel_before.shape[0]
        h_a = panel_after.shape[0]
        if h_b != h_a:
            th = max(h_b, h_a)
            panel_before = cv2.copyMakeBorder(
                panel_before, 0, th - h_b, 0, 0,
                cv2.BORDER_CONSTANT, value=(50, 50, 50))
            panel_after  = cv2.copyMakeBorder(
                panel_after,  0, th - h_a, 0, 0,
                cv2.BORDER_CONSTANT, value=(50, 50, 50))

        sep      = np.full((panel_before.shape[0], 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([panel_before, sep, panel_after])

        # legenda
        legend = np.zeros((28, combined.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(legend, (5, 6), (20, 20), COLOR_BEFORE, -1)
        cv2.putText(legend, "PRIMA (JSONL originale)", (25, 17),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cx = combined.shape[1] // 2
        cv2.rectangle(legend, (cx + 5, 6), (cx + 20, 20), COLOR_AFTER, -1)
        cv2.putText(legend, "DOPO (COCO convertito)", (cx + 25, 17),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        combined = np.vstack([combined, legend])

        # barra info
        info = np.zeros((20, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            info,
            f"id={jsonl_id}  |  boxes prima={len(words)}  dopo={len(after_anns)}  "
            f"|  img={'OK' if img_path else 'canvas'}",
            (5, 14), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        combined = np.vstack([info, combined])

        out_path = os.path.join(args.out, f"debug_{jsonl_id}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  -> {out_path}  "
              f"(before={len(words)}, after={len(after_anns)}, "
              f"img={'OK' if img_path else 'canvas grigio'})")

    print("\nDone.")


if __name__ == "__main__":
    main()
