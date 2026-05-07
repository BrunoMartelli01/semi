"""
debug_hiertext.py
=================
Script di debug per la conversione HierText → COCO.

Per ogni immagine campione mostra affiancate:
  - PRIMA : bounding box estratte dal JSON sorgente HierText (JSONL)
  - DOPO  : bounding box presenti nel JSON COCO convertito

In entrambi i casi ogni box viene annotata con il testo che dovrebbe contenere.

Uso:
    python tools/debug_hiertext.py \
        --jsonl  datasets/hiertext/validation.jsonl \
        --coco   datasets/hiertext/test.json \
        --images datasets/hiertext/images/validation \
        --out    debug_output \
        --n      5

Se --images non è disponibile (immagini non scaricate) le visualizzazioni
vengono prodotte su sfondo grigio usando le dimensioni dal JSON.
"""

import argparse
import json
import os
import random
import textwrap

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Colori
# ─────────────────────────────────────────────
COLOR_BEFORE = (0, 200, 0)    # verde  – boxes PRIMA
COLOR_AFTER  = (0, 100, 255)  # arancio – boxes DOPO
TEXT_COLOR   = (255, 255, 255)
TEXT_BG      = (0, 0, 0)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.4
FONT_THICK   = 1


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_image(img_path: str, width: int, height: int) -> np.ndarray:
    """Carica l'immagine dal disco, oppure crea un canvas grigio."""
    if img_path and os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img
    # fallback: canvas grigio
    h = height if height > 0 else 600
    w = width  if width  > 0 else 800
    canvas = np.full((h, w, 3), 80, dtype=np.uint8)
    return canvas


def draw_text_with_bg(img: np.ndarray, text: str, x: int, y: int,
                      color=TEXT_COLOR, bg=TEXT_BG):
    """Scrive testo con sfondo opaco per leggibilità."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    # clip alle dimensioni dell'immagine
    x = max(0, min(x, img.shape[1] - tw - 2))
    y = max(th + baseline, min(y, img.shape[0] - baseline))
    cv2.rectangle(img, (x - 1, y - th - baseline - 1),
                  (x + tw + 1, y + baseline + 1), bg, -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, FONT_THICK,
                cv2.LINE_AA)


def draw_poly_box(img: np.ndarray, pts, color, text: str = ""):
    """Disegna il poligono (lista di [x,y]) e il testo associato."""
    if len(pts) < 2:
        return
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=1)
    if text:
        x, y = int(pts_arr[:, 0].min()), int(pts_arr[:, 1].min()) - 3
        draw_text_with_bg(img, text[:30], x, y, color=color)


def draw_bbox(img: np.ndarray, bbox, color, text: str = ""):
    """Disegna un rettangolo COCO [x, y, w, h] e il testo associato."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    if text:
        draw_text_with_bg(img, text[:30], x, max(0, y - 3), color=color)


def label_panel(img: np.ndarray, title: str):
    """Aggiunge una barra nera in cima con il titolo del pannello."""
    bar = np.zeros((22, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (5, 15), FONT, 0.5, (220, 220, 220), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


# ─────────────────────────────────────────────
#  Parsing JSONL HierText
# ─────────────────────────────────────────────

def parse_jsonl(jsonl_path: str):
    """
    Restituisce dict  image_id → {
        'image_id', 'image_width', 'image_height',
        'words': [{'vertices': [...], 'text': str, 'legible': bool}]
    }
    Supporta sia un array JSON che JSONL (una riga = un oggetto).
    """
    samples = {}
    with open(jsonl_path, "r") as f:
        raw = f.read().strip()

    # prova prima come JSON array
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "annotations" in data:
            entries = data["annotations"]
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
    except json.JSONDecodeError:
        # JSONL: una riga per oggetto
        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    for sample in entries:
        img_id = sample.get("image_id", "")
        words = []
        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if isinstance(verts[0], dict) if verts else False:
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


# ─────────────────────────────────────────────
#  Parsing COCO JSON convertito
# ─────────────────────────────────────────────

def parse_coco(coco_path: str):
    """
    Restituisce:
      - images_meta: dict  image_id(int) → {'file_name', 'width', 'height'}
      - anns_by_img: dict  image_id(int) → lista di annotation dicts
    """
    with open(coco_path) as f:
        data = json.load(f)

    images_meta = {}
    for img in data.get("images", []):
        images_meta[img["id"]] = img

    anns_by_img = {}
    for ann in data.get("annotations", []):
        iid = ann["image_id"]
        anns_by_img.setdefault(iid, []).append(ann)

    # recupera testo da gt_source se presente (campo "text")
    return images_meta, anns_by_img


# ─────────────────────────────────────────────
#  Costruzione pannelli
# ─────────────────────────────────────────────

def build_before_panel(img: np.ndarray, words) -> np.ndarray:
    """Pannello PRIMA: poligoni dal JSONL originale + testo."""
    out = img.copy()
    for w in words:
        text  = w["text"] if w["legible"] else "[illegible]"
        draw_poly_box(out, w["vertices"], COLOR_BEFORE, text)
    return label_panel(out, "PRIMA  (HierText JSONL)")


def build_after_panel(img: np.ndarray, anns, coco_images_meta=None, img_id=None) -> np.ndarray:
    """
    Pannello DOPO: bbox COCO + testo.
    Il campo 'text' può non essere presente nel JSON clean (solo in gt_source).
    """
    out = img.copy()
    for ann in anns:
        text = ann.get("text", "")
        seg  = ann.get("segmentation", [[]])[0]
        if seg and len(seg) >= 6:
            pts = [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)]
            draw_poly_box(out, pts, COLOR_AFTER, text)
        else:
            draw_bbox(out, ann["bbox"], COLOR_AFTER, text)
    return label_panel(out, "DOPO   (COCO JSON convertito)")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug HierText box conversion")
    parser.add_argument("--jsonl",   required=True,
                        help="Path al file HierText JSONL (es. validation.jsonl)")
    parser.add_argument("--coco",    required=True,
                        help="Path al JSON COCO convertito (es. test.json o test_gt_source.json)")
    parser.add_argument("--images",  default="",
                        help="Cartella con le immagini (opzionale)")
    parser.add_argument("--out",     default="debug_output",
                        help="Cartella di output per le visualizzazioni")
    parser.add_argument("--n",       type=int, default=5,
                        help="Numero di immagini campione da visualizzare")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Caricamento JSONL: {args.jsonl}")
    before_data = parse_jsonl(args.jsonl)

    print(f"[2/3] Caricamento COCO:  {args.coco}")
    coco_meta, coco_anns = parse_coco(args.coco)

    # Mappa file_name → coco image_id
    fname_to_coco_id = {meta["file_name"]: cid
                        for cid, meta in coco_meta.items()}

    # Campiona N immagini
    all_ids = list(before_data.keys())
    random.seed(args.seed)
    sample_ids = random.sample(all_ids, min(args.n, len(all_ids)))

    print(f"[3/3] Generazione {len(sample_ids)} visualizzazioni → {args.out}/")
    for img_id_str in sample_ids:
        sample   = before_data[img_id_str]
        w_orig   = sample["image_width"]
        h_orig   = sample["image_height"]

        # trova corrispondenza nel COCO
        fname_jpg = img_id_str + ".jpg"
        fname_png = img_id_str + ".png"
        coco_id   = fname_to_coco_id.get(fname_jpg) or fname_to_coco_id.get(fname_png)

        # prova a caricare l'immagine
        img_path = ""
        if args.images:
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = os.path.join(args.images, img_id_str + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

        base_img = load_image(img_path, w_orig, h_orig)

        # ── pannello PRIMA ──
        panel_before = build_before_panel(base_img, sample["words"])

        # ── pannello DOPO ──
        after_anns = coco_anns.get(coco_id, []) if coco_id is not None else []
        panel_after = build_after_panel(base_img, after_anns)

        # ── affianca i due pannelli ──
        # adegua altezze se diverse (non dovrebbero esserlo)
        h_b, w_b = panel_before.shape[:2]
        h_a, w_a = panel_after.shape[:2]
        if h_b != h_a:
            target_h = max(h_b, h_a)
            panel_before = cv2.copyMakeBorder(panel_before, 0, target_h - h_b,
                                              0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))
            panel_after  = cv2.copyMakeBorder(panel_after,  0, target_h - h_a,
                                              0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))

        separator = np.full((panel_before.shape[0], 4, 3), 200, dtype=np.uint8)
        combined  = np.hstack([panel_before, separator, panel_after])

        # legenda
        legend_h = 28
        legend = np.zeros((legend_h, combined.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(legend, (5, 6), (20, 20), COLOR_BEFORE, -1)
        cv2.putText(legend, "PRIMA (JSONL originale)", (25, 17),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cx = combined.shape[1] // 2
        cv2.rectangle(legend, (cx + 5, 6), (cx + 20, 20), COLOR_AFTER, -1)
        cv2.putText(legend, "DOPO (COCO convertito)", (cx + 25, 17),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        combined = np.vstack([combined, legend])

        # info
        n_before = len(sample["words"])
        n_after  = len(after_anns)
        info_bar = np.zeros((20, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_bar,
                    f"id={img_id_str}  |  boxes prima={n_before}  dopo={n_after}  "
                    f"|  verde=PRIMA  arancio=DOPO",
                    (5, 14), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        combined = np.vstack([info_bar, combined])

        out_path = os.path.join(args.out, f"debug_{img_id_str}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  → {out_path}  (before={n_before} boxes, after={n_after} boxes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
