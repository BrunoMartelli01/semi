#!/usr/bin/env python3
"""
convert_hiertext.py
-------------------
Converte le annotazioni raw di HierText (formato Google) nel formato
COCO-like usato da SemiETS, con bezier_pts calcolati OFFLINE e
validazione completa dei ring prima di scrivere il JSON.

Utilizzo:
    python datasets/hiertext/convert_hiertext.py \
        --input  /path/to/train.jsonl \
        --output datasets/hiertext/train_96voc.json \
        --voc    96 \
        --split  train

Il formato bezier_pts atteso da text.py e:
    [x0_top, y0_top, x1_top, y1_top, x2_top, y2_top, x3_top, y3_top,
     x0_bot, y0_bot, x1_bot, y1_bot, x2_bot, y2_bot, x3_bot, y3_bot]
ovvero una lista piatta di 16 float: curva superiore (4 cp) seguita da
curva inferiore (4 cp), ognuna nel verso "da sinistra a destra".

Come fa CTW1500: i punti vengono generati a partire dai poligoni della
bounding box della parola. Per ogni istanza:
  - top = punti che formano il bordo superiore (ordinati da sx a dx)
  - bot = punti che formano il bordo inferiore (ordinati da dx a sx)
  - fit cubica di Bezier su top e su bot separatamente via least-squares
  - validazione del ring: top (sx->dx) + bot (dx->sx) = anello chiuso
  - in caso di fallimento si usa il convex_hull del poligono originale
    come fallback deterministico

Formati di input supportati:
  - .json / .jsonl (inizia con '{') : JSON completo {"annotations": [...]}
  - .jsonl (vero JSONL, inizia con altro) : un oggetto JSON per riga
  - .jsonl.gz / .gz : versione compressa
  Lo script detecta automaticamente il formato dal primo carattere.

Vocabolari supportati:
  - 37  : a-z + cifre + illeggibile (indice 36)
  - 96  : ASCII 32-127 (95 char) + illeggibile (indice 95)
"""

import argparse
import json
import gzip
import os
import logging
from pathlib import Path

import numpy as np
from shapely.geometry import LinearRing, Polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabolari
# ---------------------------------------------------------------------------
CHARS_37 = list("abcdefghijklmnopqrstuvwxyz0123456789")
CHARS_96 = [chr(c) for c in range(32, 127)]


def build_char2idx(voc_size):
    if voc_size == 37:
        c2i = {c: i for i, c in enumerate(CHARS_37)}
        illegible_idx = 36
    elif voc_size == 96:
        c2i = {c: i for i, c in enumerate(CHARS_96)}
        illegible_idx = 95
    else:
        raise ValueError(f"voc_size deve essere 37 o 96, ricevuto {voc_size}")
    return c2i, illegible_idx


def text_to_rec(text, c2i, illegible_idx, max_len=25):
    """Converte stringa in lista di indici (max_len elementi, padding con illegible_idx)."""
    use_lower = (len(c2i) <= 36)
    rec = []
    for ch in (text.lower() if use_lower else text):
        rec.append(c2i.get(ch, illegible_idx))
        if len(rec) == max_len:
            break
    while len(rec) < max_len:
        rec.append(illegible_idx)
    return rec


# ---------------------------------------------------------------------------
# Fit cubica di Bezier (least-squares) su una sequenza di punti 2-D
# ---------------------------------------------------------------------------
def _bernstein_matrix(n_pts):
    """Matrice di Bernstein (n_pts x 4) per una curva cubica con t in [0,1]."""
    t = np.linspace(0.0, 1.0, n_pts)
    B = np.column_stack([
        (1 - t) ** 3,
        3 * t * (1 - t) ** 2,
        3 * t ** 2 * (1 - t),
        t ** 3,
    ])
    return B


def fit_cubic_bezier(pts):
    """
    Ritorna 4 punti di controllo (shape 4x2) che approssimano pts (Nx2)
    con una cubica di Bezier vincolando P0=pts[0] e P3=pts[-1].
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 2:
        raise ValueError("Almeno 2 punti richiesti per il fit")
    if n == 2:
        p0, p3 = pts[0], pts[1]
        return np.array([p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3])

    B = _bernstein_matrix(n)
    P0, P3 = pts[0], pts[-1]
    rhs = pts - np.outer(B[:, 0], P0) - np.outer(B[:, 3], P3)
    B_mid = B[:, 1:3]
    result, _, _, _ = np.linalg.lstsq(B_mid, rhs, rcond=None)
    P1, P2 = result[0], result[1]
    return np.array([P0, P1, P2, P3])


def sample_bezier(cp, num_pts):
    """Campiona num_pts punti su una cubica di Bezier dati i 4 cp (4x2)."""
    B = _bernstein_matrix(num_pts)
    return B @ cp


# ---------------------------------------------------------------------------
# Estrazione bordi superiore / inferiore con PCA
# ---------------------------------------------------------------------------
def split_top_bottom(pts):
    """
    Divide i punti del poligono in bordo superiore e inferiore usando PCA.

    Convenzione di orientamento del ring:
      - top_pts : ordinati sx -> dx lungo l'asse principale
      - bot_pts : ordinati dx -> sx lungo l'asse principale
    In questo modo top+bot formano un anello chiuso senza incroci.

    Questo e il motivo per cui CTW1500 non genera mai self-intersection:
    usa la direzione reale del testo (asse principale via SVD) invece
    dell'asse y fisso, che fallisce per testi curvi o ruotati.
    """
    pts = np.asarray(pts, dtype=float)
    _, unique_idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]

    if len(pts) < 4:
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        return (np.array([[xmin, ymin], [xmax, ymin]]),   # top: sx->dx
                np.array([[xmax, ymax], [xmin, ymax]]))   # bot: dx->sx

    cx, cy = pts.mean(axis=0)
    centered = pts - np.array([cx, cy])
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    main_dir = Vt[0]  # direzione principale
    perp_dir = Vt[1]  # perpendicolare

    proj_main = centered @ main_dir
    proj_perp = centered @ perp_dir

    top_mask = proj_perp < 0
    bot_mask = ~top_mask

    if top_mask.sum() < 2 or bot_mask.sum() < 2:
        try:
            hull_pts = np.array(Polygon(pts).convex_hull.exterior.coords)[:-1]
            return split_top_bottom(hull_pts)
        except Exception:
            xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
            xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
            return (np.array([[xmin, ymin], [xmax, ymin]]),
                    np.array([[xmax, ymax], [xmin, ymax]]))

    # top: sx -> dx (proj_main crescente)
    top_pts = pts[top_mask][np.argsort(proj_main[top_mask])]
    # bot: dx -> sx (proj_main decrescente) per chiudere il ring
    bot_pts = pts[bot_mask][np.argsort(-proj_main[bot_mask])]
    return top_pts, bot_pts


# ---------------------------------------------------------------------------
# Validazione del ring generato dai bezier_pts
# ---------------------------------------------------------------------------
def validate_ring(cp_top, cp_bot, num_pts=25):
    """
    Verifica che il ring generato da cp_top e cp_bot sia un LinearRing
    semplice (nessuna self-intersection).

    Convenzione:
      cp_top campiona da sx a dx
      cp_bot campiona da dx a sx  (gia nella direzione giusta dal fit)
    Il ring e: top_sampled + bot_sampled  (NO inversione di bot)

    Ritorna (is_valid: bool, ring_pts: ndarray (2*num_pts, 2))
    """
    top_sampled = sample_bezier(cp_top, num_pts)   # sx -> dx
    bot_sampled = sample_bezier(cp_bot, num_pts)   # dx -> sx (gia corretto)
    ring_pts = np.vstack([top_sampled, bot_sampled])  # anello chiuso
    try:
        return LinearRing(ring_pts).is_simple, ring_pts
    except Exception:
        return False, ring_pts


def bezier_from_polygon(vertices, num_pts=25):
    """
    Calcola bezier_pts (16 float) da un poligono con validazione a 3 livelli:
      1. Poligono originale + split PCA
      2. Convex hull
      3. Bounding box rettangolare (sempre valido)
    Ritorna (bezier_pts_list | None, is_valid_bool).
    """
    pts = np.array(vertices, dtype=float)

    def _try(pts_in):
        top, bot = split_top_bottom(pts_in)
        cp_top = fit_cubic_bezier(top)
        cp_bot = fit_cubic_bezier(bot)
        ok, _ = validate_ring(cp_top, cp_bot, num_pts)
        return cp_top, cp_bot, ok

    cp_top, cp_bot, valid = _try(pts)

    if not valid:
        try:
            hull_pts = np.array(Polygon(pts).convex_hull.exterior.coords)[:-1]
            cp_top, cp_bot, valid = _try(hull_pts)
        except Exception:
            valid = False

    if not valid:
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        bbox_pts = np.array([[xmin, ymin], [xmax, ymin],
                             [xmax, ymax], [xmin, ymax]])
        cp_top, cp_bot, valid = _try(bbox_pts)
        if not valid:
            # Geometricamente impossibile per una bbox: bug numerico estremo
            log.warning("Fallback bbox invalido: poligono degenere, scartato")
            return None, False

    return cp_top.flatten().tolist() + cp_bot.flatten().tolist(), valid


# ---------------------------------------------------------------------------
# Parsing annotazioni HierText
# ---------------------------------------------------------------------------
def _read_jsonl_lines(path_str):
    """Legge un file testo riga per riga come JSONL (un oggetto per riga)."""
    opener = (gzip.open(path_str, "rt", encoding="utf-8")
              if path_str.endswith(".gz")
              else open(path_str, "r", encoding="utf-8"))
    results = []
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def parse_hiertext(input_path):
    """
    Legge annotazioni HierText con auto-detection del formato.

    Google distribuisce i file come .jsonl ma in realta sono JSON completi
    con struttura {"annotations": [...]} -- NON vero JSONL (un oggetto/riga).
    Questa funzione gestisce entrambi i casi leggendo il primo carattere:

      Caso A - primo char '{': JSON completo -> json.load -> data["annotations"]
      Caso B - altro:          vero JSONL    -> lettura riga per riga
      Caso C - .gz:            gzip + stesso auto-detect
    """
    input_path = Path(input_path)
    path_str = str(input_path)

    def _first_char(fh):
        for ch in fh.read(4096):
            if ch.strip():
                return ch
        return ""

    if path_str.endswith(".gz"):
        with gzip.open(path_str, "rt", encoding="utf-8") as f:
            first_char = _first_char(f)
    else:
        with open(path_str, "r", encoding="utf-8") as f:
            first_char = _first_char(f)

    log.info(f"Formato rilevato: primo char='{first_char}' ({input_path.name})")

    if first_char == "{":
        log.info("Modalita: JSON completo (formato Google ufficiale)")
        if path_str.endswith(".gz"):
            with gzip.open(path_str, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        records = data.get("annotations", [])
        if not records:
            records = [data]
        log.info(f"  Record trovati: {len(records)}")
        return records
    else:
        log.info("Modalita: JSONL (un oggetto per riga)")
        records = _read_jsonl_lines(path_str)
        log.info(f"  Record trovati: {len(records)}")
        return records


# ---------------------------------------------------------------------------
# Conversione principale
# ---------------------------------------------------------------------------
def convert(input_path, output_path, voc_size=96, split="train",
            num_pts=25, max_rec_len=25):
    """Converte HierText -> formato COCO-like con bezier_pts validati."""
    c2i, illegible_idx = build_char2idx(voc_size)
    records = parse_hiertext(input_path)

    images, annotations = [], []
    ann_id = img_id = 1
    stats = {"total": 0, "valid": 0, "discarded": 0, "illegible": 0}

    for record in records:
        file_name = record.get("image_id", record.get("file_name", f"img_{img_id:06d}.jpg"))
        if not file_name.endswith((".jpg", ".jpeg", ".png")):
            file_name += ".jpg"

        height = record.get("image_height", record.get("height", 0))
        width = record.get("image_width", record.get("width", 0))
        images.append({"id": img_id, "file_name": file_name,
                       "height": height, "width": width})

        for para in record.get("paragraphs", []):
            for line in para.get("lines", []):
                for word in line.get("words", []):
                    stats["total"] += 1
                    vertices = word.get("vertices", [])
                    if len(vertices) < 3:
                        stats["discarded"] += 1
                        continue

                    text_str = word.get("text", "")
                    is_legible = word.get("legible", True)

                    flat_seg = [coord for v in vertices for coord in (float(v[0]), float(v[1]))]
                    xs, ys = [v[0] for v in vertices], [v[1] for v in vertices]
                    xmin, ymin = min(xs), min(ys)
                    bbox = [xmin, ymin, max(xs) - xmin, max(ys) - ymin]
                    area = bbox[2] * bbox[3]

                    if not is_legible or not text_str:
                        rec = [illegible_idx] * max_rec_len
                        stats["illegible"] += 1
                    else:
                        rec = text_to_rec(text_str, c2i, illegible_idx, max_rec_len)

                    bezier_pts, valid = bezier_from_polygon(vertices, num_pts)
                    if bezier_pts is None:
                        stats["discarded"] += 1
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": [flat_seg],
                        "bbox": bbox,
                        "area": float(area),
                        "iscrowd": 0,
                        "bezier_pts": bezier_pts,
                        "rec": rec,
                    })
                    ann_id += 1
                    stats["valid"] += 1

        img_id += 1

    coco_out = {
        "info": {"description": f"HierText {split}", "voc_size": voc_size, "num_pts": num_pts},
        "licenses": [],
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
        "images": images,
        "annotations": annotations,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_out, f)

    log.info("=" * 60)
    log.info(f"Output: {output_path}")
    log.info(f"  Immagini  : {len(images)}")
    log.info(f"  Ann valide: {stats['valid']} / {stats['total']}")
    log.info(f"  Illeggib. : {stats['illegible']}")
    log.info(f"  Scartate  : {stats['discarded']}")
    log.info("=" * 60)
    return coco_out


# ---------------------------------------------------------------------------
# Validazione post-scrittura
# ---------------------------------------------------------------------------
def validate_output(output_path, num_pts=25, sample_size=500):
    """
    Carica il JSON e verifica i ring su un campione.
    Stampa la percentuale di ring validi / invalidi.
    """
    import random
    log.info(f"Validazione: {output_path}")
    with open(output_path) as f:
        data = json.load(f)
    anns = data["annotations"]
    if not anns:
        log.warning("Nessuna annotazione.")
        return

    sample = random.sample(anns, min(sample_size, len(anns)))
    n_valid = n_invalid = 0
    bad_ids = []

    for ann in sample:
        bpts = np.array(ann["bezier_pts"]).reshape(8, 2)
        ok, _ = validate_ring(bpts[:4], bpts[4:], num_pts)
        if ok:
            n_valid += 1
        else:
            n_invalid += 1
            bad_ids.append(ann["id"])

    tot = len(sample)
    log.info(f"  Campione: {tot} | Validi: {n_valid} ({100*n_valid/tot:.1f}%) | Invalidi: {n_invalid}")
    if n_invalid:
        log.warning(f"  Primi ID invalidi: {bad_ids[:10]}")
    else:
        log.info("  Tutti i ring sono validi (no self-intersection)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converti HierText -> COCO con bezier_pts validati"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voc", type=int, default=96, choices=[37, 96])
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_pts", type=int, default=25)
    parser.add_argument("--max_rec_len", type=int, default=25)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate_sample", type=int, default=500)
    args = parser.parse_args()

    convert(args.input, args.output, args.voc, args.split,
            args.num_pts, args.max_rec_len)

    if args.validate:
        validate_output(args.output, args.num_pts, args.validate_sample)
