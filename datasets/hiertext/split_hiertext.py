#!/usr/bin/env python3
"""
split_hiertext.py
-----------------
Genera le coppie labeled/unlabeled a partire da train_96voc.json
(o qualsiasi JSON COCO-like prodotto da convert_hiertext.py).

Convenzione identica a CTW1500:
  train_96voc_{R}_labeled.json   -- R% delle immagini, con tutte le annotazioni
  train_96voc_{R}_unlabeled.json -- restante (100-R)% delle immagini,
                                    rec azzerato (lista di illegible_idx)

Lo split avviene a livello di IMMAGINE (non di annotazione) e usa un
seed fisso per garantire riproducibilita.

Utilizzo:
    python datasets/hiertext/split_hiertext.py \
        --input  datasets/hiertext/train_96voc.json \
        --ratios 1 2 5 10 \
        --seed   42

Per generare anche 0.5%:
    --ratios 0.5 1 2 5 10

I file vengono scritti nella stessa cartella dell'input.
"""

import argparse
import json
import os
import random
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    size_mb = os.path.getsize(path) / 1e6
    log.info(f"  -> {path}  ({size_mb:.1f} MB, {len(data['images'])} img, {len(data['annotations'])} ann)")


def split(input_path, ratios, seed=42):
    """
    Genera coppie labeled/unlabeled per ogni ratio in `ratios`.

    Parametri
    ---------
    input_path : str  -- path al JSON COCO-like completo
    ratios     : list of float  -- percentuali labeled (es. [1, 5, 10])
    seed       : int  -- seed per riproducibilita
    """
    data = load_coco(input_path)
    out_dir = Path(input_path).parent
    # Deduco il prefisso dal nome file (es. "train_96voc" da "train_96voc.json")
    stem = Path(input_path).stem  # es. "train_96voc"

    images = data["images"]
    annotations = data["annotations"]
    info = data.get("info", {})
    licenses = data.get("licenses", [])
    categories = data.get("categories", [])

    # Legge l'indice illegible dal campo info (default 95 per voc=96)
    voc_size = info.get("voc_size", 96)
    illegible_idx = voc_size - 1
    max_rec_len = len(annotations[0]["rec"]) if annotations else 25

    log.info(f"Dataset: {len(images)} immagini, {len(annotations)} annotazioni")
    log.info(f"voc_size={voc_size}, illegible_idx={illegible_idx}, rec_len={max_rec_len}")

    # Indice annotazioni per image_id
    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    # Shuffle riproducibile
    rng = random.Random(seed)
    img_ids = [img["id"] for img in images]
    rng.shuffle(img_ids)
    img_by_id = {img["id"]: img for img in images}

    n_total = len(img_ids)

    for ratio in ratios:
        # Numero di immagini labeled
        n_labeled = max(1, round(n_total * ratio / 100.0))
        labeled_ids = set(img_ids[:n_labeled])
        unlabeled_ids = set(img_ids[n_labeled:])

        # Formato nome file: 0.5 -> "0.5", 1 -> "1", 10 -> "10"
        ratio_str = str(ratio).rstrip("0").rstrip(".") if "." in str(ratio) else str(ratio)
        # Gestisci il caso ratio=0.5 -> "0.5"
        if float(ratio) != int(float(ratio)):
            ratio_str = str(ratio)
        else:
            ratio_str = str(int(float(ratio)))

        log.info(f"\nRatio {ratio}%: {n_labeled} labeled / {len(unlabeled_ids)} unlabeled")

        # ---- LABELED JSON ----
        # Immagini labeled con tutte le annotazioni originali
        labeled_images = [img_by_id[i] for i in img_ids[:n_labeled]]
        labeled_anns = []
        for img_id in labeled_ids:
            labeled_anns.extend(anns_by_img[img_id])

        labeled_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": labeled_images,
            "annotations": labeled_anns,
        }
        labeled_path = out_dir / f"{stem}_{ratio_str}_labeled.json"
        save_json(labeled_data, str(labeled_path))

        # ---- UNLABELED JSON ----
        # Immagini unlabeled: rec azzerato (illegible_idx * max_rec_len)
        # bezier_pts conservato (la geometria serve per il training semi-supervisionato)
        blank_rec = [illegible_idx] * max_rec_len
        unlabeled_images = [img_by_id[i] for i in img_ids[n_labeled:]]
        unlabeled_anns = []
        for img_id in unlabeled_ids:
            for ann in anns_by_img[img_id]:
                ann_copy = dict(ann)
                ann_copy["rec"] = blank_rec
                unlabeled_anns.append(ann_copy)

        unlabeled_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": unlabeled_images,
            "annotations": unlabeled_anns,
        }
        unlabeled_path = out_dir / f"{stem}_{ratio_str}_unlabeled.json"
        save_json(unlabeled_data, str(unlabeled_path))

    log.info("\nSplit completato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera split labeled/unlabeled da un JSON COCO-like HierText"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path al JSON completo (es. datasets/hiertext/train_96voc.json)"
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[1, 2, 5, 10],
        help="Percentuali di immagini labeled (default: 1 2 5 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed random per riproducibilita (default: 42)"
    )
    args = parser.parse_args()
    split(args.input, args.ratios, args.seed)
