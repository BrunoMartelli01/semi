#!/usr/bin/env python3
"""
generate_hiertext_vocab.py
==========================
Genera i due file di vocabolario per HierText nel formato atteso da
text_evaluation_all.py (identico a quello usato per totaltext / ctw1500):

  datasets/hiertext/weak_voc_new.txt       -- una parola per riga (case-sensitive)
  datasets/hiertext/weak_voc_pair_list.txt -- "PAROLA_UPPER parola_originale" per riga

Input:  datasets/hiertext/test_gt_source.json  (prodotto da convert_hiertext.py)
        (opzionalmente anche train_gt_source.json, ma sconsigliato per l'evaluation)

Uso:
    python tools/generate_hiertext_vocab.py
oppure specificando percorsi diversi:
    python tools/generate_hiertext_vocab.py \
        --gt_sources datasets/hiertext/test_gt_source.json \
        --out_dir    datasets/hiertext
"""

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Caratteri validi: stesso alfabeto 96-voc usato da SemiETS / DeepSolo
# (ASCII stampabili 32..126, 95 simboli)
# ---------------------------------------------------------------------------
VALID_CHARS = set(chr(i) for i in range(32, 127))


def is_valid_word(word: str) -> bool:
    """
    Filtra:
      - parole vuote
      - parole con spazi interni (rompono line.split(' ')[0] in pair_list.txt)
      - caratteri non-ASCII o fuori dal vocabolario 96-char
    """
    if not word:
        return False
    if ' ' in word:
        return False
    return all(c in VALID_CHARS for c in word)


def collect_words(gt_source_paths):
    """
    Raccoglie tutte le parole uniche (originali, case-sensitive) dai file
    *_gt_source.json.  Solo le annotazioni non-ignored (ignore==0 o assente)
    vengono considerate.

    Restituisce la lista ordinata lessicograficamente (case-insensitive come
    primary key, originale come secondary key per stabilita').
    """
    words = set()
    for path in gt_source_paths:
        print(f"  Lettura {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            if ann.get("ignore", 0) == 1:
                continue
            text = ann.get("text", "").strip()
            if is_valid_word(text):
                words.add(text)
    return sorted(words, key=lambda w: (w.upper(), w))


def write_vocab_files(words, out_dir):
    """
    Scrive i due file nel formato letto da text_evaluation_all.py:

    weak_voc_new.txt:
        Una parola per riga (esattamente come compare nel GT, case-sensitive).

    weak_voc_pair_list.txt:
        "PAROLA_UPPERCASE parola_originale"
        Il codice in sort_detection() fa:
            word = line.split(' ')[0].upper()
            word_gt = line[len(word) + 1:]
            pairs[word] = word_gt
        Quindi la riga deve essere  "UPPER original"  (un singolo spazio).

    NOTA: poiche' le parole con spazi interni sono gia' filtrate da
    is_valid_word(), il formato e' sempre non-ambiguo.
    """
    os.makedirs(out_dir, exist_ok=True)

    voc_path  = os.path.join(out_dir, "weak_voc_new.txt")
    pair_path = os.path.join(out_dir, "weak_voc_pair_list.txt")

    with open(voc_path, "w", encoding="utf-8") as fv, \
         open(pair_path, "w", encoding="utf-8") as fp:
        for word in words:
            fv.write(word + "\n")
            fp.write(word.upper() + " " + word + "\n")

    print(f"  Scritto {voc_path}  ({len(words)} parole)")
    print(f"  Scritto {pair_path} ({len(words)} righe)")


def main():
    parser = argparse.ArgumentParser(
        description="Genera weak_voc_new.txt e weak_voc_pair_list.txt per HierText."
    )
    parser.add_argument(
        "--gt_sources",
        nargs="+",
        default=[
            # Solo test: il weak lexicon serve per l'evaluation, non per il training.
            # Usare le parole di train gonfia artificialmente le metriche e
            # rende find_match_word ordini di grandezza piu' lenta.
            "datasets/hiertext/test_gt_source.json",
        ],
        help=(
            "Uno o piu' file *_gt_source.json da cui estrarre le parole. "
            "Default: solo test_gt_source.json (raccomandato)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="datasets/hiertext",
        help="Directory di output (default: datasets/hiertext).",
    )
    args = parser.parse_args()

    print("\n[1/2] Raccolta parole dai GT source ...")
    words = collect_words(args.gt_sources)
    print(f"  Parole uniche trovate: {len(words)}")

    print("\n[2/2] Scrittura file vocabolario ...")
    write_vocab_files(words, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()