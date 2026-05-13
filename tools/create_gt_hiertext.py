import json
import math
import zipfile
import re
from collections import defaultdict

# Regex: almeno 6 coppie di interi (3 punti minimo per un poligono valido)
# poi ,#### e testo opzionale. Garantisce compatibilita' con rrc_evaluation_funcs.
VALID_LINE_RE = re.compile(
    r'^(-?\d+,-?\d+)(-?,-?\d+,-?\d+)+(,####.*)$'
)

# Numero di punti da campionare per ogni curva bezier (top + bot)
# CTW1500 usa N_SAMPLE=7: 7 top + 7 bot = 14 punti = 28 coordinate
N_SAMPLE = 7


def _bezier_cubic(p0, p1, p2, p3, t):
    """Valuta una curva cubica di Bezier al parametro t in [0,1]."""
    u = 1.0 - t
    return [
        u**3 * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t**3 * p3[1],
    ]


def _sample_curve(ctrl, n):
    """
    Campiona una curva cubica di Bezier in n punti uniformemente spaziati
    (t = 0, 1/(n-1), ..., 1) e restituisce i punti come interi.

    Arrotondamento: math.floor con correzione epsilon (+1e-9) per eliminare
    il floating point noise che causa 118.9999... -> 118 invece di 119.
    Questa convenzione corrisponde esattamente al formato GT CTW1500 verificato
    sul file 0001001.txt (es. 51+47/6=58.833 -> floor=58, 119.000 -> 119).
    """
    p0, p1, p2, p3 = ctrl
    step = 1.0 / (n - 1)
    pts = []
    for i in range(n):
        t = i * step
        xy = _bezier_cubic(p0, p1, p2, p3, t)
        # floor con epsilon per stabilita' floating point
        xi = math.floor(xy[0] + 1e-9)
        yi = math.floor(xy[1] + 1e-9)
        pts.append((xi, yi))
    return pts


def _bezier_pts_to_gt_coords(bezier_pts, n_sample=N_SAMPLE):
    """
    Converte gli 8 punti di controllo bezier (16 valori float) nel formato
    GT CTW1500: una lista di (n_sample*2)*2 interi.

    Convenzione bezier_pts (stessa di convert_hiertext.py):
      [TOP_p0, TOP_p1, TOP_p2, TOP_p3, BOT_p0, BOT_p1, BOT_p2, BOT_p3]
      TOP:  TL -> TR  (sinistra -> destra, margine superiore)
      BOT:  BR -> BL  (destra -> sinistra, margine inferiore)

    Output: lista di n_sample*2 coppie (x,y) come interi, prima tutti i
    punti TOP poi tutti i punti BOT, concatenati in una sequenza piatta
    [x0,y0, x1,y1, ..., x_{2n-1},y_{2n-1}].
    """
    bp = bezier_pts
    if len(bp) != 16:
        return None

    top_ctrl = [bp[0:2], bp[2:4], bp[4:6], bp[6:8]]
    bot_ctrl  = [bp[8:10], bp[10:12], bp[12:14], bp[14:16]]

    top_pts = _sample_curve(top_ctrl, n_sample)
    bot_pts = _sample_curve(bot_ctrl, n_sample)

    # Formato GT: prima tutti i top points poi tutti i bot points
    all_pts = top_pts + bot_pts
    coords = [v for xy in all_pts for v in xy]
    return coords


def make_gt_line(ann):
    """
    Costruisce la riga GT nel formato CTW1500:
      x0,y0,x1,y1,...,x13,y13,####testo

    Le coordinate sono i 14 punti campionati dalle due curve bezier cubiche
    (7 punti top + 7 punti bottom), ciascuno come intero (floor + epsilon).

    Il campo 'ignore' e' determinato da:
      1. ann.get('ignore', 0) == 1  (da gt_source con ignore esplicito)
      2. testo vuoto dopo strip()

    Se ignore=True, il testo e' lasciato vuoto (####).
    Restituisce None se bezier_pts e' assente/malformato o la riga non e'
    valida secondo VALID_LINE_RE.
    """
    bezier_pts = ann.get('bezier_pts')
    if not bezier_pts or len(bezier_pts) != 16:
        return None

    coords = _bezier_pts_to_gt_coords(bezier_pts)
    if coords is None or len(coords) < 6:
        return None

    coords_str = ','.join(str(v) for v in coords)

    text = ann.get('text', '').strip()
    ignored = ann.get('ignore', 0) == 1 or text == ''

    if ignored:
        line = f"{coords_str},####"
    else:
        # rimuovi qualsiasi '####' nel testo per non rompere lo split
        safe_text = text.replace('####', '').strip()
        line = f"{coords_str},####{safe_text}" if safe_text else f"{coords_str},####"

    # Validazione finale
    if not VALID_LINE_RE.match(line):
        return None
    parts = line.split(',####')
    if len(parts) != 2:
        return None
    if len(parts[0].split(',')) < 6:
        return None

    return line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
with open('datasets/hiertext/test_gt_source.json', encoding='utf-8') as f:
    data = json.load(f)

ann_by_img = defaultdict(list)
for ann in data['annotations']:
    ann_by_img[ann['image_id']].append(ann)

out_zip = 'datasets/evaluation/gt_hiertext.zip'
skipped_total = 0

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for img in data['images']:
        img_id   = img['id']
        filename = '{:07d}.txt'.format(img_id)
        lines    = []

        for ann in ann_by_img[img_id]:
            line = make_gt_line(ann)
            if line is None:
                skipped_total += 1
                continue
            lines.append(line)

        content = '\n'.join(lines)
        zf.writestr(filename, content)

print(f"GT scritto: {out_zip}")
print(f"Annotazioni scartate (malformate): {skipped_total}")

# ---------------------------------------------------------------------------
# Verifica approfondita
# ---------------------------------------------------------------------------
print("\n--- Verifica ---")
with zipfile.ZipFile(out_zip) as zf:
    names = zf.namelist()
    print(f"File nel zip: {len(names)}")
    total_bad = 0
    for name in names:
        content = zf.read(name).decode('utf-8', errors='replace')
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',####')
            if len(parts) != 2:
                total_bad += 1
                print(f"  MALFORMATA (split) in {name}: '{line[:80]}'")
                continue
            coord_vals = parts[0].split(',')
            if len(coord_vals) < 6 or len(coord_vals) % 2 != 0:
                total_bad += 1
                print(f"  MALFORMATA (coords) in {name}: '{line[:80]}'")

    print(f"Totale righe malformate nel zip: {total_bad}")
    if total_bad == 0:
        print("  \u2705 Tutte le righe sono valide")

    # Mostra sample delle prime righe
    if names:
        sample = names[0]
        lines_sample = zf.read(sample).decode().split('\n')
        print(f"\nSample {sample} ({len(lines_sample)} righe):")
        for l in lines_sample[:5]:
            # mostra anche il numero di coordinate
            if ',####' in l:
                n_coords = len(l.split(',####')[0].split(','))
                print(f"  [{n_coords} coord] {l}")
            else:
                print(f"  {l}")
