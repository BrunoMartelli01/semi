import json
import math
import zipfile
import re
from collections import defaultdict

# Regex: almeno 6 coppie di interi (3 punti minimo per un poligono valido)
# poi ,#### e testo opzionale. Garantisce compatibilita' con rrc_evaluation_funcs.
VALID_LINE_RE = re.compile(
    r'^(-?\d+,-?\d+)(,-?\d+,-?\d+){5,}(,####.*)$'
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

    Arrotondamento: round ai valori interi piu' vicini.
    Nota: il rounding puo' introdurre piccoli artefatti topologici,
    quindi a valle viene eseguito un controllo di self-intersection.
    """
    p0, p1, p2, p3 = ctrl
    step = 1.0 / (n - 1)
    pts = []
    for i in range(n):
        t = i * step
        xy = _bezier_cubic(p0, p1, p2, p3, t)
        xi = int(round(xy[0]))
        yi = int(round(xy[1]))
        pts.append((xi, yi))
    return pts


def _bezier_pts_to_gt_coords(bezier_pts, n_sample=N_SAMPLE):
    """
    Converte gli 8 punti di controllo bezier (16 valori float) nel formato
    GT CTW1500: una lista di (n_sample*2)*2 interi.

    Convenzione bezier_pts:
      [TOP_p0, TOP_p1, TOP_p2, TOP_p3, BOT_p0, BOT_p1, BOT_p2, BOT_p3]
      TOP:  TL -> TR  (sinistra -> destra, margine superiore)
      BOT:  BR -> BL  (destra -> sinistra, margine inferiore)

    Output: lista di n_sample*2 coppie (x,y) come interi, prima tutti i
    punti TOP poi tutti i punti BOT, concatenati in una sequenza piatta.
    """
    bp = bezier_pts
    if len(bp) != 16:
        return None

    top_ctrl = [bp[0:2], bp[2:4], bp[4:6], bp[6:8]]
    bot_ctrl = [bp[8:10], bp[10:12], bp[12:14], bp[14:16]]

    top_pts = _sample_curve(top_ctrl, n_sample)
    bot_pts = _sample_curve(bot_ctrl, n_sample)

    all_pts = top_pts + bot_pts
    coords = [v for xy in all_pts for v in xy]
    return coords


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
        min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True

    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


def _remove_consecutive_duplicate_points(pts):
    cleaned = []
    for p in pts:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _has_self_intersection(pts):
    """
    Controlla self-intersection/ring self-intersection su un anello chiuso.
    I segmenti adiacenti condividono un vertice e non vanno testati tra loro.
    """
    pts = _remove_consecutive_duplicate_points(pts)
    n = len(pts)
    if n < 3:
        return True
    if n == 3:
        return False

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]

        if a == b:
            return True

        for j in range(i + 1, n):
            if j == i:
                continue
            if (i + 1) % n == j:
                continue
            if (j + 1) % n == i:
                continue
            if i == 0 and j == n - 1:
                continue

            c = pts[j]
            d = pts[(j + 1) % n]

            if c == d:
                return True

            if _segments_intersect(a, b, c, d):
                return True

    return False


def make_gt_line(ann, debug_prefix=''):
    """
    Costruisce la riga GT nel formato CTW1500:
      x0,y0,x1,y1,...,x13,y13,####testo

    Le coordinate sono i 14 punti campionati dalle due curve bezier cubiche
    (7 punti top + 7 punti bottom), ciascuno come intero.

    Il campo 'ignore' e' determinato da:
      1. ann.get('ignore', 0) == 1
      2. testo vuoto dopo strip()

    Se il poligono risultante contiene self-intersection / ring self-intersection
    dopo il cast a int, l'annotazione viene scartata.
    """
    bezier_pts = ann.get('bezier_pts')
    if not bezier_pts or len(bezier_pts) != 16:
        return None

    coords = _bezier_pts_to_gt_coords(bezier_pts)
    if coords is None or len(coords) < 6:
        return None

    pts = list(zip(coords[0::2], coords[1::2]))
    pts = _remove_consecutive_duplicate_points(pts)

    if len(pts) < 3:
        return None

    if _has_self_intersection(pts):
        if debug_prefix:
            print(f"SELF-INTERSECTION scartata: {debug_prefix} ann_id={ann.get('id', 'n/a')}")
        return None

    coords_str = ','.join(str(v) for xy in pts for v in xy)

    text = ann.get('text', '').strip()
    ignored = ann.get('ignore', 0) == 1 or text == ''

    if ignored:
        line = f"{coords_str},####"
    else:
        safe_text = text.replace('####', '').strip()
        line = f"{coords_str},####{safe_text}" if safe_text else f"{coords_str},####"

    if not VALID_LINE_RE.match(line):
        if debug_prefix:
            print(f"RIGA NON VALIDA: {debug_prefix} ann_id={ann.get('id', 'n/a')} -> {line}")
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
written_total = 0

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for img in data['images']:
        img_id = img['id']
        filename = '{:07d}.txt'.format(img_id)
        lines = []

        for ann in ann_by_img[img_id]:
            line = make_gt_line(ann, debug_prefix=f"img={img_id}")
            if line is None:
                skipped_total += 1
                continue
            lines.append(line)
            written_total += 1

        content = '\n'.join(lines)
        zf.writestr(filename, content)

print(f"GT scritto: {out_zip}")
print(f"Annotazioni scritte: {written_total}")
print(f"Annotazioni scartate (malformate o self-intersecting): {skipped_total}")

# ---------------------------------------------------------------------------
# Verifica approfondita
# ---------------------------------------------------------------------------
print("\n--- Verifica ---")
with zipfile.ZipFile(out_zip) as zf:
    names = zf.namelist()
    print(f"File nel zip: {len(names)}")
    total_bad = 0
    total_self_intersections = 0

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
                continue

            pts = []
            for i in range(0, len(coord_vals), 2):
                pts.append((int(coord_vals[i]), int(coord_vals[i + 1])))

            if _has_self_intersection(pts):
                total_self_intersections += 1
                print(f"  SELF-INTERSECTION in {name}: '{line[:80]}'")

    print(f"Totale righe malformate nel zip: {total_bad}")
    print(f"Totale righe self-intersecting nel zip: {total_self_intersections}")
    if total_bad == 0 and total_self_intersections == 0:
        print("  ✅ Tutte le righe sono valide e non self-intersecting")

    if names:
        sample = names[0]
        lines_sample = zf.read(sample).decode().split('\n')
        print(f"\nSample {sample} ({len(lines_sample)} righe):")
        for l in lines_sample[:5]:
            if ',####' in l:
                n_coords = len(l.split(',####')[0].split(','))
                print(f"  [{n_coords} coord / {n_coords//2} pt] {l[:120]}")