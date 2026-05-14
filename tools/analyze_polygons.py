import json
import math
from collections import Counter

# Importa le funzioni che hai già nel tuo convert_hiertext.py
from convert_hiertext import _read_jsonl, _vertices_to_pixels, _poly_to_bezier


# ---------- Utility geometriche di base ----------

def _dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def _poly_area(poly):
    """Area firmata (shoelace)."""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


# ---------- Rilevamento auto‑intersezioni su una polilinea chiusa ----------

def _orientation(p, q, r):
    """Orientazione (p,q,r) per test di intersezione segmenti."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(p, q, r):
    """True se q è sul segmento pr."""
    return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
            min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)


def _segments_intersect(p1, q1, p2, q2):
    """Test robusto di intersezione tra due segmenti (anche collineari)."""
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Casi collineari
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    return False


def has_self_intersection(poly, closed=True):
    """
    Ritorna True se la polilinea (eventualmente chiusa) ha auto‑intersezioni.
    Algoritmo O(N^2), sufficiente per pochi punti.[web:35]
    """
    pts = list(poly)
    n = len(pts)
    if n < 4:
        return False

    # Costruisci lista segmenti; se closed chiudi l'anello.
    segments = []
    for i in range(n - 1):
        segments.append((pts[i], pts[i + 1]))
    if closed:
        segments.append((pts[-1], pts[0]))
        seg_count = len(segments)
    else:
        seg_count = len(segments)

    for i in range(seg_count):
        p1, q1 = segments[i]
        for j in range(i + 1, seg_count):
            p2, q2 = segments[j]

            # salta segmenti adiacenti (condividono un vertice)
            if p1 in (p2, q2) or q1 in (p2, q2):
                continue

            if _segments_intersect(p1, q1, p2, q2):
                return True
    return False


# ---------- Bezier: da 8 punti di controllo a polilinea campionata ----------

def _eval_cubic(P0, P1, P2, P3, t):
    """Valuta una cubica di Bezier nel parametro t in [0,1].[web:36]"""
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    a = mt2 * mt
    b = 3 * mt2 * t
    c = 3 * mt * t2
    d = t * t2
    x = a * P0[0] + b * P1[0] + c * P2[0] + d * P3[0]
    y = a * P0[1] + b * P1[1] + c * P2[1] + d * P3[1]
    return [x, y]


def bezier_ring_to_poly(bezier, num_samples=40):
    """
    bezier: lista [x0,y0,...,x7,y7] (8 punti di controllo).
    Restituisce una polilinea che approssima il contorno chiuso
    (curva TOP + curva BOT).[web:31]
    """
    assert len(bezier) == 16
    pts = []
    # TOP: p0..p3
    top = [[bezier[0], bezier[1]],
           [bezier[2], bezier[3]],
           [bezier[4], bezier[5]],
           [bezier[6], bezier[7]]]
    # BOT: p4..p7
    bot = [[bezier[8], bezier[9]],
           [bezier[10], bezier[11]],
           [bezier[12], bezier[13]],
           [bezier[14], bezier[15]]]

    # campiona TOP da t=0..1
    for i in range(num_samples + 1):
        t = i / num_samples
        pts.append(_eval_cubic(top[0], top[1], top[2], top[3], t))

    # campiona BOT da t=0..1
    for i in range(1, num_samples + 1):
        t = i / num_samples
        pts.append(_eval_cubic(bot[0], bot[1], bot[2], bot[3], t))

    return pts  # anello quasi chiuso


# ---------- Analisi globale del dataset ----------

def analyze_hiertext(jsonl_path, max_examples=20, img_suffix='.jpg'):
    samples = _read_jsonl(jsonl_path)

    stats_n_vertices = Counter()
    stats_self_poly = 0
    stats_self_bezier = 0
    stats_total = 0

    examples_bad_bezier = []

    for sample in samples:
        img_w = sample.get("image_width", 0)
        img_h = sample.get("image_height", 0)

        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if len(verts) < 3:
                        continue

                    pts, _ = _vertices_to_pixels(verts, img_w, img_h)
                    if len(pts) < 3:
                        continue

                    stats_total += 1
                    n = len(pts)
                    stats_n_vertices[n] += 1

                    # Poligono originale
                    orig_self = has_self_intersection(pts, closed=True)
                    if orig_self:
                        stats_self_poly += 1

                    # Bezier generato dal tuo _poly_to_bezier
                    bez = _poly_to_bezier(pts)
                    ring_poly = bezier_ring_to_poly(bez, num_samples=40)
                    bez_self = has_self_intersection(ring_poly, closed=True)
                    if bez_self:
                        stats_self_bezier += 1
                        if len(examples_bad_bezier) < max_examples:
                            bbox = _bbox(pts)
                            area = abs(_poly_area(pts))
                            examples_bad_bezier.append({
                                "text": word.get("text", ""),
                                "n_vertices": n,
                                "orig_self_intersection": orig_self,
                                "bbox": bbox,
                                "area": area,
                                "vertices": pts,
                                "bezier": bez,
                            })

    # --- Stampa report sintetico ---
    print("=== ANALISI POLIGONI HIERTEXT ===")
    print(f"Totale word con almeno 3 vertici: {stats_total}")
    print("Distribuzione numero vertici:")
    for k in sorted(stats_n_vertices):
        print(f"  n={k}: {stats_n_vertices[k]}")

    print(f"\nPoligoni originali con auto-intersezioni: {stats_self_poly}")
    print(f"Ring Bezier con auto-intersezioni (dopo _poly_to_bezier): {stats_self_bezier}")

    print("\nEsempi di word con ring Bezier auto-intersecante (max", max_examples, "):")
    for ex in examples_bad_bezier:
        print("------------------------------------------------")
        print(f"text = '{ex['text']}'  n_vertices={ex['n_vertices']}")
        print(f"orig_self_intersection = {ex['orig_self_intersection']}")
        print(f"bbox = {ex['bbox']}")
        print(f"area = {ex['area']:.1f}")
        print(f"vertices = {ex['vertices']}")
        print(f"bezier   = {ex['bezier']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analisi forme poligoni HierText + Bezier.")
    parser.add_argument("--jsonl", required=True, help="Path a train.jsonl o validation.jsonl di HierText")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Quanti esempi stampare di Bezier auto-intersecanti")
    args = parser.parse_args()

    analyze_hiertext(args.jsonl, max_examples=args.max_examples)