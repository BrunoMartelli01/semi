import json
import math
import random

# 96-voc convention (same as CTW1500 / SemiETS):
# ASCII printable characters: ' ' (32) ... '~' (126)  -> 95 symbols
# CTLABELS[i] = chr(i + 32)  for i in 0..94
# blank/unknown for CTC decoding: 95 (= VOC_SIZE - 1)
# dataset padding sentinel used in rec arrays: 96 (= VOC_SIZE)
CTLABELS = [chr(i) for i in range(32, 127)]  # 95 printable ASCII chars
assert len(CTLABELS) == 95

VOC_SIZE = 96
BLANK_TOKEN = VOC_SIZE - 1   # 95
PAD_TOKEN = VOC_SIZE          # 96

# CTW1500 usa rec di lunghezza 100 (verificato sul file di riferimento)
MAX_LEN = 100

# Struttura categories identica a CTW1500
CATEGORIES = [{
    "supercategory": "beverage",
    "id": 1,
    "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
    "name": "text"
}]


def text_to_rec(text, max_len=MAX_LEN):
    """
    Encode text using DeepSolo/SemiETS 96-voc format.

    Mapping:
      ASCII printable chars (space=32 ... ~=126) -> 0..94
      blank/unknown is reserved at 95 and is NOT written into GT rec
      padding sentinel is 96

    Characters outside the printable ASCII range are ignored.
    Output length is fixed to max_len with PAD_TOKEN=96.
    """
    rec = []
    for c in str(text).strip():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:max_len]
    rec += [PAD_TOKEN] * (max_len - len(rec))
    assert len(rec) == max_len
    assert all((0 <= t < BLANK_TOKEN) or (t == PAD_TOKEN) for t in rec), \
        f"invalid rec tokens for DeepSolo 96-voc: {rec}"
    return rec


def _lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _vertices_to_pixels(verts, img_w, img_h):
    """
    HierText vertices possono essere:
      - normalizzati in [0, 1]  -> moltiplicare per img_w / img_h
      - gia' in pixel assoluti  -> usare direttamente

    La distinzione e' semplice: se tutti i valori x e y sono <= 1.0
    (con un piccolo margine per floating point) le coordinate sono
    normalizzate; altrimenti sono gia' in pixel.

    Restituisce una lista di [x_px, y_px] in pixel assoluti float.
    """
    if isinstance(verts[0], dict):
        pts_raw = [[v["x"], v["y"]] for v in verts]
    else:
        pts_raw = [[v[0], v[1]] for v in verts]

    xs_raw = [p[0] for p in pts_raw]
    ys_raw = [p[1] for p in pts_raw]

    # Se tutti i valori sono in [0, 1] le coordinate sono normalizzate
    is_normalized = (max(xs_raw) <= 1.0 + 1e-6) and (max(ys_raw) <= 1.0 + 1e-6)

    if is_normalized:
        pts = [[p[0] * img_w, p[1] * img_h] for p in pts_raw]
    else:
        pts = pts_raw  # gia' in pixel

    return pts, is_normalized


def _reorder_quad(pts):
    """
    Riordina 4 vertici di un quadrilatero in [TL, TR, BR, BL] garantendo
    che la curva TOP vada da sinistra a destra (TL->TR) e la curva BOT
    torni da destra a sinistra (BR->BL).

    Questo previene la self-intersection del ring bezier quando lo
    strumento di validazione ricostruisce il contorno come:
      ring = TOP_curve + BOT_curve + chiusura

    HierText puo' fornire i vertici in ordine arbitrario (orario,
    antiorario, o partendo da un angolo diverso da TL). Questa funzione
    normalizza sempre a [TL, TR, BR, BL] basandosi sulle coordinate y:
      - I due punti con y MINORE (piu' in ALTO) formano il TOP edge
      - I due punti con y MAGGIORE (piu' in BASSO) formano il BOT edge
      - Dentro ogni coppia si ordina per x crescente (sinistra -> destra)

    Funziona per rettangoli, parallelogrammi e quadrilateri obliqui generici.
    """
    # Ordina per y crescente (y minore = piu' in alto in coordinate immagine)
    by_y = sorted(pts, key=lambda p: p[1])
    top_two = sorted(by_y[:2], key=lambda p: p[0])   # i due piu' in alto, da sx a dx
    bot_two = sorted(by_y[2:], key=lambda p: p[0])   # i due piu' in basso, da sx a dx
    TL, TR = top_two[0], top_two[1]
    BL, BR = bot_two[0], bot_two[1]
    return [TL, TR, BR, BL]


def _principal_axis(pts):
    """
    Calcola l'asse principale del poligono tramite PCA approssimata:
    restituisce il vettore unitario (dx, dy) che massimizza la varianza
    dei punti proiettati.

    Viene usato per separare top/bottom nei poligoni obliqui: l'asse
    perpendicolare a questo vettore divide il poligono in meta' superiore
    e meta' inferiore rispetto alla direzione del testo.
    """
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # Matrice di covarianza 2x2
    sxx = sum((p[0] - cx) ** 2 for p in pts)
    syy = sum((p[1] - cy) ** 2 for p in pts)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in pts)

    # Autovettore dominante tramite formula analitica 2x2
    diff = sxx - syy
    angle = 0.5 * math.atan2(2.0 * sxy, diff) if (diff != 0 or sxy != 0) else 0.0
    dx = math.cos(angle)
    dy = math.sin(angle)
    return dx, dy, cx, cy


def _split_top_bottom_by_axis(pts):
    """
    Divide i punti del poligono in due gruppi (top / bottom) proiettando
    ogni punto sull'asse PERPENDICOLARE all'asse principale del poligono.

    Questo e' il fix per i casi complessi (poligoni obliqui, ruotati,
    non-rettangolari con n>4 vertici) dove la semplice divisione per y
    produce self-intersection nelle curve di Bezier.

    Strategia:
      1. Calcola l'asse principale (PCA) e il centroide.
      2. Proietta ogni punto sull'asse perpendicolare all'asse principale.
      3. Punti con proiezione < 0 -> top half (margine superiore del testo)
         Punti con proiezione >= 0 -> bottom half (margine inferiore)
      4. Se un gruppo e' vuoto (es. tutti i punti su un lato), fallback
         alla divisione per y globale.

    Restituisce (top_pts, bot_pts) dove:
      top_pts: lista di punti del bordo superiore, ordinati per x crescente
      bot_pts: lista di punti del bordo inferiore, ordinati per x crescente
    """
    dx, dy, cx, cy = _principal_axis(pts)

    # Vettore perpendicolare all'asse principale (ruota di 90 gradi)
    px, py = -dy, dx

    # Proiezione di ogni punto sull'asse perpendicolare (rispetto al centroide)
    projections = [(p[0] - cx) * px + (p[1] - cy) * py for p in pts]
    med = sorted(projections)[len(projections) // 2]  # mediana

    top_pts = [p for p, proj in zip(pts, projections) if proj <= med]
    bot_pts = [p for p, proj in zip(pts, projections) if proj > med]

    # Fallback: se un gruppo e' vuoto usa divisione per y
    if not top_pts or not bot_pts:
        by_y = sorted(pts, key=lambda p: p[1])
        half = max(1, len(pts) // 2)
        top_pts = by_y[:half]
        bot_pts = by_y[half:]

    # Ordina ciascun gruppo per x crescente (sinistra -> destra)
    top_pts = sorted(top_pts, key=lambda p: p[0])
    bot_pts = sorted(bot_pts, key=lambda p: p[0])

    return top_pts, bot_pts


def _eval_cubic(p0, p1, p2, p3, t):
    """Valuta una Bezier cubica in t."""
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    a = mt2 * mt
    b = 3.0 * mt2 * t
    c = 3.0 * mt * t2
    d = t * t2
    x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
    y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
    return [x, y]


def _segments_intersect(a1, a2, b1, b2, eps=1e-6):
    """Test di intersezione fra segmenti a1-a2 e b1-b2, robusto con segmenti degeneri."""
    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_seg(p, q, r):
        return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
                min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps)

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) < eps and _on_seg(a1, b1, a2):
        return True
    if abs(o2) < eps and _on_seg(a1, b2, a2):
        return True
    if abs(o3) < eps and _on_seg(b1, a1, b2):
        return True
    if abs(o4) < eps and _on_seg(b1, a2, b2):
        return True
    return False


def _ring_has_self_intersection(bezier, n_samples=40):
    """Campiona il ring (top+bottom) e controlla self-intersection."""
    pts_ctrl = [[bezier[2 * i], bezier[2 * i + 1]] for i in range(8)]
    top = pts_ctrl[0:4]
    bot = pts_ctrl[4:8]

    ts = [i / (n_samples - 1) for i in range(n_samples)]
    top_s = [_eval_cubic(*top, t) for t in ts]
    bot_s = [_eval_cubic(*bot, t) for t in ts]
    bot_s.reverse()

    ring = top_s + bot_s
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    m = len(ring)
    for i in range(m - 1):
        a1, a2 = ring[i], ring[i + 1]
        for j in range(i + 2, m - 1):
            if j == i or j == i + 1:
                continue
            b1, b2 = ring[j], ring[j + 1]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _bbox_bezier_from_pts(pts):
    """Fallback: bounding box dei punti come ring Bezier rettangolare CTW1500."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    TL = [xmin, ymin]
    TR = [xmax, ymin]
    BR = [xmax, ymax]
    BL = [xmin, ymax]

    top_p0 = TL
    top_p3 = TR
    top_p1 = _lerp(TL, TR, 1.0 / 3.0)
    top_p2 = _lerp(TL, TR, 2.0 / 3.0)

    bot_p0 = BR
    bot_p3 = BL
    bot_p1 = _lerp(BR, BL, 1.0 / 3.0)
    bot_p2 = _lerp(BR, BL, 2.0 / 3.0)

    bezier = [coord for p in [top_p0, top_p1, top_p2, top_p3,
                              bot_p0, bot_p1, bot_p2, bot_p3]
              for coord in p]
    return bezier


def _curve_to_bezier(curve):
    """Adatta una polilinea 2D con una Bezier cubica (4 controlli) in least-squares."""
    curve = np.asarray(curve, dtype=float).reshape(-1, 2)
    m = len(curve)
    if m <= 1:
        return np.vstack([curve[0]] * 4)

    if m == 2:
        p0, p3 = curve[0], curve[1]
        p1 = _lerp(p0, p3, 1.0 / 3.0)
        p2 = _lerp(p0, p3, 2.0 / 3.0)
        return np.vstack([p0, p1, p2, p3])

    diff = curve[1:] - curve[:-1]
    dist = np.linalg.norm(diff, axis=-1)
    total = dist.sum()
    if total <= 1e-6:
        return np.vstack([curve[0]] * 4)

    norm = dist / total
    norm = np.concatenate([[0.0], norm])
    t = norm.cumsum()

    B = np.stack([
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3
    ], axis=1)

    pseudo_inv = np.linalg.pinv(B)
    ctrl = pseudo_inv.dot(curve)
    ctrl[0] = curve[0]
    ctrl[-1] = curve[-1]
    return ctrl


def _poly_to_bezier(pts):
    """Versione finale robusta per HierText, con controllo self-intersection."""
    pts = [list(p) for p in pts]
    if not pts:
        return [0.0] * 16

    # deduplica consecutivi
    dedup = [pts[0]]
    for p in pts[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    pts = dedup
    n = len(pts)

    if n == 1:
        p0 = pts[0]
        return [p0[0], p0[1]] * 8

    if n == 2:
        p0, p3 = pts[0], pts[1]
        top_p0 = p0
        top_p3 = p3
        top_p1 = _lerp(p0, p3, 1.0 / 3.0)
        top_p2 = _lerp(p0, p3, 2.0 / 3.0)
        bot_p0, bot_p3 = top_p3, top_p0
        bot_p1, bot_p2 = top_p2, top_p1
        bezier = [coord for p in [top_p0, top_p1, top_p2, top_p3,
                                  bot_p0, bot_p1, bot_p2, bot_p3]
                  for coord in p]
        return bezier

    if n == 3:
        by_y = sorted(pts, key=lambda p: p[1])
        apex = by_y[0]
        base = sorted(by_y[1:], key=lambda p: p[0])
        TL = apex
        TR = apex
        BL = base[0]
        BR = base[1]

        top_p0 = TL
        top_p1 = _lerp(TL, TR, 1.0 / 3.0)
        top_p2 = _lerp(TL, TR, 2.0 / 3.0)
        top_p3 = TR

        bot_p0 = BR
        bot_p1 = _lerp(BR, BL, 1.0 / 3.0)
        bot_p2 = _lerp(BR, BL, 2.0 / 3.0)
        bot_p3 = BL

        bezier = [coord for p in [top_p0, top_p1, top_p2, top_p3,
                                  bot_p0, bot_p1, bot_p2, bot_p3]
                  for coord in p]
        return bezier

    # n >= 4
    if n == 4:
        TL, TR, BR, BL = _reorder_quad(pts)

        top_p0 = TL
        top_p3 = TR
        top_p1 = _lerp(TL, TR, 1.0 / 3.0)
        top_p2 = _lerp(TL, TR, 2.0 / 3.0)

        bot_p0 = BR
        bot_p3 = BL
        bot_p1 = _lerp(BR, BL, 1.0 / 3.0)
        bot_p2 = _lerp(BR, BL, 2.0 / 3.0)

        bezier = [coord for p in [top_p0, top_p1, top_p2, top_p3,
                                  bot_p0, bot_p1, bot_p2, bot_p3]
                  for coord in p]
        if _ring_has_self_intersection(bezier):
            return _bbox_bezier_from_pts(pts)
        return bezier

    # n > 4: separa contorno in catena superiore/inferiore
    def _left_key(p):
        return (p[0], p[1])

    def _right_key(p):
        return (-p[0], p[1])

    left_idx = min(range(n), key=lambda i: _left_key(pts[i]))
    right_idx = min(range(n), key=lambda i: _right_key(pts[i]))

    if left_idx <= right_idx:
        chain1 = pts[left_idx:right_idx + 1]
        chain2 = pts[right_idx:] + pts[:left_idx + 1]
    else:
        chain1 = pts[left_idx:] + pts[:right_idx + 1]
        chain2 = pts[right_idx:left_idx + 1]

    def _mean_y(chain):
        return sum(p[1] for p in chain) / max(1, len(chain))

    if _mean_y(chain1) <= _mean_y(chain2):
        top_chain, bot_chain = chain1, chain2
    else:
        top_chain, bot_chain = chain2, chain1

    if len(top_chain) >= 2 and top_chain[0][0] > top_chain[-1][0]:
        top_chain = list(reversed(top_chain))
    if len(bot_chain) >= 2 and bot_chain[0][0] < bot_chain[-1][0]:
        bot_chain = list(reversed(bot_chain))

    top_ctrl = _curve_to_bezier(top_chain)
    bot_ctrl = _curve_to_bezier(bot_chain)

    top_p0, top_p1, top_p2, top_p3 = top_ctrl.tolist()
    bot_p0, bot_p1, bot_p2, bot_p3 = bot_ctrl.tolist()

    bezier = [coord for p in [top_p0, top_p1, top_p2, top_p3,
                              bot_p0, bot_p1, bot_p2, bot_p3]
              for coord in p]

    if _ring_has_self_intersection(bezier):
        bezier = _bbox_bezier_from_pts(pts)

    return bezier

def _bezier_bbox(bezier, img_w=None, img_h=None):
    """
    Calcola il bbox XYWH a partire dagli 8 punti di controllo bezier,
    identico alla procedura usata in CTW1500:

      x_min = min di tutti gli x dei punti di controllo
      y_min = min di tutti gli y dei punti di controllo
      width  = max_x - x_min
      height = max_y - y_min

    CLIPPING: se y_min o x_min risultano negativi (punti fuori bordo
    immagine), vengono clippati a 0, come verificato in CTW1500
    (es. annotation id=4079, img 0541.jpg: y_ctrl=-16.43 -> y_min=0).
    Il clipping si applica SOLO a x_min e y_min; width e height vengono
    ricalcolati coerentemente dopo il clip.
    """
    xs = [bezier[i]     for i in range(0,  16, 2)]
    ys = [bezier[i + 1] for i in range(0,  16, 2)]

    x_min_raw = min(xs)
    y_min_raw = min(ys)
    x_max     = max(xs)
    y_max     = max(ys)

    # Clip ai bordi dell'immagine
    x_min = max(0.0, x_min_raw)
    y_min = max(0.0, y_min_raw)

    if img_w is not None:
        x_min = min(x_min, img_w)
        x_max = min(x_max, img_w)
    if img_h is not None:
        y_min = min(y_min, img_h)
        y_max = min(y_max, img_h)

    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)

    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]


def _read_jsonl(jsonl_path):
    """
    Legge un file HierText .jsonl.

    HierText usa un formato ibrido: il file contiene UN singolo oggetto JSON
    con chiave 'annotations' che contiene la lista di tutte le immagini.
    In alternativa, supporta anche il formato JSON Lines puro (un oggetto per riga).
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()

    # Prova prima come singolo oggetto JSON (formato HierText standard)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "annotations" in data:
            return data["annotations"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: formato JSON Lines (una riga = un oggetto)
    samples = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def convert(jsonl_path, out_json_path, out_gt_source_path, img_suffix='.jpg'):
    """
    Convert HierText annotations to the COCO-like format expected by DeepSolo/SemiETS,
    identico allo schema di datasets/ctw1500/train_96voc.json.

    Formato output per ogni annotazione:
      {
        "image_id": <int>,
        "area": <float>,
        "category_id": 1,
        "iscrowd": 0,
        "id": <int>,
        "bezier_pts": [x0,y0, x1,y1, ..., x7,y7],   # 8 punti = 16 valori
        "rec": [int * 100],                           # lunghezza 100, PAD=96
        "bbox": [x_min, y_min, width, height]         # XYWH_ABS, clippato a [0, img_wh]
      }

    Struttura bezier_pts (convenzione CTW1500):
      Punti 0-3: curva cubica TOP  (TL -> TR, da sinistra a destra)
      Punti 4-7: curva cubica BOT  (BR -> BL, da destra a sinistra)
      I punti di controllo interni sono interpolati linearmente a 1/3 e 2/3.

      Gestione poligoni complessi (FIX self-intersection):
        - n==3 (triangolo): apice = TL=TR, base = BL/BR
        - n==4 (quad):      _reorder_quad -> [TL, TR, BR, BL]
        - n>4  (poligono):  _split_top_bottom_by_axis (PCA) -> top/bot
          La separazione PCA gestisce poligoni obliqui, ruotati e concavi
          dove la semplice divisione per y generava self-intersection.

    Struttura bbox (convenzione CTW1500):
      Calcolata come min/max degli 8 punti di controllo bezier (NON dei
      vertices originali), poi clippata a [0, img_w] x [0, img_h].

    Campi NON presenti (rimossi rispetto alla versione precedente):
      - "segmentation"  (non esiste in CTW1500)
      - "text"          (non esiste in CTW1500 supervised)
      - "ignore"        (non esiste in CTW1500)

    Struttura "images" identica a CTW1500:
      {width, date_captured, license, flickr_url, file_name, id, coco_url, height}
    """
    samples = _read_jsonl(jsonl_path)

    images, annotations_all = [], []
    ann_id = 1
    n_normalized = 0
    n_absolute = 0
    bbox_samples = []  # per sanity check

    for img_id, sample in enumerate(samples):
        fname = sample['image_id'] + img_suffix
        img_w = sample.get("image_width", 0)
        img_h = sample.get("image_height", 0)

        # Struttura images identica a CTW1500
        images.append({
            "width": img_w,
            "date_captured": "",
            "license": 0,
            "flickr_url": "",
            "file_name": fname,
            "id": img_id,
            "coco_url": "",
            "height": img_h,
        })

        for para in sample.get("paragraphs", []):
            for ln in para.get("lines", []):
                for word in ln.get("words", []):
                    verts = word.get("vertices", [])
                    if len(verts) < 3:
                        continue

                    # --- converti vertices in pixel assoluti ---
                    pts, was_normalized = _vertices_to_pixels(verts, img_w, img_h)
                    if was_normalized:
                        n_normalized += 1
                    else:
                        n_absolute += 1

                    # --- bezier_pts ---
                    # _poly_to_bezier gestisce n==3, n==4 e n>4 con algoritmi
                    # dedicati per prevenire self-intersection in tutti i casi.
                    bezier = _poly_to_bezier(pts)

                    # --- bbox: min/max sugli 8 punti di controllo bezier, clippato ---
                    # Identico alla procedura CTW1500 (verificata su train_96voc_1_labeled.json)
                    bbox = _bezier_bbox(bezier,
                                        img_w=img_w if img_w > 0 else None,
                                        img_h=img_h if img_h > 0 else None)
                    x_min, y_min, w_box, h_box = bbox

                    # 96-voc rec con MAX_LEN=100 (come CTW1500)
                    text_orig = str(word.get("text", ""))
                    legible = word.get("legible", True)
                    text_norm = text_orig.strip()
                    rec = text_to_rec(text_norm)

                    has_real_text = any(t != PAD_TOKEN for t in rec)
                    ignore = 1 if (not legible) or (not has_real_text) else 0

                    if w_box <= 0 or h_box <= 0:
                        ignore = 1

                    # area calcolata sulla bbox clippata (evita valori negativi)
                    area = round(w_box * h_box, 1) if w_box > 0 and h_box > 0 else 0.0

                    # Annotazione nel formato CTW1500 esatto.
                    # I campi con prefisso _ sono interni e vengono rimossi
                    # prima della serializzazione finale.
                    ann = {
                        "image_id": img_id,
                        "area": area,
                        "category_id": 1,
                        "iscrowd": 0,
                        "id": ann_id,
                        "bezier_pts": bezier,
                        "rec": rec,
                        "bbox": bbox,
                        # campi interni (rimossi al momento della scrittura)
                        "_ignore": ignore,
                        "_text": text_norm,
                    }
                    annotations_all.append(ann)
                    ann_id += 1

                    if len(bbox_samples) < 5:
                        bbox_samples.append({
                            "img": fname, "img_wh": (img_w, img_h),
                            "was_normalized": was_normalized,
                            "bbox_xywh": bbox,
                        })

    # --- sanity check a schermo ---
    print(f"\n[SANITY CHECK bbox] vertices normalizzati={n_normalized}, gia' in pixel={n_absolute}")
    print("Prime 5 bbox (formato XYWH salvato nel JSON):")
    for s in bbox_samples:
        print(f"  img={s['img']} ({s['img_wh'][0]}x{s['img_wh'][1]}) "
              f"normalized={s['was_normalized']} bbox_xywh={s['bbox_xywh']}")
    print()

    # --- helper per pulizia finale delle annotazioni ---
    def _clean(ann):
        return {k: v for k, v in ann.items() if not k.startswith("_")}

    def _clean_with_text(ann):
        d = _clean(ann)
        d["text"] = ann["_text"]
        return d

    annotations_supervised = [
        _clean(ann) for ann in annotations_all if ann["_ignore"] == 0
    ]
    annotations_gt = [_clean_with_text(ann) for ann in annotations_all]

    # --- supervised (no text, no ignore, no segmentation) ---
    coco_supervised = {
        "images": images,
        "annotations": annotations_supervised,
        "categories": CATEGORIES,
    }
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_supervised, f)

    # --- gt_source: tutte le annotazioni + campo text per evaluation ---
    coco_gt_source = {
        "images": images,
        "annotations": annotations_gt,
        "categories": CATEGORIES,
    }
    with open(out_gt_source_path, 'w', encoding='utf-8') as f:
        json.dump(coco_gt_source, f)

    n_ignored = len(annotations_all) - len(annotations_supervised)
    print(f"Scritto {out_json_path}")
    print(f"  -> {len(images)} immagini, {len(annotations_supervised)} ann valide, {n_ignored} ignored")
    print(f"Scritto {out_gt_source_path}")
    print(f"  -> {len(annotations_all)} annotazioni totali (ignored incluse)")

    # Restituisce anche le annotazioni grezze (con campi interni) per uso in make_semi_splits
    return images, annotations_all, annotations_supervised


def make_semi_splits(images, annotations_all, label_ratio, out_dir,
                     split_name="train_96voc", seed=42):
    random.seed(seed)
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)
    n_label = max(1, int(len(img_ids) * label_ratio))
    labeled_ids = set(img_ids[:n_label])
    unlabeled_ids = set(img_ids[n_label:])

    ann_by_img = {}
    for ann in annotations_all:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    def _clean(ann):
        return {k: v for k, v in ann.items() if not k.startswith("_")}

    def build_split(ids, include_text: bool):
        split_imgs = [img for img in images if img["id"] in ids]
        split_anns = []
        for img in split_imgs:
            for ann in ann_by_img.get(img["id"], []):
                if ann["_ignore"] == 0:
                    d = _clean(ann)
                    if include_text:
                        d["text"] = ann["_text"]
                    split_anns.append(d)
        return {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": CATEGORIES,
        }
    if label_ratio * 100>= 1:
        ratio_str = str(int(label_ratio*100))
    else:
        ratio_str = str(label_ratio*100)

    labeled_path = f"{out_dir}/{split_name}_{ratio_str}_labeled.json"
    unlabeled_path = f"{out_dir}/{split_name}_{ratio_str}_unlabeled.json"

    labeled_data = build_split(labeled_ids, include_text=True)
    unlabeled_data = build_split(unlabeled_ids, include_text=False)

    with open(labeled_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f)
    with open(unlabeled_path, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_data, f)

    print(f"  labeled   -> {labeled_path}")
    print(f"              {len(labeled_ids)} img, {len(labeled_data['annotations'])} ann")
    print(f"  unlabeled -> {unlabeled_path}")
    print(f"              {len(unlabeled_ids)} img, {len(unlabeled_data['annotations'])} ann")


if __name__ == "__main__":
    BASE = "datasets/hiertext"

    print("\n[1/3] Conversione validation -> test.json + test_gt_source.json")
    convert(
        jsonl_path=f"{BASE}/validation.jsonl",
        out_json_path=f"{BASE}/test.json",
        out_gt_source_path=f"{BASE}/test_gt_source.json",
    )

    print("\n[2/3] Conversione train -> train_96voc.json + train_gt_source.json")
    images, annotations_all, annotations_supervised = convert(
        jsonl_path=f"{BASE}/train.jsonl",
        out_json_path=f"{BASE}/train_96voc.json",
        out_gt_source_path=f"{BASE}/train_gt_source.json",
    )

    print("\n[3/3] Generazione split semi-supervised")
    for ratio in [0.005, 0.010, 0.020, 0.050, 0.10]:
        print(f"  ratio={ratio}")
        make_semi_splits(images, annotations_all, ratio, BASE)

    print("\n[VERIFICA] Controllo coerenza file generati...")
    import numpy as np

    checks = [
        (f"{BASE}/train_96voc.json", False, "train_96voc (supervised)"),
        (f"{BASE}/train_96voc_10_labeled.json", True, "10% labeled"),
        (f"{BASE}/train_96voc_10_unlabeled.json", False, "10% unlabeled"),
    ]
    for path, expect_text, label in checks:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        anns = d["annotations"]
        recs = np.array([a["rec"] for a in anns]) if anns else np.zeros((0, MAX_LEN), dtype=np.int64)
        bad = int(((recs == BLANK_TOKEN).sum() if recs.size else 0))
        over = int(((recs > PAD_TOKEN).sum() if recs.size else 0))
        has_text = all("text" in a for a in anns) if anns else True
        has_ignore = any("ignore" in a for a in anns) if anns else False
        has_seg = any("segmentation" in a for a in anns) if anns else False
        all_pad_rows = int(np.sum(np.all(recs == PAD_TOKEN, axis=1))) if recs.size else 0
        wrong_rec_len = sum(1 for a in anns if len(a["rec"]) != MAX_LEN)

        bboxes = np.array([a["bbox"] for a in anns]) if anns else np.zeros((0, 4))
        bad_bbox = int(np.sum((bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0))) if bboxes.size else 0
        neg_bbox = int(np.sum((bboxes[:, 0] < 0) | (bboxes[:, 1] < 0))) if bboxes.size else 0
        mean_w = float(np.mean(bboxes[:, 2])) if bboxes.size else 0.0
        mean_h = float(np.mean(bboxes[:, 3])) if bboxes.size else 0.0

        bez_lens = [len(a["bezier_pts"]) for a in anns] if anns else []
        wrong_bez = sum(1 for l in bez_lens if l != 16)

        status = []
        if bad > 0:
            status.append(f"ERRORE: trovati token blank=95 nei GT ({bad})")
        if over > 0:
            status.append(f"ERRORE: {over} token rec > {PAD_TOKEN}")
        if has_ignore:
            status.append("ERRORE: campo 'ignore' presente")
        if has_seg:
            status.append("ERRORE: campo 'segmentation' presente (non deve esserci)")
        if expect_text and not has_text:
            status.append("ERRORE: 'text' mancante")
        if not expect_text and has_text:
            status.append("WARN: 'text' presente (inatteso)")
        if all_pad_rows > 0:
            status.append(f"ERRORE: {all_pad_rows} istanze con rec tutto padding")
        if bad_bbox > 0:
            status.append(f"ERRORE: {bad_bbox} bbox con w<=0 o h<=0")
        if neg_bbox > 0:
            status.append(f"ERRORE: {neg_bbox} bbox con x_min<0 o y_min<0 (clip mancante)")
        if wrong_rec_len > 0:
            status.append(f"ERRORE: {wrong_rec_len} rec con lunghezza != {MAX_LEN}")
        if wrong_bez > 0:
            status.append(f"ERRORE: {wrong_bez} bezier_pts con lunghezza != 16")

        result = "OK" if not status else " | ".join(status)
        print(f"  [{label}]  img={len(d['images'])}  ann={len(anns)}  "
              f"rec_len={MAX_LEN}  bbox_mean=[w={mean_w:.1f}, h={mean_h:.1f}]  -> {result}")
