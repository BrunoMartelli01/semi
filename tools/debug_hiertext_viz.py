import argparse
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Vocabolari supportati
# ---------------------------------------------------------------------------
CTLABELS_96 = [chr(i) for i in range(32, 127)]
CTLABELS_37 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                'q','r','s','t','u','v','w','x','y','z',
                '0','1','2','3','4','5','6','7','8','9']


def build_vocab(voc_size: int):
    if voc_size == 96:
        labels = CTLABELS_96
    elif voc_size == 37:
        labels = CTLABELS_37
    else:
        raise ValueError(f"Unsupported voc_size {voc_size}.")
    blank = len(labels)
    pad   = len(labels) + 1
    return labels, blank, pad


def text_to_rec(text, ctlabels, pad_token, max_len=25, lowercase=False):
    if lowercase:
        text = str(text).lower()
    rec = []
    for c in str(text):
        if c in ctlabels:
            rec.append(ctlabels.index(c))
    rec = rec[:max_len]
    rec += [pad_token] * (max_len - len(rec))
    return rec


def decode_rec(rec, ctlabels, pad_token):
    last_char = None
    out = []
    for c in rec:
        c = int(c)
        if c < pad_token - 1:
            if last_char != c:
                out.append(ctlabels[c])
                last_char = c
        else:
            last_char = None
    return ''.join(out)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def load_font(size=16):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json_or_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt:
        return {}
    if txt[0] == '{':
        return json.loads(txt)
    lines = [json.loads(x) for x in txt.splitlines() if x.strip()]
    if len(lines) == 1:
        return lines[0]
    return {"annotations": lines}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_word_poly(word):
    verts = word.get('vertices', [])
    if not verts:
        return None
    if isinstance(verts[0], dict):
        pts = [(float(v['x']), float(v['y'])) for v in verts]
    else:
        pts = [(float(v[0]), float(v[1])) for v in verts]
    return pts if len(pts) >= 3 else None


def poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Bezier helpers (same logic as convert_hiertext.py)
# ---------------------------------------------------------------------------

def _lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _vertices_to_pixels(verts, img_w, img_h):
    if isinstance(verts[0], dict):
        pts_raw = [[v["x"], v["y"]] for v in verts]
    else:
        pts_raw = [[v[0], v[1]] for v in verts]
    xs_raw = [p[0] for p in pts_raw]
    ys_raw = [p[1] for p in pts_raw]
    is_normalized = (max(xs_raw) <= 1.0 + 1e-6) and (max(ys_raw) <= 1.0 + 1e-6)
    if is_normalized:
        pts = [[p[0] * img_w, p[1] * img_h] for p in pts_raw]
    else:
        pts = pts_raw
    return pts


def _reorder_quad(pts):
    by_y = sorted(pts, key=lambda p: p[1])
    top_two = sorted(by_y[:2], key=lambda p: p[0])
    bot_two = sorted(by_y[2:], key=lambda p: p[0])
    TL, TR = top_two[0], top_two[1]
    BL, BR = bot_two[0], bot_two[1]
    return [TL, TR, BR, BL]


def _principal_axis(pts):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    sxx = sum((p[0]-cx)**2 for p in pts)
    syy = sum((p[1]-cy)**2 for p in pts)
    sxy = sum((p[0]-cx)*(p[1]-cy) for p in pts)
    diff = sxx - syy
    angle = 0.5*math.atan2(2.0*sxy, diff) if (diff!=0 or sxy!=0) else 0.0
    return math.cos(angle), math.sin(angle), cx, cy


def _eval_cubic(p0, p1, p2, p3, t):
    mt = 1.0 - t
    mt2 = mt*mt; t2 = t*t
    a = mt2*mt; b = 3.0*mt2*t; c = 3.0*mt*t2; d = t*t2
    return [a*p0[0]+b*p1[0]+c*p2[0]+d*p3[0], a*p0[1]+b*p1[1]+c*p2[1]+d*p3[1]]


def _segments_intersect(a1, a2, b1, b2, eps=1e-6):
    def _orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def _on_seg(p, q, r):
        return (min(p[0],r[0])-eps <= q[0] <= max(p[0],r[0])+eps and
                min(p[1],r[1])-eps <= q[1] <= max(p[1],r[1])+eps)
    o1=_orient(a1,a2,b1); o2=_orient(a1,a2,b2)
    o3=_orient(b1,b2,a1); o4=_orient(b1,b2,a2)
    if o1*o2<0 and o3*o4<0: return True
    if abs(o1)<eps and _on_seg(a1,b1,a2): return True
    if abs(o2)<eps and _on_seg(a1,b2,a2): return True
    if abs(o3)<eps and _on_seg(b1,a1,b2): return True
    if abs(o4)<eps and _on_seg(b1,a2,b2): return True
    return False


def _ring_has_self_intersection(bezier, n_samples=40):
    pts_ctrl = [[bezier[2*i], bezier[2*i+1]] for i in range(8)]
    top = pts_ctrl[0:4]; bot = pts_ctrl[4:8]
    ts = [i/(n_samples-1) for i in range(n_samples)]
    top_s = [_eval_cubic(*top, t) for t in ts]
    bot_s = [_eval_cubic(*bot, t) for t in ts]; bot_s.reverse()
    ring = top_s + bot_s
    if ring[0] != ring[-1]: ring.append(ring[0])
    m = len(ring)
    for i in range(m-1):
        a1,a2 = ring[i],ring[i+1]
        for j in range(i+2, m-1):
            if j==i or j==i+1: continue
            b1,b2 = ring[j],ring[j+1]
            if _segments_intersect(a1,a2,b1,b2): return True
    return False


def _bbox_bezier_from_pts(pts):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    TL=[xmin,ymin]; TR=[xmax,ymin]; BR=[xmax,ymax]; BL=[xmin,ymax]
    top_p0=TL; top_p3=TR
    top_p1=_lerp(TL,TR,1/3); top_p2=_lerp(TL,TR,2/3)
    bot_p0=BR; bot_p3=BL
    bot_p1=_lerp(BR,BL,1/3); bot_p2=_lerp(BR,BL,2/3)
    bezier=[coord for p in [top_p0,top_p1,top_p2,top_p3,bot_p0,bot_p1,bot_p2,bot_p3] for coord in p]
    return bezier


def _curve_to_bezier(curve):
    curve = np.asarray(curve, dtype=float).reshape(-1,2)
    m = len(curve)
    if m <= 1: return np.vstack([curve[0]]*4)
    if m == 2:
        p0,p3 = curve[0],curve[1]
        p1=_lerp(p0,p3,1/3); p2=_lerp(p0,p3,2/3)
        return np.vstack([p0,p1,p2,p3])
    diff = curve[1:]-curve[:-1]
    dist = np.linalg.norm(diff,axis=-1)
    total = dist.sum()
    if total <= 1e-6: return np.vstack([curve[0]]*4)
    norm = dist/total
    norm = np.concatenate([[0.0],norm])
    t = norm.cumsum()
    B = np.stack([(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3], axis=1)
    ctrl = np.linalg.pinv(B).dot(curve)
    ctrl[0] = curve[0]; ctrl[-1] = curve[-1]
    return ctrl


def _poly_to_bezier(pts):
    pts = [list(p) for p in pts]
    if not pts: return [0.0]*16
    dedup = [pts[0]]
    for p in pts[1:]:
        if p != dedup[-1]: dedup.append(p)
    pts = dedup
    n = len(pts)
    if n == 1:
        p0=pts[0]; return [p0[0],p0[1]]*8
    if n == 2:
        p0,p3=pts[0],pts[1]
        top_p0=p0; top_p3=p3
        top_p1=_lerp(p0,p3,1/3); top_p2=_lerp(p0,p3,2/3)
        bot_p0,bot_p3=top_p3,top_p0
        bot_p1,bot_p2=top_p2,top_p1
        return [coord for p in [top_p0,top_p1,top_p2,top_p3,bot_p0,bot_p1,bot_p2,bot_p3] for coord in p]
    if n == 3:
        by_y=sorted(pts,key=lambda p:p[1])
        apex=by_y[0]; base=sorted(by_y[1:],key=lambda p:p[0])
        TL=apex; TR=apex; BL=base[0]; BR=base[1]
        top_p0=TL; top_p1=_lerp(TL,TR,1/3); top_p2=_lerp(TL,TR,2/3); top_p3=TR
        bot_p0=BR; bot_p1=_lerp(BR,BL,1/3); bot_p2=_lerp(BR,BL,2/3); bot_p3=BL
        bezier=[coord for p in [top_p0,top_p1,top_p2,top_p3,bot_p0,bot_p1,bot_p2,bot_p3] for coord in p]
        return bezier
    if n == 4:
        TL,TR,BR,BL=_reorder_quad(pts)
        top_p0=TL; top_p3=TR; top_p1=_lerp(TL,TR,1/3); top_p2=_lerp(TL,TR,2/3)
        bot_p0=BR; bot_p3=BL; bot_p1=_lerp(BR,BL,1/3); bot_p2=_lerp(BR,BL,2/3)
        bezier=[coord for p in [top_p0,top_p1,top_p2,top_p3,bot_p0,bot_p1,bot_p2,bot_p3] for coord in p]
        if _ring_has_self_intersection(bezier): return _bbox_bezier_from_pts(pts)
        return bezier
    # n > 4
    def _left_key(p): return (p[0],p[1])
    def _right_key(p): return (-p[0],p[1])
    left_idx  = min(range(n), key=lambda i: _left_key(pts[i]))
    right_idx = min(range(n), key=lambda i: _right_key(pts[i]))
    if left_idx <= right_idx:
        chain1 = pts[left_idx:right_idx+1]
        chain2 = pts[right_idx:] + pts[:left_idx+1]
    else:
        chain1 = pts[left_idx:] + pts[:right_idx+1]
        chain2 = pts[right_idx:left_idx+1]
    def _mean_y(c): return sum(p[1] for p in c)/max(1,len(c))
    if _mean_y(chain1) <= _mean_y(chain2):
        top_chain,bot_chain = chain1,chain2
    else:
        top_chain,bot_chain = chain2,chain1
    if len(top_chain)>=2 and top_chain[0][0] > top_chain[-1][0]:
        top_chain = list(reversed(top_chain))
    if len(bot_chain)>=2 and bot_chain[0][0] < bot_chain[-1][0]:
        bot_chain = list(reversed(bot_chain))
    top_ctrl = _curve_to_bezier(top_chain)
    bot_ctrl  = _curve_to_bezier(bot_chain)
    top_p0,top_p1,top_p2,top_p3 = top_ctrl.tolist()
    bot_p0,bot_p1,bot_p2,bot_p3 = bot_ctrl.tolist()
    bezier=[coord for p in [top_p0,top_p1,top_p2,top_p3,bot_p0,bot_p1,bot_p2,bot_p3] for coord in p]
    if _ring_has_self_intersection(bezier): bezier=_bbox_bezier_from_pts(pts)
    return bezier


def sample_bezier_curve(p0, p1, p2, p3, n_pts=40):
    """Campiona n_pts punti su una curva di Bezier cubica."""
    ts = np.linspace(0, 1, n_pts)
    pts = []
    for t in ts:
        mt = 1.0 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def bezier_to_ring_points(bezier_flat, n_pts=40):
    """Restituisce i punti del ring (top + bottom) da un array piatto di 16 valori."""
    ctrl = [[bezier_flat[2*i], bezier_flat[2*i+1]] for i in range(8)]
    top = sample_bezier_curve(ctrl[0], ctrl[1], ctrl[2], ctrl[3], n_pts)
    bot = sample_bezier_curve(ctrl[4], ctrl[5], ctrl[6], ctrl[7], n_pts)
    return top, bot


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_label(draw, x, y, text, font, fg=(255,255,255), bg=(0,0,0)):
    text = str(text)
    try:
        box = draw.textbbox((x,y), text, font=font)
        tw = box[2]-box[0]; th = box[3]-box[1]
    except Exception:
        tw,th = draw.textsize(text,font=font)
    pad = 2
    draw.rectangle([x, y, x+tw+pad*2, y+th+pad*2], fill=bg)
    draw.text((x+pad, y+pad), text, fill=fg, font=font)


def draw_word_full(draw, word, font, img_w, img_h, scale=1.0):
    """
    Disegna per una singola parola:
      - Poligono originale (vertici) in VERDE
      - Curva Bezier calcolata (ring top+bottom) in ROSSO
      - Bounding box della Bezier in BLU
      - Label testo
    """
    verts = word.get('raw_vertices', [])
    label = word.get('label', '')
    n_verts = word.get('n_verts', 0)

    if not verts:
        return

    # 1. Poligono originale - VERDE
    COLOR_POLY   = (0, 220, 80)    # verde
    COLOR_BEZIER = (255, 60, 60)   # rosso
    COLOR_CTRL   = (255, 180, 0)   # arancione (punti di controllo)
    COLOR_BBOX   = (60, 140, 255)  # blu

    poly_px = [(int(p[0]*scale), int(p[1]*scale)) for p in verts]
    # Chiudi il poligono
    draw.line(poly_px + [poly_px[0]], fill=COLOR_POLY, width=2)
    # Disegna i vertici del poligono come cerchietti
    for px, py in poly_px:
        r = 4
        draw.ellipse([px-r, py-r, px+r, py+r], outline=COLOR_POLY, fill=(0,200,60), width=2)

    # 2. Curva Bezier - ROSSO
    bezier = word.get('bezier', None)
    if bezier is not None:
        top_pts, bot_pts = bezier_to_ring_points(bezier, n_pts=50)
        top_px = [(int(p[0]*scale), int(p[1]*scale)) for p in top_pts]
        bot_px = [(int(p[0]*scale), int(p[1]*scale)) for p in bot_pts]
        if len(top_px) > 1:
            draw.line(top_px, fill=COLOR_BEZIER, width=3)
        if len(bot_px) > 1:
            draw.line(bot_px, fill=COLOR_BEZIER, width=3)
        # Linee di chiusura sinistra/destra del ring
        if top_px and bot_px:
            draw.line([top_px[0], bot_px[-1]], fill=COLOR_BEZIER, width=2)
            draw.line([top_px[-1], bot_px[0]], fill=COLOR_BEZIER, width=2)

        # Punti di controllo Bezier (8 punti, arancione)
        ctrl_pts = [[bezier[2*i]*scale, bezier[2*i+1]*scale] for i in range(8)]
        for cx, cy in ctrl_pts:
            r = 5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=COLOR_CTRL, fill=COLOR_CTRL, width=2)
        # Tangenti del primo e secondo gruppo di controllo
        for group_start in [0, 4]:
            cp = ctrl_pts[group_start:group_start+4]
            draw.line([(int(cp[0][0]),int(cp[0][1])),(int(cp[1][0]),int(cp[1][1]))],
                      fill=COLOR_CTRL, width=1)
            draw.line([(int(cp[2][0]),int(cp[2][1])),(int(cp[3][0]),int(cp[3][1]))],
                      fill=COLOR_CTRL, width=1)

        # 3. Bounding box - BLU
        xs_ctrl = bezier[0::2]; ys_ctrl = bezier[1::2]
        xmin = min(xs_ctrl)*scale; xmax = max(xs_ctrl)*scale
        ymin = min(ys_ctrl)*scale; ymax = max(ys_ctrl)*scale
        draw.rectangle([int(xmin), int(ymin), int(xmax), int(ymax)],
                       outline=COLOR_BBOX, width=2)

    # 4. Label testo + numero vertici
    x0, y0, _, _ = poly_bbox(verts)
    ly = max(32, int(y0*scale) - 22)
    draw_label(draw, int(x0*scale), ly, f'{label} [{n_verts}v]', font,
               fg=(255,255,255), bg=(30,30,30))


# ---------------------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------------------

def open_image(images_dir, image_id, file_name, width, height):
    if images_dir:
        candidates = []
        if file_name:
            candidates.append(Path(images_dir) / file_name)
        candidates += [
            Path(images_dir) / f'{image_id}.jpg',
            Path(images_dir) / f'{image_id}.jpeg',
            Path(images_dir) / f'{image_id}.png',
        ]
        for p in candidates:
            if p.exists():
                return Image.open(p).convert('RGB')
    return Image.new('RGB', (width, height), (60, 60, 80))


def draw_panel_full(base_img, words, title, scale=1.0):
    """
    Panel con visualizzazione completa:
    - poligono vertici (verde)
    - curva bezier (rosso)
    - bounding box bezier (blu)
    - punti di controllo (arancione)
    """
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title_font = load_font(18)

    # Header
    draw.rectangle([0, 0, img.width, 32], fill=(20, 20, 30))
    draw.text((10, 6), title, fill=(255, 255, 255), font=title_font)

    # Legenda
    legend_y = img.height - 28
    legend_items = [
        ((0, 220, 80),    'Poligono vertici'),
        ((255, 60, 60),   'Curva Bezier'),
        ((255, 180, 0),   'Punti di controllo'),
        ((60, 140, 255),  'Bounding box'),
    ]
    lx = 10
    for color, lbl in legend_items:
        draw.rectangle([lx, legend_y+4, lx+16, legend_y+18], fill=color)
        draw.text((lx+20, legend_y+2), lbl, fill=(220,220,220), font=font)
        lx += 140

    for w in words:
        draw_word_full(draw, w, font, img.width, img.height, scale=scale)

    return img


def make_canvas(panel, title_text, n_verts_label):
    """Crea canvas singolo con header informativo."""
    W, H = panel.width, panel.height + 50
    canvas = Image.new('RGB', (W, H), (30, 30, 40))
    draw = ImageDraw.Draw(canvas)
    font = load_font(20)
    small = load_font(13)
    draw.rectangle([0, 0, W, 42], fill=(15, 15, 25))
    draw.text((12, 8), title_text, fill=(240, 200, 80), font=font)
    draw.text((12, H-22), n_verts_label, fill=(180,180,200), font=small)
    canvas.paste(panel, (0, 42))
    return canvas


# ---------------------------------------------------------------------------
# Parsing source data
# ---------------------------------------------------------------------------

def parse_original_words_with_bezier(data, ctlabels, pad_token, img_w, img_h, lowercase=False):
    """
    Legge le annotazioni HierText e calcola anche la curva bezier per ogni parola.
    Restituisce una lista di dict con: poly, label, raw_vertices, bezier, n_verts.
    """
    words = []
    for sample in data.get('annotations', []):
        s_img_w = int(sample.get('image_width', img_w) or img_w)
        s_img_h = int(sample.get('image_height', img_h) or img_h)
        for para in sample.get('paragraphs', []):
            for line in para.get('lines', []):
                for word in line.get('words', []):
                    verts = word.get('vertices', [])
                    if not verts or len(verts) < 3:
                        continue
                    pts_px = _vertices_to_pixels(verts, s_img_w, s_img_h)
                    bezier = _poly_to_bezier(pts_px)
                    raw_text = word.get('text', '')
                    legible = word.get('legible', True)
                    rec = text_to_rec(raw_text, ctlabels, pad_token, lowercase=lowercase)
                    dec = decode_rec(rec, ctlabels, pad_token)
                    if not legible and not dec: dec = '[illegible]'
                    elif not dec: dec = '[empty]'
                    words.append({
                        'poly': [(p[0], p[1]) for p in pts_px],
                        'raw_vertices': pts_px,
                        'label': dec,
                        'bezier': bezier,
                        'n_verts': len(pts_px),
                    })
    return words


def parse_source_images(data):
    """Estrae le immagini dall'JSONL HierText con dimensioni."""
    items = []
    for sample in data.get('annotations', []):
        items.append({
            'image_id': sample.get('image_id', ''),
            'width': int(sample.get('image_width', 1280) or 1280),
            'height': int(sample.get('image_height', 720) or 720),
            'sample': sample,
        })
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            'Visualizza le annotazioni HierText con:\n'
            '  - Poligono originale (vertici, verde)\n'
            '  - Curva Bezier calcolata (rosso)\n'
            '  - Punti di controllo Bezier (arancione)\n'
            '  - Bounding box Bezier (blu)\n\n'
            'Genera 2 immagini: una con un poligono a 4 vertici, '
            'le altre con poligoni a più di 4 vertici.'
        )
    )
    ap.add_argument('--jsonl', required=True,
                    help='HierText source json/jsonl')
    ap.add_argument('--images', default='',
                    help='Directory immagini (opzionale)')
    ap.add_argument('--out', default='debug_bezier_output',
                    help='Directory di output')
    ap.add_argument('--n-poly4', type=int, default=1,
                    help='Quante immagini con poligoni a 4 vertici (default: 1)')
    ap.add_argument('--n-poly5plus', type=int, default=4,
                    help='Quante immagini con poligoni a >4 vertici (default: 4)')
    ap.add_argument('--max-words', type=int, default=30,
                    help='Max parole da disegnare per immagine (default: 30)')
    ap.add_argument('--voc', type=int, default=96, choices=[37, 96])
    args = ap.parse_args()

    ctlabels, blank_token, pad_token = build_vocab(args.voc)
    lowercase = (args.voc == 37)

    print(f'[debug_bezier] voc_size={args.voc}  blank={blank_token}  pad={pad_token}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_data = load_json_or_jsonl(args.jsonl)
    image_items = parse_source_images(src_data)

    count_4 = 0
    count_5p = 0

    for item in image_items:
        if count_4 >= args.n_poly4 and count_5p >= args.n_poly5plus:
            break

        image_id = item['image_id']
        img_w = item['width']
        img_h = item['height']

        # Recupera tutte le parole con Bezier calcolata
        fake_data = {'annotations': [item['sample']]}
        words = parse_original_words_with_bezier(
            fake_data, ctlabels, pad_token, img_w, img_h, lowercase=lowercase
        )

        if not words:
            continue

        # Conta quante parole hanno 4 e quante >4 vertici
        words_4  = [w for w in words if w['n_verts'] == 4]
        words_5p = [w for w in words if w['n_verts'] > 4]

        has_4  = len(words_4) > 0
        has_5p = len(words_5p) > 0

        # Decidi se usare questa immagine
        use_for_4  = has_4  and count_4  < args.n_poly4
        use_for_5p = has_5p and count_5p < args.n_poly5plus

        if not use_for_4 and not use_for_5p:
            continue

        # Carica immagine base
        base = open_image(args.images, image_id, image_id+'.jpg', img_w, img_h)

        # --- OUTPUT per poligoni a 4 vertici ---
        if use_for_4:
            # Seleziona solo parole con 4 vertici (max max_words)
            selected = words_4[:args.max_words]
            panel = draw_panel_full(
                base, selected,
                f'id={image_id} | Solo quad (4 vertici) | n={len(selected)}'
            )
            n_verts_all = sorted(set(w['n_verts'] for w in selected))
            info = f'Vertici per parola: {n_verts_all}  |  Algo: _reorder_quad -> lerp 1/3, 2/3'
            canvas = make_canvas(panel, f'Bezier da poligoni a 4 vertici — {image_id}', info)
            out_path = out_dir / f'{image_id}_quad4.png'
            canvas.save(out_path)
            print(f'  [4-vert]  -> {out_path}  ({len(selected)} parole)')
            count_4 += 1

        # --- OUTPUT per poligoni a >4 vertici ---
        if use_for_5p:
            selected = words_5p[:args.max_words]
            panel = draw_panel_full(
                base, selected,
                f'id={image_id} | Solo poligoni (>4 vertici) | n={len(selected)}'
            )
            n_verts_all = sorted(set(w['n_verts'] for w in selected))
            info = f'Vertici per parola: {n_verts_all}  |  Algo: chain split (left/right) -> least squares'
            canvas = make_canvas(panel, f'Bezier da poligoni >4 vertici — {image_id}', info)
            out_path = out_dir / f'{image_id}_poly5plus.png'
            canvas.save(out_path)
            print(f'  [5+vert] -> {out_path}  ({len(selected)} parole)')
            count_5p += 1

    print(f'\nCompletato: {count_4} immagini quad4, {count_5p} immagini poly5+')
    print(f'Output in: {out_dir}')
    print()
    print('Legenda colori:')
    print('  VERDE   = poligono originale (vertici HierText)')
    print('  ROSSO   = curva Bezier calcolata (ring top+bottom)')
    print('  ARANCIO = punti di controllo Bezier (8 punti)')
    print('  BLU     = bounding box Bezier (min/max dei punti di controllo)')
    print()
    print('Label: "testo [Nv]" dove N = numero di vertici del poligono originale')


if __name__ == '__main__':
    main()
