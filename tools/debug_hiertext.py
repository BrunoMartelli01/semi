import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Vocabolari supportati
# ---------------------------------------------------------------------------
CTLABELS_96 = [chr(i) for i in range(32, 127)]   # 95 chars, blank=95, pad=96
CTLABELS_37 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                'q','r','s','t','u','v','w','x','y','z',
                '0','1','2','3','4','5','6','7','8','9']  # 36 chars, blank=36, pad=37


def build_vocab(voc_size: int):
    if voc_size == 96:
        labels = CTLABELS_96
    elif voc_size == 37:
        labels = CTLABELS_37
    else:
        raise ValueError(f"Unsupported voc_size {voc_size}. Use 96 or 37.")
    blank = len(labels)
    pad   = len(labels) + 1
    return labels, blank, pad


def text_to_rec(text: str, ctlabels, pad_token: int, max_len: int = 25,
                lowercase: bool = False) -> list:
    if lowercase:
        text = str(text).lower()
    rec = []
    for c in str(text):
        if c in ctlabels:
            rec.append(ctlabels.index(c))
    rec = rec[:max_len]
    rec += [pad_token] * (max_len - len(rec))
    return rec


def decode_rec(rec, ctlabels, pad_token: int) -> str:
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


def coco_ann_poly(ann):
    seg = ann.get('segmentation', [])
    if seg and seg[0]:
        s = seg[0]
        if len(s) >= 6 and len(s) % 2 == 0:
            return [(float(s[i]), float(s[i + 1])) for i in range(0, len(s), 2)]
    bbox = ann.get('bbox', None)
    if bbox and len(bbox) == 4:
        x, y, w, h = bbox
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return None


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------

def sample_cubic_bezier(ctrl_pts, n=80):
    """
    Campiona n punti su una curva di Bezier cubica.
    ctrl_pts: array (4, 2) dei punti di controllo.
    Restituisce array (n, 2) di int.
    """
    p = np.array(ctrl_pts, dtype=np.float64)
    u = np.linspace(0.0, 1.0, n)[:, None]
    curve = (
        ((1 - u) ** 3)            * p[0] +
        (3 * u * (1 - u) ** 2)    * p[1] +
        (3 * u ** 2 * (1 - u))    * p[2] +
        (u ** 3)                  * p[3]
    )
    return curve.astype(np.int32)


def draw_bezier_on_panel(draw, bezier_pts_flat,
                         color_top=(255, 50, 50),
                         color_bot=(50, 50, 255),
                         color_ctrl=(255, 200, 0),
                         width=2, dot_r=4):
    """
    Disegna le due curve di Bezier (bordo top rosso, bordo bot blu)
    e i punti di controllo gialli.
    bezier_pts_flat: lista di 16 float [top_4pts(8val) + bot_4pts(8val)].
    """
    if bezier_pts_flat is None or len(bezier_pts_flat) != 16:
        return
    pts = np.array(bezier_pts_flat, dtype=np.float64).reshape(8, 2)
    top_ctrl = pts[:4]
    bot_ctrl = pts[4:]

    for ctrl, color in [(top_ctrl, color_top), (bot_ctrl, color_bot)]:
        curve = sample_cubic_bezier(ctrl, n=80)
        curve_xy = [(int(p[0]), int(p[1])) for p in curve]
        if len(curve_xy) >= 2:
            draw.line(curve_xy, fill=color, width=width)
        for cp in ctrl:
            cx, cy = int(cp[0]), int(cp[1])
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                         fill=color_ctrl, outline=(0, 0, 0))
        # Segmenti tangenti tra punti di controllo
        pts_xy = [(int(p[0]), int(p[1])) for p in ctrl]
        draw.line(pts_xy, fill=color, width=1)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_label(draw, x, y, text, font, fg=(255, 255, 255), bg=(0, 0, 0)):
    text = str(text)
    try:
        box = draw.textbbox((x, y), text, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    pad = 2
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=bg)
    draw.text((x + pad, y + pad), text, fill=fg, font=font)


# ---------------------------------------------------------------------------
# Parsing source / COCO data
# ---------------------------------------------------------------------------

def parse_original_words(data, ctlabels, pad_token, lowercase=False):
    items = []
    for sample in data.get('annotations', []):
        image_id = sample.get('image_id', '')
        width  = int(sample.get('image_width',  1280) or 1280)
        height = int(sample.get('image_height',  720) or  720)
        words = []
        for para in sample.get('paragraphs', []):
            for line in para.get('lines', []):
                for word in line.get('words', []):
                    poly = get_word_poly(word)
                    if not poly:
                        continue
                    raw_text = word.get('text', '')
                    legible  = word.get('legible', True)
                    rec = text_to_rec(raw_text, ctlabels, pad_token,
                                      lowercase=lowercase)
                    dec = decode_rec(rec, ctlabels, pad_token)
                    if not legible and not dec:
                        dec = '[illegible]'
                    elif not dec:
                        dec = '[empty]'
                    words.append({'poly': poly, 'label': dec})
        items.append({
            'image_id': image_id,
            'width': width,
            'height': height,
            'words': words,
        })
    return items


def parse_coco(data, ctlabels, pad_token):
    images = {img['id']: img for img in data.get('images', [])}
    anns_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        anns_by_img[ann['image_id']].append(ann)
    items = []
    for img_id, img in images.items():
        words = []
        for ann in anns_by_img.get(img_id, []):
            poly = coco_ann_poly(ann)
            if not poly:
                continue
            rec = ann.get('rec', [])
            dec = decode_rec(rec, ctlabels, pad_token) if rec else ann.get('text', '')
            if not dec:
                dec = '[empty]'
            words.append({
                'poly':       poly,
                'label':      dec,
                'bezier_pts': ann.get('bezier_pts', None),
            })
        items.append({
            'image_id':  Path(img.get('file_name', '')).stem,
            'file_name': img.get('file_name', ''),
            'width':     int(img.get('width',  1280) or 1280),
            'height':    int(img.get('height',  720) or  720),
            'words':     words,
        })
    return items


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
    return Image.new('RGB', (width, height), (128, 128, 128))


def _draw_legend(draw, img_h, img_w, small):
    legend_y = img_h - 60
    draw.rectangle([0, legend_y, img_w, img_h], fill=(20, 20, 20))
    draw.rectangle([8,  legend_y + 6,  20, legend_y + 16], fill=(255, 140, 0))
    draw.text((24, legend_y + 4),  'Poligono originale (vertici > 4)', fill=(220, 220, 220), font=small)
    draw.rectangle([8,  legend_y + 22, 20, legend_y + 32], fill=(255, 50, 50))
    draw.text((24, legend_y + 20), 'Bezier top',    fill=(220, 220, 220), font=small)
    draw.rectangle([8,  legend_y + 38, 20, legend_y + 48], fill=(50, 50, 255))
    draw.text((24, legend_y + 36), 'Bezier bottom', fill=(220, 220, 220), font=small)
    draw.ellipse([200, legend_y + 20, 212, legend_y + 32], fill=(255, 200, 0), outline=(0, 0, 0))
    draw.text((216, legend_y + 20), 'Punti di controllo', fill=(220, 220, 220), font=small)


def draw_panel_original(base_img, words, title, color_poly, color_text):
    """Pannello sinistro: poligono originale HierText."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(16)
    title_font = load_font(20)
    draw.rectangle([0, 0, img.width, 30], fill=(20, 20, 20))
    draw.text((10, 5), title, fill=(255, 255, 255), font=title_font)
    for w in words:
        pts = w['poly']
        draw.line(pts + [pts[0]], fill=color_poly, width=3)
        x0, y0, _, _ = poly_bbox(pts)
        ly = max(32, int(y0) - 22)
        draw_label(draw, int(x0), ly, w['label'], font,
                   fg=(255, 255, 255), bg=color_text)
    return img


def draw_panel_bezier(base_img, words, title, show_only_complex_poly=True):
    """
    Pannello con:
      - Poligono originale (segmentation) in arancione, solo se vertici > 4
      - Curve Bezier top (rosso) e bottom (blu) con punti di controllo (giallo)
    """
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    font  = load_font(16)
    small = load_font(13)
    title_font = load_font(20)

    draw.rectangle([0, 0, img.width, 30], fill=(20, 20, 20))
    draw.text((10, 5), title, fill=(255, 255, 255), font=title_font)
    _draw_legend(draw, img.height, img.width, small)

    for w in words:
        pts     = w['poly']
        bez_pts = w.get('bezier_pts', None)
        label   = w['label']
        n_verts = len(pts)

        if not show_only_complex_poly or n_verts > 4:
            poly_color = (255, 140, 0)
            draw.line(pts + [pts[0]], fill=poly_color, width=2)
            for idx, (vx, vy) in enumerate(pts):
                r = 3
                draw.ellipse([vx-r, vy-r, vx+r, vy+r], fill=poly_color)
                draw.text((int(vx)+4, int(vy)-8), str(idx),
                          fill=poly_color, font=small)

        draw_bezier_on_panel(
            draw, bez_pts,
            color_top=(255, 50, 50),
            color_bot=(50, 50, 255),
            color_ctrl=(255, 200, 0),
            width=2, dot_r=4
        )

        x0, y0, _, _ = poly_bbox(pts)
        ly = max(32, int(y0) - 22)
        draw_label(draw, int(x0), ly, f'[{n_verts}v] {label}', font,
                   fg=(255, 255, 255), bg=(20, 80, 180))

    return img


def draw_single_panel(base_img, words, title, show_only_complex_poly=True):
    """
    Pannello unico (modalita' coco-only): mostra poligoni + Bezier
    sull'immagine originale senza confronto fianco-a-fianco.
    """
    return draw_panel_bezier(base_img, words, title,
                             show_only_complex_poly=show_only_complex_poly)


def stack_side_by_side(left, right, header):
    w = left.width + right.width + 8
    h = max(left.height, right.height) + 70
    canvas = Image.new('RGB', (w, h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)
    font   = load_font(22)
    small  = load_font(16)
    draw.rectangle([0, 0, w, 42], fill=(35, 35, 35))
    draw.text((12, 8), header, fill=(255, 255, 255), font=font)
    canvas.paste(left,  (0, 42))
    canvas.paste(right, (left.width + 8, 42))
    draw.rectangle([left.width, 42, left.width + 8, h], fill=(220, 220, 220))
    footer = ('Sinistra: poligoni HierText originali'
              ' | Destra: poligoni (arancio, >4 vertici) + curve Bezier (rosso/blu)')
    draw.text((12, h - 22), footer, fill=(20, 20, 20), font=small)
    return canvas


def add_header(img, header_text):
    """Aggiunge un header in cima a un'immagine singola."""
    h = img.height + 42
    canvas = Image.new('RGB', (img.width, h), (35, 35, 35))
    draw   = ImageDraw.Draw(canvas)
    font   = load_font(20)
    draw.text((12, 8), header_text, fill=(255, 255, 255), font=font)
    canvas.paste(img, (0, 42))
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            'Visualizza annotazioni con curve Bezier su immagini.\n'
            '\n'
            'Modalita\' completa (HierText):  --jsonl + --coco\n'
            'Modalita\' solo COCO (CTW1500):  --coco  (--jsonl omesso)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--jsonl', default='',
                    help='HierText source json/jsonl (annotazione originale). '
                         'Ometti per usare la modalita\' solo COCO.')
    ap.add_argument('--coco', required=True,
                    help='COCO json con bezier_pts (es. train_96voc.json, hiertext_val_deepsolo.json)')
    ap.add_argument('--images', default='',
                    help='Directory immagini (opzionale, grigio se assente)')
    ap.add_argument('--out', default='debug_output',
                    help='Directory di output per le immagini debug')
    ap.add_argument('--n', type=int, default=5,
                    help='Numero di immagini da esportare')
    ap.add_argument('--voc', type=int, default=96, choices=[37, 96],
                    help='Vocabulary size: 96 (default) o 37 (legacy a-z0-9)')
    ap.add_argument('--all-poly', action='store_true',
                    help='Mostra tutti i poligoni, non solo quelli con >4 vertici')
    args = ap.parse_args()

    coco_only = (args.jsonl == '')

    ctlabels, blank_token, pad_token = build_vocab(args.voc)
    lowercase = (args.voc == 37)

    mode_str = 'coco-only' if coco_only else 'hiertext+coco'
    print(f'[debug_hiertext] mode={mode_str}  voc_size={args.voc}  '
          f'blank={blank_token}  pad={pad_token}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_data  = load_json_or_jsonl(args.coco)
    coco_items = parse_coco(coco_data, ctlabels, pad_token)

    show_only_complex = not args.all_poly

    # ------------------------------------------------------------------
    # Modalita' solo COCO (CTW1500, ecc.): un pannello unico per immagine
    # ------------------------------------------------------------------
    if coco_only:
        count = 0
        for item in coco_items:
            stem      = item['image_id']
            file_name = item.get('file_name', '')
            base = open_image(args.images, stem, file_name,
                              item['width'], item['height'])

            complex_count = sum(1 for w in item['words'] if len(w['poly']) > 4)
            bez_count     = sum(1 for w in item['words'] if w.get('bezier_pts') is not None)

            panel = draw_single_panel(
                base, item['words'],
                title=f'BEZIER  |  id={stem}  |  ann={len(item["words"])}'
                      f'  poly>4v={complex_count}  bez={bez_count}',
                show_only_complex_poly=show_only_complex
            )
            board = add_header(
                panel,
                f'{stem}  |  ann={len(item["words"])}  '
                f'poly>4v={complex_count}  bezier_pts={bez_count}'
            )

            out_path = out_dir / f'{stem}_debug.png'
            board.save(out_path)
            print(f'  {stem}: ann={len(item["words"])}, '
                  f'poly>4v={complex_count}, con bezier_pts={bez_count}')
            count += 1
            if count >= args.n:
                break
        print(f'\nSalvate {count} immagini in {out_dir}')
        return

    # ------------------------------------------------------------------
    # Modalita' completa HierText: confronto fianco-a-fianco
    # ------------------------------------------------------------------
    src_data   = load_json_or_jsonl(args.jsonl)
    src_items  = parse_original_words(src_data, ctlabels, pad_token, lowercase=lowercase)
    coco_by_stem = {item['image_id']: item for item in coco_items}

    count = 0
    for src in src_items:
        stem = src['image_id']
        coco = coco_by_stem.get(stem)
        if coco is None:
            continue

        base = open_image(args.images, stem, coco.get('file_name', ''),
                          src['width'], src['height'])

        left = draw_panel_original(
            base, src['words'],
            f'ORIGINALE - parole={len(src["words"])}',
            color_poly=(0, 180, 0), color_text=(0, 100, 0)
        )
        right = draw_panel_bezier(
            base, coco['words'],
            f'BEZIER + POLIGONI ({"solo >4v" if show_only_complex else "tutti"})'
            f'  -  ann={len(coco["words"])}',
            show_only_complex_poly=show_only_complex
        )

        board = stack_side_by_side(
            left, right,
            f'id={stem}  |  originale={len(src["words"])}  |  coco={len(coco["words"])}'
        )

        complex_count = sum(1 for w in coco['words'] if len(w['poly']) > 4)
        bez_count     = sum(1 for w in coco['words'] if w.get('bezier_pts') is not None)
        print(f'  {stem}: coco={len(coco["words"])} ann, '
              f'poly>4v={complex_count}, con bezier_pts={bez_count}')

        out_path = out_dir / f'{stem}_debug.png'
        board.save(out_path)
        count += 1
        if count >= args.n:
            break

    print(f'\nSalvate {count} immagini in {out_dir}')


if __name__ == '__main__':
    main()
