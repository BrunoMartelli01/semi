import argparse
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
            't','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
VOC_SIZE = 37
PAD_TOKEN = VOC_SIZE - 1


def text_to_rec(text, max_len=25):
    rec = []
    for c in str(text).lower():
        if c in CTLABELS:
            rec.append(CTLABELS.index(c))
    rec = rec[:max_len]
    rec += [PAD_TOKEN] * (max_len - len(rec))
    return rec


def decode_rec(rec):
    last_char = None
    out = []
    for c in rec:
        c = int(c)
        if c < PAD_TOKEN:
            if last_char != c:
                out.append(CTLABELS[c])
                last_char = c
        else:
            last_char = None
    return ''.join(out)


def load_font(size=16):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


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


def draw_label(draw, x, y, text, font, fg=(255,255,255), bg=(0,0,0)):
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


def parse_original_words(data):
    items = []
    for sample in data.get('annotations', []):
        image_id = sample.get('image_id', '')
        width = int(sample.get('image_width', 1280) or 1280)
        height = int(sample.get('image_height', 720) or 720)
        words = []
        for para in sample.get('paragraphs', []):
            for line in para.get('lines', []):
                for word in line.get('words', []):
                    poly = get_word_poly(word)
                    if not poly:
                        continue
                    raw_text = word.get('text', '')
                    legible = word.get('legible', True)
                    rec = text_to_rec(raw_text)
                    dec = decode_rec(rec)
                    if not legible and not dec:
                        dec = '[illegible]'
                    elif not dec:
                        dec = '[empty]'
                    words.append({
                        'poly': poly,
                        'label': dec,
                    })
        items.append({
            'image_id': image_id,
            'width': width,
            'height': height,
            'words': words,
        })
    return items


def parse_coco(data):
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
            dec = decode_rec(rec) if rec else ann.get('text', '')
            if not dec:
                dec = '[empty]'
            words.append({
                'poly': poly,
                'label': dec,
            })
        items.append({
            'image_id': Path(img.get('file_name', '')).stem,
            'file_name': img.get('file_name', ''),
            'width': int(img.get('width', 1280) or 1280),
            'height': int(img.get('height', 720) or 720),
            'words': words,
        })
    return items


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


def draw_panel(base_img, words, title, color_poly, color_text):
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
        draw_label(draw, int(x0), ly, w['label'], font, fg=(255,255,255), bg=color_text)
    return img


def stack_side_by_side(left, right, header):
    w = left.width + right.width + 8
    h = max(left.height, right.height) + 70
    canvas = Image.new('RGB', (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small = load_font(16)
    draw.rectangle([0, 0, w, 42], fill=(35, 35, 35))
    draw.text((12, 8), header, fill=(255, 255, 255), font=font)
    canvas.paste(left, (0, 42))
    canvas.paste(right, (left.width + 8, 42))
    draw.rectangle([left.width, 42, left.width + 8, h], fill=(220, 220, 220))
    footer = 'Sinistra: originale con rec decodificato | Destra: JSON convertito con rec decodificato'
    draw.text((12, h - 22), footer, fill=(20, 20, 20), font=small)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='HierText source json/jsonl')
    ap.add_argument('--coco', required=True, help='Converted COCO json')
    ap.add_argument('--images', default='', help='Images directory')
    ap.add_argument('--out', default='debug_output', help='Output directory')
    ap.add_argument('--n', type=int, default=5, help='How many images to export')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_data = load_json_or_jsonl(args.jsonl)
    coco_data = load_json_or_jsonl(args.coco)

    src_items = parse_original_words(src_data)
    coco_items = parse_coco(coco_data)
    coco_by_stem = {item['image_id']: item for item in coco_items}

    count = 0
    for src in src_items:
        stem = src['image_id']
        coco = coco_by_stem.get(stem)
        if coco is None:
            continue
        base = open_image(args.images, stem, coco.get('file_name', ''), src['width'], src['height'])
        left = draw_panel(base, src['words'], f'PRIMA - boxes={len(src["words"])}', (0, 180, 0), (0, 120, 0))
        right = draw_panel(base, coco['words'], f'DOPO - boxes={len(coco["words"])}', (255, 140, 0), (20, 80, 180))
        board = stack_side_by_side(left, right, f'id={stem} | prima={len(src["words"])} | dopo={len(coco["words"])}')
        out_path = out_dir / f'{stem}_debug.png'
        board.save(out_path)
        count += 1
        if count >= args.n:
            break

    print(f'Salvate {count} immagini in {out_dir}')


if __name__ == '__main__':
    main()
