import json, zipfile, re
from collections import defaultdict

# Regex: almeno 3 coppie di interi separati da virgola, poi ,#### e testo opzionale
# Questo garantisce che il parser rrc_evaluation_funcs non vada mai in IndexError
VALID_LINE_RE = re.compile(
    r'^(-?\d+,-?\d+)(-?,-?\d+,-?\d+)+(,####.*)$'
)

def make_gt_line(ann):
    """Restituisce la stringa della riga GT oppure None se malformata."""
    seg = ann.get('segmentation', [[]])
    if not seg or not seg[0]:
        return None
    seg = seg[0]
    if len(seg) % 2 != 0 or len(seg) < 6:
        return None

    coords = ','.join(str(int(round(v))) for v in seg)
    if not coords or coords.startswith(','):
        return None

    text    = ann.get('text', '').strip().lower()
    ignored = ann.get('ignore', 0) == 1 or text == ''

    if ignored:
        line = f"{coords},####"
    else:
        # rimuovi qualsiasi occorrenza di #### nel testo (romperebbe lo split)
        safe_text = text.replace('####', '').strip()
        line = f"{coords},####{safe_text}" if safe_text else f"{coords},####"

    # validazione finale: la riga deve matchare il formato atteso dal parser
    if not VALID_LINE_RE.match(line):
        return None

    # doppio controllo: split su ,#### deve dare esattamente 2 parti
    parts = line.split(',####')
    if len(parts) != 2:
        return None
    if len(parts[0].split(',')) < 6:
        return None

    return line


with open('datasets/hiertext/test_gt_source.json') as f:
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

# --- verifica approfondita ---
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
            if len(parts) != 2 or len(parts[0].split(',')) < 6:
                total_bad += 1
                print(f"  MALFORMATA in {name}: '{line[:80]}'")
    print(f"Totale righe malformate nel zip: {total_bad}")
    if total_bad == 0:
        print("  \u2705 Tutte le righe sono valide")

    # mostra sample
    if names:
        sample = names[0]
        lines_sample = zf.read(sample).decode().split('\n')
        print(f"\nSample {sample} ({len(lines_sample)} righe):")
        for l in lines_sample[:5]:
            print(f"  {l}")
