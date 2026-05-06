import json, zipfile
from collections import defaultdict

with open('datasets/hiertext/test_gt_source.json') as f:
    data = json.load(f)

ann_by_img = defaultdict(list)
for ann in data['annotations']:
    ann_by_img[ann['image_id']].append(ann)

out_zip = 'datasets/evaluation/gt_hiertext.zip'
bad_lines_total = 0

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for img in data['images']:
        img_id   = img['id']
        filename = '{:07d}.txt'.format(img_id)
        lines    = []

        for ann in ann_by_img[img_id]:
            seg = ann['segmentation'][0]

            # salta se coordinate malformate
            if not seg or len(seg) % 2 != 0 or len(seg) < 6:
                continue

            coords  = ','.join(str(int(round(v))) for v in seg)
            text    = ann.get('text', '').strip().lower()
            ignored = ann.get('ignore', 0) == 1 or text == ''

            if ignored:
                line = f"{coords},#######"
            else:
                line = f"{coords},####{text}"

            # doppio controllo: la riga deve avere #### e coordinate non vuote
            if '####' not in line or line.startswith(','):
                bad_lines_total += 1
                continue

            lines.append(line)

        # scrivi senza riga vuota finale
        content = '\n'.join(lines)
        zf.writestr(filename, content)

print(f"GT scritto: {out_zip}")
print(f"Righe scartate per malformazione: {bad_lines_total}")

# verifica approfondita
print("\n--- Verifica ---")
with zipfile.ZipFile(out_zip) as zf:
    names = zf.namelist()
    print(f"File nel zip: {len(names)}")
    total_bad = 0
    for name in names:
        content = zf.read(name).decode('utf-8', errors='replace')
        if not content:
            continue  # file vuoto OK
        for line in content.split('\n'):
            line = line.strip()
            if line and '####' not in line:
                total_bad += 1
                print(f"  MALFORMATA in {name}: '{line[:80]}'")
    print(f"Totale righe malformate nel zip: {total_bad}")

    # mostra sample
    sample = names[0]
    lines  = zf.read(sample).decode().split('\n')
    print(f"\nSample {sample} ({len(lines)} righe):")
    for l in lines[:5]:
        print(f"  {l}")