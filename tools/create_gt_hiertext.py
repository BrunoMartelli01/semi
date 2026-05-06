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
                bad_lines_total += 1
                continue

            coords  = ','.join(str(int(round(v))) for v in seg)

            # salta se coords vuote o iniziano con virgola
            if not coords or coords.startswith(','):
                bad_lines_total += 1
                continue

            text    = ann.get('text', '').strip().lower()
            ignored = ann.get('ignore', 0) == 1 or text == ''

            if ignored:
                # righe ignorate: delimitatore #### con testo vuoto
                # il parser si aspetta ptr[1] dopo split(',####') quindi
                # il formato corretto e' coords,####
                line = f"{coords},####"
            else:
                # sanifica il testo: rimuovi caratteri che rompono il parsing
                # il parser usa split(',####') quindi #### nel testo e' pericoloso
                safe_text = text.replace('####', '').strip()
                if safe_text == '':
                    line = f"{coords},####"
                else:
                    line = f"{coords},####{safe_text}"

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
            if not line:
                continue
            parts = line.split(',####')
            # deve avere esattamente 2 parti: coords e testo (anche vuoto)
            if len(parts) != 2:
                total_bad += 1
                print(f"  MALFORMATA in {name}: '{line[:80]}'")
                continue
            coords_part = parts[0]
            if not coords_part or len(coords_part.split(',')) < 6:
                total_bad += 1
                print(f"  COORDS MALFORMATE in {name}: '{line[:80]}'")
    print(f"Totale righe malformate nel zip: {total_bad}")

    # mostra sample
    if names:
        sample = names[0]
        lines_sample = zf.read(sample).decode().split('\n')
        print(f"\nSample {sample} ({len(lines_sample)} righe):")
        for l in lines_sample[:5]:
            print(f"  {l}")
