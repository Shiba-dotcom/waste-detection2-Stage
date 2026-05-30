import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open(r'c:\Users\Asus\Downloads\waste-detection2-Stage\notebooks\final_sahi_kaggle.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

cells = data['cells']
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    ctype = c['cell_type']
    source = ''.join(c['source'])
    print(f"\n{'='*60}")
    print(f"Cell {i} ({ctype})")
    print(f"{'='*60}")
    print(source[:3000])
