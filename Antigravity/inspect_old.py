"""Inspect the old notebooks to understand what was tried."""
import json

nb = json.load(open('trials/notebooks/NFL_2_Ultimate.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = src.strip().split('\n')[0] if src.strip() else '(empty)'
    print(f"Cell {i}: type={c['cell_type']}, lines={len(c['source'])}")
    print(f"  First line: {first_line[:120]}")
    # Print any markdown headers
    if c['cell_type'] == 'markdown':
        print(f"  Content: {src[:200]}")
    print()
