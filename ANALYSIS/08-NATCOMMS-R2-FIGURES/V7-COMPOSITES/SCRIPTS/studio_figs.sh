#!/bin/bash
# Regenerate R2 figure panels on the Studio into /tmp/heimfig. Copies of the generators are patched
# only for output paths, the Fig 4 b-e panel letters and the ED Fig 1c annotation; the iCloud tree is not written.
set -euo pipefail
R="$HOME/Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS"
PY=/opt/homebrew/bin/python3.11
O=/tmp/heimfig; rm -rf "$O"; mkdir -p "$O/ARCHIVE" "$O/src"
BASE="Path(\"$R\")"

# Fig 3 composite (the script that produced the submitted image3.png)
sed -e "s|base = Path(__file__).resolve().parent.parent|base = $BASE|" \
    -e "s|out = base / \"docs\" / \"figures\" / \"Fig_03_Translation_Dimension_LEGIBLE.png\"|out = Path(\"$O/Fig3.png\")|" \
    "$R/PYTHON/fig3_combined_legible.py" > "$O/src/fig3.py"
grep -q "$O/Fig3.png" "$O/src/fig3.py"
(cd "$O" && $PY src/fig3.py)

# Fig 4a UMAP (same seed and parameters as submitted)
sed -e "s|base = Path(__file__).resolve().parent.parent|base = $BASE|" \
    -e "s|out_png = base / \"docs\" / \"figures\" / \"Fig_04_UMAP_Semantic_Landscape_LEGIBLE.png\"|out_png = Path(\"$O/Fig4a.png\")|" \
    "$R/PYTHON/fig4a_legible.py" > "$O/src/fig4a.py"
grep -q "$O/Fig4a.png" "$O/src/fig4a.py"
(cd "$O" && $PY src/fig4a.py)

# Fig 4 b-e (05-04 combined panels, relettered b-e)
sed -e "s|^SCRIPT_DIR = Path(__file__).parent|SCRIPT_DIR = Path(\"$R/PYTHON\")|" \
    -e "s|\['a', 'b', 'c', 'd'\]|['b', 'c', 'd', 'e']|" \
    -e "s|hspace=0.38, wspace=0.35|hspace=0.50, wspace=0.75|" \
    -e "s|ax.text(-0.08, 1.08, label|ax.text(-0.20, 1.10, label|" \
    -e "s|Path(__file__).resolve().parent.parent / \"docs\"|Path(\"$R\") / \"docs\"|" \
    "$R/PYTHON/05-04-heim-sem-generate-figures.py" > "$O/src/sem.py"
grep -q "\['b', 'c', 'd', 'e'\]" "$O/src/sem.py"
(cd "$O/src" && $PY - <<EOF
import logging, sys
from pathlib import Path
sys.path.insert(0, "$O/src")
import sem
logging.basicConfig(level=logging.INFO)
m = sem.load_metrics(); assert m and len(m['metrics']) == 175, len(m['metrics'])
ok = sem.generate_fig5_combined(m, Path("$O"), logging.getLogger("f"))
print("fig4be", ok)
EOF
)
ls -l "$O"/*.png "$O"/*.pdf "$O"/ARCHIVE/*.png
