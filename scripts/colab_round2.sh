#!/usr/bin/env bash
# Round 2 on Colab: keep round-1 results, train two more KOA-ORDNet variants (focal gamma 1 and 0),
# select the KOA-ORDNet run by VALIDATION macro-F1 only, then evaluate on the test set and on the
# external dataset (RG3). Usage (in a Colab cell):  !cd /content/KOA-ORDNet && git pull -q && bash scripts/colab_round2.sh
set -euo pipefail
cd /content/KOA-ORDNet
SAVE=/content/drive/MyDrive/KOA-ORDNet
RUNS=$SAVE/runs
DATA=$(python - <<'EOF'
import kagglehub
from pathlib import Path
root = Path(kagglehub.dataset_download('shashwatwork/knee-osteoarthritis-dataset-with-severity'))
print(next(str(p.parent) for p in sorted(root.rglob('train'))
           if all((p / str(g)).is_dir() for g in range(5)) and (p.parent / 'test').is_dir()))
EOF
)
EXT=$(python - <<'EOF'
import kagglehub
from pathlib import Path
root = Path(kagglehub.dataset_download('tommyngx/digital-knee-xray'))
print(next(str(p) for p in root.rglob('MedicalExpert-I') if p.is_dir()))
EOF
)
DATA=$(printf "%s
" "$DATA" | tail -n 1); EXT=$(printf "%s
" "$EXT" | tail -n 1)
echo "DATA=$DATA"; echo "EXT=$EXT"

[ -d "$SAVE/results_round1" ] || { [ -d results ] && cp -r results "$SAVE/results_round1" || true; }

for g in 1 0; do
  if [ -f "$RUNS/koa_g$g/history.json" ]; then echo "== koa_g$g already trained"; continue; fi
  echo "== training koa_g$g (focal gamma $g)"
  python train.py --data "$DATA" --out "$RUNS" --loss koa --gamma "$g" --name "koa_g$g" --workers 2
done

python evaluate.py --data "$DATA" --runs "$RUNS" --external "$EXT" --workers 2
cp -r results "$SAVE"/
SEL=$(python -c "import json;print(json.load(open('results/results.json'))['koa']['run'])")
cp "$RUNS/$SEL/best.pt" "$SAVE/koa_ordnet.pt"
echo "selected KOA-ORDNet run: $SEL"
cat results/RESULTS.md
echo "RESULTS_JSON_BEGIN"; cat results/results.json; echo "RESULTS_JSON_END"
