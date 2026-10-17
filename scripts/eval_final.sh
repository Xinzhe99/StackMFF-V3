#!/bin/bash
# After training: evaluate every epoch on the four benchmarks (grey, skimage SSIM/PSNR, hard and weighted fusion),
# select the epoch with the best mean SSIM of the weighted fusion (sigma 0.5), and write the colour results of that
# epoch for both fusion rules (same epoch, as in the paper):
#   $OUT/${TAG}_soft/<dataset>/<scene>.png (weighted fusion) and $OUT/${TAG}_hard/<dataset>/<scene>.png (hard selection)
# The paper's model is epoch_2.pth (third epoch) of its run. Then compute all metrics with evaluation/run_eval.py.
set -e
RUN=${RUN:-runs/final}
BENCH=${BENCH:-/path/to/benchmarks}
OUT=${OUT:-results}
TAG=${TAG:-final}
cd "$(dirname "$0")/.."
export SIGMA=0.5
python tools/eval_epochs.py --run_dir $RUN --tag ${TAG}_all --out $OUT --bench_root "$BENCH" --skip_select   # all epochs
EP=$(python -c "
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['fusion'] == 'soft']
print(max(rows, key=lambda r: float(r['mean_SSIM']))['epoch'])" $OUT/${TAG}_all_epochs.csv)
echo "selected epoch $EP"
python tools/eval_epochs.py --run_dir $RUN --tag $TAG --out $OUT --bench_root "$BENCH" --epoch $EP
