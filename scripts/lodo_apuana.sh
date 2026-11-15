#!/usr/bin/env bash
# LODO do seletor compartilhado no Apuana: array de shards + merge dependente.
# Uso (na raiz do projeto no cluster): bash scripts/lodo_apuana.sh
set -euo pipefail

SHARDS=${SHARDS:-41}
PY=${PY:-$HOME/env_dp_meta_v3/bin/python}
CONDITIONS=${CONDITIONS:-results/portfolio-984-current-slim/conditions.csv}
OUT=${OUT:-results/portfolio-984-shared-selector-lodo}
ARGS="--conditions $CONDITIONS --output $OUT --lodo --num-shards $SHARDS"
ENV="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.; cd $PWD"
mkdir -p "$OUT/logs"

array=$(sbatch --parsable --partition=short-simple --time=12:00:00 \
  --cpus-per-task=8 --mem=8G --array=0-$((SHARDS - 1)) --job-name=lodo-shard \
  --output="$OUT/logs/shard-%a.out" --error="$OUT/logs/shard-%a.err" \
  --wrap="$ENV; $PY -u scripts/analyze_shared_selector.py $ARGS --shard \$SLURM_ARRAY_TASK_ID")
merge=$(sbatch --parsable --partition=short-simple --time=02:00:00 \
  --cpus-per-task=2 --mem=8G --dependency=afterok:$array --job-name=lodo-merge \
  --output="$OUT/logs/merge.out" --error="$OUT/logs/merge.err" \
  --wrap="$ENV; $PY -u scripts/analyze_shared_selector.py $ARGS --merge-shards")
echo "array=$array merge=$merge"
