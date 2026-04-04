#!/usr/bin/env bash
set -euo pipefail

HW4_ROOT="/home/astar/workspace/course/NLP/HW4"
ROOT="$HW4_ROOT/code/part2"
SUBMISSION_DIR="$HW4_ROOT/_gradescope/submission"
SUBMISSION_ZIP="$HW4_ROOT/_gradescope/submission.zip"
SUBMISSION_FLAT_ZIP="$HW4_ROOT/_gradescope/submission_flat.zip"
CONDA_SH="/home/astar/miniconda3/etc/profile.d/conda.sh"
RUN_LOG="$ROOT/autotune_supervisor.log"
STATUS_TXT="$ROOT/autotune_status.txt"
START_TS=$(date +%s)
DEADLINE_TS=$((START_TS + 3900))

DEV_F1_THRESHOLD="0.57"
DEV_ERROR_THRESHOLD="0.06"
SCRATCH_FLOOR_F1="0.5625231597926456"
BASELINE_EXP="scratch_autotune_stage1"

CANONICAL_EC_SQL_NAME="t5_ft_experiment_ec_test.sql"
CANONICAL_EC_PKL_NAME="t5_ft_experiment_ec_test.pkl"
CANONICAL_EC_SQL="$ROOT/results/$CANONICAL_EC_SQL_NAME"
CANONICAL_EC_PKL="$ROOT/records/$CANONICAL_EC_PKL_NAME"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

remaining_secs() {
  echo $(( DEADLINE_TS - $(date +%s) ))
}

candidate_sql_path() {
  local exp_name="$1"
  echo "$ROOT/results/${exp_name}_candidate_test.sql"
}

candidate_pkl_path() {
  local exp_name="$1"
  echo "$ROOT/records/${exp_name}_candidate_test.pkl"
}

final_dev_sql_path() {
  local exp_name="$1"
  echo "$ROOT/results/${exp_name}_final_dev.sql"
}

final_dev_pkl_path() {
  local exp_name="$1"
  echo "$ROOT/records/${exp_name}_final_dev.pkl"
}

final_dev_metrics_path() {
  local exp_name="$1"
  echo "$ROOT/results/${exp_name}_final_dev.metrics.json"
}

read_metric() {
  local metrics_json="$1"
  local metric_name="$2"
  python - "$metrics_json" "$metric_name" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
metric_name = sys.argv[2]
metrics = json.loads(path.read_text())
print(metrics[metric_name])
PY
}

acceptance_passed() {
  local metrics_json="$1"
  python - "$metrics_json" "$DEV_F1_THRESHOLD" "$DEV_ERROR_THRESHOLD" "$SCRATCH_FLOOR_F1" <<'PY'
import json, pathlib, sys
metrics = json.loads(pathlib.Path(sys.argv[1]).read_text())
dev_f1_threshold = float(sys.argv[2])
dev_error_threshold = float(sys.argv[3])
scratch_floor_f1 = float(sys.argv[4])
passed = (
    metrics["record_f1"] >= dev_f1_threshold
    and metrics["error_rate"] <= dev_error_threshold
    and metrics["record_f1"] >= scratch_floor_f1
)
print("yes" if passed else "no")
PY
}

is_strictly_better() {
  local candidate="$1"
  local incumbent="$2"
  python - "$candidate" "$incumbent" <<'PY'
import sys
print("yes" if float(sys.argv[1]) > float(sys.argv[2]) else "no")
PY
}

cleanup_candidate_artifacts() {
  rm -f "$ROOT"/results/*_candidate_test.sql
  rm -f "$ROOT"/records/*_candidate_test.pkl
}

cleanup_submission_dir() {
  local keep_names=(
    "main.pdf"
    "out_augmented_original.txt"
    "out_augmented_transformed.txt"
    "out_original.txt"
    "out_transformed.txt"
    "t5_ft_experiment_test.sql"
    "t5_ft_experiment_test.pkl"
    "$CANONICAL_EC_SQL_NAME"
    "$CANONICAL_EC_PKL_NAME"
  )
  local keep_expr=()
  local keep_name
  for keep_name in "${keep_names[@]}"; do
    keep_expr+=(-name "$keep_name" -o)
  done
  unset 'keep_expr[${#keep_expr[@]}-1]'
  find "$SUBMISSION_DIR" -maxdepth 1 -type f ! \( "${keep_expr[@]}" \) -delete
}

rebuild_submission_archives() {
  cleanup_submission_dir
  rm -f "$SUBMISSION_ZIP" "$SUBMISSION_FLAT_ZIP"
  (
    cd "$SUBMISSION_DIR"
    zip -q "$SUBMISSION_ZIP" ./*
    zip -jq "$SUBMISSION_FLAT_ZIP" ./*
  )
}

sync_submission_bundle() {
  cp "$CANONICAL_EC_SQL" "$SUBMISSION_DIR/$CANONICAL_EC_SQL_NAME"
  cp "$CANONICAL_EC_PKL" "$SUBMISSION_DIR/$CANONICAL_EC_PKL_NAME"
  rebuild_submission_archives
}

run_stage() {
  local stage_name="$1"
  shift
  local logfile="$ROOT/${stage_name}.log"
  log "starting $stage_name"
  echo "running: $stage_name" > "$STATUS_TXT"
  (
    source "$CONDA_SH"
    conda activate nlp_hw4_p2
    cd "$ROOT"
    python train_t5.py "$@" | tee "$logfile"
  )
}

export_submission_candidate() {
  local source_exp="$1"
  local final_dev_sql
  local final_dev_pkl
  local candidate_test_sql
  local candidate_test_pkl
  local metrics_json
  local export_log

  final_dev_sql="$(final_dev_sql_path "$source_exp")"
  final_dev_pkl="$(final_dev_pkl_path "$source_exp")"
  candidate_test_sql="$(candidate_sql_path "$source_exp")"
  candidate_test_pkl="$(candidate_pkl_path "$source_exp")"
  metrics_json="$(final_dev_metrics_path "$source_exp")"
  export_log="$ROOT/export_${source_exp}.log"

  log "exporting scratch candidate from $source_exp"
  (
    source "$CONDA_SH"
    conda activate nlp_hw4_p2
    cd "$ROOT"
    SOURCE_EXP="$source_exp" \
    FINAL_DEV_SQL="$final_dev_sql" \
    FINAL_DEV_PKL="$final_dev_pkl" \
    CANDIDATE_TEST_SQL="$candidate_test_sql" \
    CANDIDATE_TEST_PKL="$candidate_test_pkl" \
    METRICS_JSON="$metrics_json" \
    python - <<'PY'
import json
import os
import sys
from argparse import Namespace
from load_data import load_t5_data
from t5_utils import load_model_from_checkpoint
from train_t5 import test_inference, eval_epoch, DEVICE

ROOT = '/home/astar/workspace/course/NLP/HW4/code/part2'
source_exp = os.environ['SOURCE_EXP']
final_dev_sql = os.environ['FINAL_DEV_SQL']
final_dev_pkl = os.environ['FINAL_DEV_PKL']
candidate_test_sql = os.environ['CANDIDATE_TEST_SQL']
candidate_test_pkl = os.environ['CANDIDATE_TEST_PKL']
metrics_json = os.environ['METRICS_JSON']
args = Namespace(
    finetune=False,
    batch_size=8,
    test_batch_size=8,
    num_beams=1,
    max_generation_tokens=256,
    length_penalty=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    experiment_name=source_exp,
    resume_experiment_name=None,
    augment_sql_vocab=True,
)
train_loader, dev_loader, test_loader = load_t5_data(8, 8, augment_sql_vocab=True)
model = load_model_from_checkpoint(args, best=True).to(DEVICE)
model.tokenizer = dev_loader.dataset.tokenizer

dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
    args,
    model,
    dev_loader,
    os.path.join(ROOT, 'data', 'dev.sql'),
    final_dev_sql,
    os.path.join(ROOT, 'records', 'ground_truth_dev.pkl'),
    final_dev_pkl,
)
metrics = {
    'experiment': source_exp,
    'dev_loss': dev_loss,
    'record_f1': dev_record_f1,
    'record_em': dev_record_em,
    'sql_em': dev_sql_em,
    'error_rate': dev_error_rate,
}
with open(metrics_json, 'w') as f:
    json.dump(metrics, f, indent=2, sort_keys=True)
print('FINAL_DEV_METRICS', json.dumps(metrics, sort_keys=True))

model.tokenizer = test_loader.dataset.tokenizer
test_inference(
    args,
    model,
    test_loader,
    candidate_test_sql,
    candidate_test_pkl,
)
print('EXPORTED', source_exp)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
PY
  ) | tee "$export_log"

  (
    source "$CONDA_SH"
    conda activate nlp_hw4_p2
    cd "$ROOT"
    python evaluate.py \
      --predicted_sql "results/${source_exp}_final_dev.sql" \
      --predicted_records "records/${source_exp}_final_dev.pkl" \
      --development_sql data/dev.sql \
      --development_records records/ground_truth_dev.pkl
    python validate_submission.py \
      --sql "results/${source_exp}_candidate_test.sql" \
      --records "records/${source_exp}_candidate_test.pkl"
  ) | tee -a "$export_log"
}

promote_submission_candidate() {
  local source_exp="$1"
  cp "$(candidate_sql_path "$source_exp")" "$CANONICAL_EC_SQL"
  cp "$(candidate_pkl_path "$source_exp")" "$CANONICAL_EC_PKL"
  sync_submission_bundle
  log "promoted ${source_exp} candidate artifacts to canonical extra-credit submission files"
}

main() {
  rm -f "$RUN_LOG" "$STATUS_TXT"
  cleanup_candidate_artifacts
  log "autotune supervisor started"
  log "time budget seconds=$(remaining_secs)"

  export_submission_candidate "$BASELINE_EXP"

  local best_exp="$BASELINE_EXP"
  local best_metrics_json
  local best_f1
  local baseline_f1
  local baseline_accepted
  local cont_f1
  local improvement_ok

  best_metrics_json="$(final_dev_metrics_path "$BASELINE_EXP")"
  best_f1="$(read_metric "$best_metrics_json" record_f1)"
  baseline_f1="$best_f1"
  baseline_accepted="$(acceptance_passed "$best_metrics_json")"
  log "baseline ${BASELINE_EXP}: record_f1=${best_f1}, acceptance=${baseline_accepted}"

  if [[ "$baseline_accepted" == "yes" ]]; then
    promote_submission_candidate "$BASELINE_EXP"
    cleanup_candidate_artifacts
    echo "done: $BASELINE_EXP" > "$STATUS_TXT"
    log "autotune supervisor finished with best=$BASELINE_EXP"
    return 0
  fi

  if (( $(remaining_secs) > 1200 )); then
    run_stage scratch_stage1_cont_lr3e5 \
      --augment_sql_vocab \
      --optimizer_type AdamW \
      --learning_rate 3e-5 \
      --adam_beta2 0.98 \
      --scheduler_type linear \
      --num_warmup_steps 0 \
      --max_n_epochs 4 \
      --patience_epochs 2 \
      --batch_size 8 \
      --test_batch_size 8 \
      --num_beams 1 \
      --gradient_accumulation_steps 8 \
      --max_grad_norm 1.0 \
      --full_eval_every_epochs 1 \
      --selection_metric record_f1 \
      --experiment_name scratch_stage1_cont_lr3e5 \
      --resume_from_best \
      --resume_experiment_name "$BASELINE_EXP" \
      --skip_test_export

    export_submission_candidate "scratch_stage1_cont_lr3e5"
    cont_f1="$(read_metric "$(final_dev_metrics_path scratch_stage1_cont_lr3e5)" record_f1)"
    log "continuation scratch_stage1_cont_lr3e5: record_f1=${cont_f1}"
    if [[ "$(is_strictly_better "$cont_f1" "$best_f1")" == "yes" ]]; then
      best_exp="scratch_stage1_cont_lr3e5"
      best_metrics_json="$(final_dev_metrics_path "$best_exp")"
      best_f1="$cont_f1"
    fi
  fi

  if (( $(remaining_secs) > 1200 )); then
    run_stage scratch_stage1_cont_lr5e5 \
      --augment_sql_vocab \
      --optimizer_type AdamW \
      --learning_rate 5e-5 \
      --adam_beta2 0.98 \
      --scheduler_type linear \
      --num_warmup_steps 0 \
      --max_n_epochs 4 \
      --patience_epochs 2 \
      --batch_size 8 \
      --test_batch_size 8 \
      --num_beams 1 \
      --gradient_accumulation_steps 8 \
      --max_grad_norm 1.0 \
      --full_eval_every_epochs 1 \
      --selection_metric record_f1 \
      --experiment_name scratch_stage1_cont_lr5e5 \
      --resume_from_best \
      --resume_experiment_name "$BASELINE_EXP" \
      --skip_test_export

    export_submission_candidate "scratch_stage1_cont_lr5e5"
    cont_f1="$(read_metric "$(final_dev_metrics_path scratch_stage1_cont_lr5e5)" record_f1)"
    log "continuation scratch_stage1_cont_lr5e5: record_f1=${cont_f1}"
    if [[ "$(is_strictly_better "$cont_f1" "$best_f1")" == "yes" ]]; then
      best_exp="scratch_stage1_cont_lr5e5"
      best_metrics_json="$(final_dev_metrics_path "$best_exp")"
      best_f1="$cont_f1"
    fi
  fi

  improvement_ok="$(python - "$best_f1" "$baseline_f1" <<'PY'
import sys
best_f1 = float(sys.argv[1])
baseline_f1 = float(sys.argv[2])
print("yes" if best_f1 > baseline_f1 + 0.005 else "no")
PY
  )"
  if [[ "$improvement_ok" != "yes" ]]; then
    best_exp="$BASELINE_EXP"
    best_metrics_json="$(final_dev_metrics_path "$BASELINE_EXP")"
    best_f1="$baseline_f1"
    log "continuation gains did not beat baseline by 0.005; falling back to ${BASELINE_EXP}"
  fi

  promote_submission_candidate "$best_exp"
  cleanup_candidate_artifacts
  echo "done: $best_exp" > "$STATUS_TXT"
  log "autotune supervisor finished with best=$best_exp record_f1=$best_f1 metrics=$best_metrics_json"
}

main "$@"
