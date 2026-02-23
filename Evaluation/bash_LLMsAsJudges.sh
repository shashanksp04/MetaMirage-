#!/usr/bin/env bash
set -eo pipefail

############################################
# Parallelism (vLLM-safe)
NUM_PROCESSES=8

############################################
# Model configuration
JUDGE_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
SUBJECT_NAME="Llama-3.2-11B-Vision-Instruct"

JUDGE_CLEAN="Llama-3.2-11B-Vision-Instruct"
SUBJECT_CLEAN="Llama-3.2-11B-Vision-Instruct"


TEMPERATURE=0.6

MAX_RECORDS=409

# ✅ vLLM OpenAI-compatible endpoint
OPENAI_API_BASE="http://127.0.0.1:11434/v1"

############################################
# Paths
INPUT_DIR="../Datasets/sample_inference"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/benchmark_run_${JUDGE_CLEAN}.log"
> "$LOG_FILE"

############################################
TOTAL_TASKS=2
COMPLETED_TASKS=0
START_TIME=$(date +%s)

############################################
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_progress() {
  COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
  log "Progress: $((COMPLETED_TASKS * 100 / TOTAL_TASKS))% ($COMPLETED_TASKS/$TOTAL_TASKS)"
}

############################################
run_benchmark() {
  local bench="$1"
  local mode="$2"
  local input_file="$3"
  local outdir="$4"

  log "------------------------------------------"
  log "Benchmark: $bench ($mode)"
  log "API Model: $JUDGE_NAME"
  log "Input: $input_file"

  if [ ! -f "$input_file" ]; then
    log "❌ ERROR: Input file not found"
    update_progress
    return
  fi

  mkdir -p "$outdir"

  python "LLMsAsJudges_${mode}.py" \
    --input_file "$input_file" \
    --output_file "$outdir/score_${SUBJECT_CLEAN}.json" \
    --model_name "$JUDGE_NAME" \
    --subject_name "$SUBJECT_NAME" \
    --num_processes "$NUM_PROCESSES" \
    --temperature "$TEMPERATURE" \
    --openai_api_base "$OPENAI_API_BASE" \
    --max_records "$MAX_RECORDS" \
    >> "$LOG_FILE" 2>&1

  log "✅ Completed: $bench ($mode)"
  update_progress
}

############################################
log "=========================================="
log "LLMs-as-Judges | STANDARD"
log "Model (Judge=Subject): $JUDGE_NAME"
log "API Base: $OPENAI_API_BASE"
log "Processes: $NUM_PROCESSES"
log "=========================================="

# ID scoring
run_benchmark \
  "standard" \
  "ID" \
  "$INPUT_DIR/sample_standard_benchmark_ID_${SUBJECT_CLEAN}.json" \
  "./results/standard_ID_score/${JUDGE_CLEAN}"

# MG scoring
run_benchmark \
  "standard" \
  "MG" \
  "$INPUT_DIR/sample_standard_benchmark_MG_${SUBJECT_CLEAN}.json" \
  "./results/standard_MG_score/${JUDGE_CLEAN}"

############################################
ELAPSED=$(( $(date +%s) - START_TIME ))
log "Finished in ${ELAPSED}s"
