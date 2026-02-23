#!/usr/bin/env bash

# Confidence benchmark
BENCH_TYPE="standard"

# ID or MG
MODE="ID"

# Filesystem-safe model names
SUBJECT_NAME="Llama-3.2-11B-Vision-Instruct"
JUDGE_NAME="Llama-3.2-11B-Vision-Instruct"

python print_scores.py \
  --bench_type "$BENCH_TYPE" \
  --mode "$MODE" \
  --subject_name "$SUBJECT_NAME" \
  --judge_name "$JUDGE_NAME"
