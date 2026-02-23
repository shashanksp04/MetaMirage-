#!/usr/bin/env python3
import json
import pandas as pd
from tabulate import tabulate
import os
import argparse
import sys

"""
Compute and display metrics for a single subject model and a single judge model
on a specified benchmark type and mode (ID or MG).
For ID mode: identification_accuracy and reasoning_accuracy.
For MG mode: accuracy, relevance, completeness, parsimony, and a weighted sum.
"""

METRICS_MG = ["accuracy", "relevance", "completeness", "parsimony"]
WEIGHTS_MG = {"accuracy": 2, "relevance": 1, "completeness": 1, "parsimony": 1}
MAX_METRIC_VALUE = 4  # Maximum score for each metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print benchmark scores for one subject and one judge model"
    )
    parser.add_argument(
        "--bench_type",
        type=str,
        default="standard",
        help="Benchmark type: 'standard' or 'contextual'",
    )
    parser.add_argument(
        "--judge_name",
        type=str,
        required=True,
        help="Name of the judge model, e.g. 'Qwen3-32B'",
    )
    parser.add_argument(
        "--subject_name",
        type=str,
        required=True,
        help="Name of the subject model, e.g. 'gpt-4o-mini'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ID", "MG"],
        default="ID",
        help="Benchmark mode: 'ID' for identification or 'MG' for management",
    )
    return parser.parse_args()


def print_id_scores(subject_name, judge_name, bench_type):
    # Path for ID mode scores
    score_path = (
        f"./results/{bench_type}_ID_score/{judge_name}/"
        f"score_{subject_name}.json"
    )
    if not os.path.exists(score_path):
        sys.exit(f"ERROR: Score file not found: {score_path}")

    id_acc_sum = 0.0
    reasoning_sum = 0.0
    n_samples = 0

    with open(score_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            score = rec.get("score", {})
            id_acc_sum    += int(score.get("identification_accuracy", 0))
            reasoning_sum += int(score.get("reasoning_accuracy", 0))
            n_samples     += 1

    if n_samples:
        id_acc_mean    = round(id_acc_sum / n_samples * 100, 2)
        reasoning_mean = round(reasoning_sum / n_samples, 2)
    else:
        id_acc_mean = reasoning_mean = 0.0

    df = pd.DataFrame([{
        "subject": subject_name,
        "identification_accuracy(%)": id_acc_mean,
        "reasoning_accuracy": reasoning_mean
    }])

    print(
        f"\nJudge: {judge_name} | Subject: {subject_name} | "
        f"Mode: ID | Benchmark: {bench_type}\n"
    )
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def print_mg_scores(subject_name, judge_name, bench_type):
    # Path for MG mode scores
    score_path = (
        f"./results/{bench_type}_MG_score/{judge_name}/"
        f"score_{subject_name}.json"
    )
    if not os.path.exists(score_path):
        sys.exit(f"ERROR: Score file not found: {score_path}")

    sums = {m: 0.0 for m in METRICS_MG}
    n_samples = 0

    with open(score_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            score = rec.get("score", {})
            for m in METRICS_MG:
                sums[m] += score.get(m, 0)
            n_samples += 1

    # Compute means and weighted sum
    row = {"subject": subject_name}
    for m in METRICS_MG:
        row[m] = round(sums[m] / n_samples, 2) if n_samples else None

    # Weighted sum: weights 2:1:1:1, normalized to [0,1]
    if n_samples:
        weighted_raw = sum(WEIGHTS_MG[m] * row[m] for m in METRICS_MG)
        max_raw = sum(WEIGHTS_MG.values()) * MAX_METRIC_VALUE
        row["Weighted_Sum"] = round(weighted_raw / max_raw, 4)
    else:
        row["Weighted_Sum"] = None

    df = pd.DataFrame([row])

    print(
        f"\nJudge: {judge_name} | Subject: {subject_name} | "
        f"Mode: MG | Benchmark: {bench_type}\n"
    )
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "ID":
        print_id_scores(
            args.subject_name,
            args.judge_name,
            args.bench_type
        )
    else:
        print_mg_scores(
            args.subject_name,
            args.judge_name,
            args.bench_type
        )
