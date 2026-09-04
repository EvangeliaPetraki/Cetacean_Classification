import argparse
import csv
import json
import os
from typing import Dict, List, Optional


def rounded_to(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def find_matching_tp(support: int, recall_str: str, digits: int = 4) -> int:
    candidates = []
    for tp in range(support + 1):
        recall = 0.0 if support == 0 else tp / support
        if rounded_to(recall, digits) == recall_str:
            candidates.append(tp)

    if not candidates:
        raise ValueError(
            f"Could not reconstruct TP from recall={recall_str} and support={support}."
        )

    recall_value = float(recall_str)
    return min(candidates, key=lambda tp: abs((tp / support if support else 0.0) - recall_value))


def find_matching_fp(tp: int, precision_str: str, digits: int = 4) -> int:
    if tp == 0:
        return 0

    precision_value = float(precision_str)
    max_fp = max(100000, tp * 1000)
    candidates = []

    for fp in range(max_fp + 1):
        precision = tp / (tp + fp)
        if rounded_to(precision, digits) == precision_str:
            candidates.append(fp)

    if not candidates:
        fp_estimate = max(0, round(tp * (1.0 / precision_value - 1.0)))
        window_start = max(0, fp_estimate - 10)
        window_end = fp_estimate + 10
        for fp in range(window_start, window_end + 1):
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if rounded_to(precision, digits) == precision_str:
                candidates.append(fp)

    if not candidates:
        raise ValueError(
            f"Could not reconstruct FP from precision={precision_str} and TP={tp}."
        )

    return min(candidates, key=lambda fp: abs((tp / (tp + fp)) - precision_value))


def parse_classification_report(report_path: str) -> Dict:
    rows = []
    accuracy = None
    macro_avg = None

    with open(report_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()

            # sklearn's "accuracy" summary line has no precision/recall columns:
            # "accuracy                         0.9877       733"
            if len(parts) == 3 and parts[0] == "accuracy":
                try:
                    accuracy = float(parts[1])
                except ValueError:
                    pass
                continue

            # "macro avg     0.9897    0.9890    0.9893       733"
            if len(parts) == 6 and parts[0] == "macro" and parts[1] == "avg":
                try:
                    macro_avg = {
                        "precision": float(parts[2]),
                        "recall": float(parts[3]),
                        "f1": float(parts[4]),
                    }
                except ValueError:
                    pass
                continue

            if len(parts) != 5:
                continue

            label, precision_str, recall_str, f1_str, support_str = parts

            try:
                support = int(support_str)
                float(precision_str)
                float(recall_str)
                float(f1_str)
            except ValueError:
                continue

            rows.append(
                {
                    "label": label,
                    "precision_str": precision_str,
                    "recall_str": recall_str,
                    "f1_str": f1_str,
                    "support": support,
                }
            )

    if not rows:
        raise ValueError(f"No per-class rows were found in {report_path}")

    total_samples = sum(row["support"] for row in rows)
    class_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    macro_auc_sum = 0.0

    for row in rows:
        tp = find_matching_tp(row["support"], row["recall_str"])
        fp = find_matching_fp(tp, row["precision_str"])
        fn = row["support"] - tp
        tn = total_samples - tp - fp - fn

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        # This is a hard-label one-vs-rest approximation.
        auc_ovr_approx = 0.5 * (recall + specificity)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        macro_auc_sum += auc_ovr_approx

        class_metrics.append(
            {
                "label": row["label"],
                "support": row["support"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(row["precision_str"]),
                "recall": float(row["recall_str"]),
                "f1": float(row["f1_str"]),
                "auc_ovr_approx": auc_ovr_approx,
            }
        )

    return {
        "report_path": report_path,
        "n_classes": len(class_metrics),
        "total_samples": total_samples,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        # Precision/recall/F1/AUC are all macro-averaged: every class counts equally
        # regardless of its support, so a poorly-served minority class isn't masked
        # by strong majority-class performance.
        "accuracy": accuracy,
        "precision": macro_avg["precision"] if macro_avg else None,
        "recall": macro_avg["recall"] if macro_avg else None,
        "f1": macro_avg["f1"] if macro_avg else None,
        "auc": macro_auc_sum / len(class_metrics),
        "class_metrics": class_metrics,
    }


def collect_report_summaries(runs_root: str, report_name: str) -> List[Dict]:
    summaries = []

    for entry in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, entry)
        if not os.path.isdir(run_dir):
            continue

        report_path = os.path.join(run_dir, report_name)
        if not os.path.exists(report_path):
            continue

        summary = parse_classification_report(report_path)
        summary["run_id"] = entry
        summary["source_root"] = runs_root

        is_ensemble = "__PLUS__" in entry
        summary["is_ensemble"] = is_ensemble
        summary["ensemble_size"] = entry.count("__PLUS__") + 1 if is_ensemble else 1

        # run_id keeps the loro_ prefix; feature_type/model are derived from the
        # bare name so they match the pre-LORO pipeline's values.
        bare = entry[len("loro_"):] if entry.startswith("loro_") else entry
        parts = bare.split("__")
        if len(parts) >= 2:
            summary["feature_type"] = parts[0]
            summary["model"] = parts[1]
        else:
            summary["feature_type"] = ""
            summary["model"] = ""

        summaries.append(summary)

    return summaries


def write_summary_csv(summaries: List[Dict], out_csv: str) -> None:
    fieldnames = [
        "run_id",
        "source_root",
        "is_ensemble",
        "ensemble_size",
        "feature_type",
        "model",
        "n_classes",
        "total_samples",
        "total_tp",
        "total_fp",
        "total_fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "report_path",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({name: summary.get(name, "") for name in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate classification_report_test.txt files under best_models and ensemble_models "
            "into macro-averaged accuracy/precision/recall/F1/AUC per run."
        )
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        nargs="+",
        default=[
            "./best_models",
            "./ensemble_models/final models",
            "./ensemble_models/ensemble members",
        ],
        help="One or more directories containing run folders (single-model and/or ensemble).",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="loro_classification_report_test.txt",
        help="Name of the classification report file inside each run directory.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional output JSON path. Defaults to <first runs_root>/loro_classification_report_summary.json",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to <first runs_root>/loro_classification_report_summary.csv",
    )
    args = parser.parse_args()

    summaries = []
    seen_run_ids = set()
    for runs_root in args.runs_root:
        if not os.path.isdir(runs_root):
            print(f"[Warn] Skipping missing runs_root: {runs_root}")
            continue
        for summary in collect_report_summaries(runs_root, args.report_name):
            if summary["run_id"] in seen_run_ids:
                print(
                    f"[Warn] Duplicate run_id '{summary['run_id']}' found under {runs_root}; "
                    "keeping the first occurrence."
                )
                continue
            seen_run_ids.add(summary["run_id"])
            summaries.append(summary)

    if not summaries:
        raise RuntimeError(
            f"No '{args.report_name}' files were found under any of: {args.runs_root}"
        )

    out_json = args.out_json or os.path.join(args.runs_root[0], "loro_classification_report_summary.json")
    out_csv = args.out_csv or os.path.join(args.runs_root[0], "loro_classification_report_summary.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    write_summary_csv(summaries, out_csv)

    print(f"[Done] Wrote JSON summary to {out_json}")
    print(f"[Done] Wrote CSV summary to {out_csv}")
    print()
    for summary in summaries:
        print(
            f"{summary['run_id']}: "
            f"accuracy={summary['accuracy']:.4f}, "
            f"precision={summary['precision']:.4f}, "
            f"recall={summary['recall']:.4f}, "
            f"f1={summary['f1']:.4f}, "
            f"auc={summary['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
