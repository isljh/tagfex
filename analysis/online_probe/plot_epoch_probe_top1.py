import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


STEP_COLUMNS = ["Step", "step", "global_step", "Global Step", "global step"]
VALUE_COLUMNS = [
    "init/probe_top1",
    "probe_top1",
    "task0_extra/probe_top1",
    "Value",
    "value",
    "Y",
    "y",
]


def pick_column(fieldnames, requested, candidates, label):
    if requested:
        if requested not in fieldnames:
            raise ValueError("{} column '{}' not found. Available columns: {}".format(label, requested, fieldnames))
        return requested
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError("Cannot infer {} column. Available columns: {}".format(label, fieldnames))


def read_csv_rows(path, step_column=None, value_column=None):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: {}".format(path))
        step_col = pick_column(reader.fieldnames, step_column, STEP_COLUMNS, "step")
        value_col = pick_column(reader.fieldnames, value_column, VALUE_COLUMNS, "value")
        rows = []
        for row in reader:
            raw_step = row.get(step_col, "")
            raw_value = row.get(value_col, "")
            if raw_step == "" or raw_value == "":
                continue
            try:
                step = int(float(raw_step))
                value = float(raw_value)
            except ValueError:
                continue
            if math.isfinite(value):
                rows.append((step, value))
    rows.sort(key=lambda item: item[0])
    return rows, step_col, value_col


def read_swanlab_backup_rows(path, value_column=None):
    metric_key = value_column or "init/probe_top1"
    rows = []
    with open(path, "rb") as f:
        for raw_line in f:
            start = raw_line.find(b'{"model_type"')
            if start < 0:
                continue
            try:
                obj = json.loads(raw_line[start:].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if obj.get("model_type") != "Scalar":
                continue
            data = obj.get("data", {})
            if data.get("key") != metric_key:
                continue
            metric = data.get("metric", {})
            try:
                step = int(float(data["step"]))
                value = float(metric["data"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows.append((step, value))
    rows.sort(key=lambda item: item[0])
    return rows, "step", metric_key


def read_rows(path, step_column=None, value_column=None):
    suffix = Path(path).suffix.lower()
    if suffix == ".swanlab" or Path(path).name == "backup.swanlab":
        return read_swanlab_backup_rows(path, value_column)
    return read_csv_rows(path, step_column, value_column)


def aggregate_by_epoch(rows, batches_per_epoch):
    buckets = {}
    for step, value in rows:
        epoch = step // batches_per_epoch
        buckets.setdefault(epoch, []).append(value)
    epoch_rows = []
    for epoch in sorted(buckets):
        values = buckets[epoch]
        epoch_rows.append({
            "epoch": epoch + 1,
            "num_batches": len(values),
            "probe_top1_mean": sum(values) / len(values),
            "probe_top1_first": values[0],
            "probe_top1_last": values[-1],
        })
    return epoch_rows


def write_epoch_csv(path, epoch_rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["epoch", "num_batches", "probe_top1_mean", "probe_top1_first", "probe_top1_last"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)


def plot_epoch_curve(path, epoch_rows, title):
    epochs = [row["epoch"] for row in epoch_rows]
    values = [row["probe_top1_mean"] for row in epoch_rows]
    plt.figure(figsize=(8, 4.8))
    plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5)
    plt.xlabel("Epoch")
    plt.ylabel("Mean batch probe top1 (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate step-level online probe top1 into epoch-level mean-batch accuracy."
    )
    parser.add_argument("--input", required=True, help="CSV exported from SwanLab, local metrics CSV, or swanlog backup.swanlab.")
    parser.add_argument("--output-dir", default="analysis/online_probe/results", help="Directory for PNG and aggregated CSV.")
    parser.add_argument("--batches-per-epoch", type=int, default=202, help="Number of train batches per epoch.")
    parser.add_argument("--step-column", default=None, help="Optional explicit step column name.")
    parser.add_argument("--value-column", default=None, help="Optional explicit probe top1 column name. For backup.swanlab this is the metric key, e.g. init/probe_top1.")
    parser.add_argument("--title", default="Epoch mean online probe top1", help="Plot title.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, step_col, value_col = read_rows(input_path, args.step_column, args.value_column)
    if not rows:
        raise ValueError("No valid step/value rows found in {}.".format(input_path))

    epoch_rows = aggregate_by_epoch(rows, args.batches_per_epoch)
    stem = input_path.stem
    csv_path = output_dir / "{}_epoch_mean_probe_top1.csv".format(stem)
    png_path = output_dir / "{}_epoch_mean_probe_top1.png".format(stem)
    write_epoch_csv(csv_path, epoch_rows)
    plot_epoch_curve(png_path, epoch_rows, args.title)

    print("Read {} step rows from {}".format(len(rows), input_path))
    print("Using step column '{}' and value column '{}'".format(step_col, value_col))
    print("Aggregated into {} epochs with batches_per_epoch={}".format(len(epoch_rows), args.batches_per_epoch))
    print("Saved epoch CSV: {}".format(csv_path))
    print("Saved plot: {}".format(png_path))


if __name__ == "__main__":
    main()
