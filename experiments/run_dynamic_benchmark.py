#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


DATASETS = [
    ("glove", "angular"),
    ("coco", "angular"),
    ("fashion", "euclidean"),
]


def parse_metric(name, output):
    match = re.search(rf"{re.escape(name)}:\s+([0-9.eE+-]+)", output)
    if not match:
        raise ValueError(f"Could not parse '{name}' from benchmark output")
    return float(match.group(1))


def parse_text(name, output):
    match = re.search(rf"{re.escape(name)}:\s+([A-Za-z_+-]+)", output)
    if not match:
        raise ValueError(f"Could not parse '{name}' from benchmark output")
    return match.group(1)


def run_one(algorithm, binary, data_dir, dataset, metric, base_count, op_count, r, l, alpha):
    cmd = [str(binary), str(data_dir), dataset, metric, "--dynamic",
           "--base-count", str(base_count), "--op-count", str(op_count)]
    if algorithm == "diskann":
        cmd.extend(["--R", str(r), "--L", str(l), "--alpha", str(alpha)])
    elif algorithm == "hnsw":
        cmd.extend(["--M", str(r), "--ef-construction", str(l)])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out = completed.stdout
    return {
        "algorithm": algorithm,
        "dataset": dataset,
        "metric": metric,
        "base_count": base_count,
        "op_count": op_count,
        "degree_param": r,
        "candidate_param": l,
        "alpha": alpha,
        "build_time_s": parse_metric("Build time", out),
        "insert_time_s": parse_metric("Insert time", out),
        "insert_throughput_ops_s": parse_metric("Insert throughput", out),
        "delete_time_s": parse_metric("Delete time", out),
        "delete_throughput_ops_s": parse_metric("Delete throughput", out),
        "active_vectors_after": int(parse_metric("Active vectors after benchmark", out)),
        "deleted_edge_check": parse_text("Deleted edge check", out),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DiskANN and HNSW dynamic insert/delete throughput experiments.")
    parser.add_argument("--diskann-binary", default="src/diskann_test")
    parser.add_argument("--hnsw-binary", default="HNSW/hnsw_test")
    parser.add_argument("--data-dir", default="/home/xulin/data/ann")
    parser.add_argument("--base-count", type=int, default=1000)
    parser.add_argument("--op-counts", default="100, 400, 1600, 6400")
    parser.add_argument("--R", type=int, default=16)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--output", default="experiments/dynamic_results.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    diskann_binary = Path(args.diskann_binary)
    hnsw_binary = Path(args.hnsw_binary)
    if not diskann_binary.is_absolute():
        diskann_binary = root / diskann_binary
    if not hnsw_binary.is_absolute():
        hnsw_binary = root / hnsw_binary
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    op_counts = [int(item.strip()) for item in args.op_counts.split(",") if item.strip()]

    rows = []
    for dataset, metric in DATASETS:
        for op_count in op_counts:
            rows.append(run_one("diskann", diskann_binary, args.data_dir, dataset, metric,
                                args.base_count, op_count, args.R, args.L, args.alpha))
            rows.append(run_one("hnsw", hnsw_binary, args.data_dir, dataset, metric,
                                args.base_count, op_count, args.R, args.L, args.alpha))

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['algorithm']}/{row['dataset']}/ops={row['op_count']}: "
            f"insert={row['insert_throughput_ops_s']:.2f} ops/s, "
            f"delete={row['delete_throughput_ops_s']:.2f} ops/s"
        )


if __name__ == "__main__":
    main()
