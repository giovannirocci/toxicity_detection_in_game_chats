import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def is_scalar_metrics(results):
    return isinstance(results["precision"], float)

def plot_overall_metrics(models, output_path, title):
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [model[m] for m in metrics]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=model["model"])

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved overall metrics plot to: {output_path}")

def plot_class_metrics(models, class_labels, output_path, title):
    x = np.arange(len(class_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, model["f1"], width, label=model["model"])

    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved per-class F1 plot to: {output_path}")

# === CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--model1", "-m1", required=True)
parser.add_argument("--model2", "-m2", required=True)
parser.add_argument("--model3", "-m3", help="Optional third model")
parser.add_argument("--output_dir", "-o", default="plots")
parser.add_argument("--class_labels", "-c", nargs="+", default=["Explicitly Toxic", "Implicitly Toxic", "Action", "Other"])
parser.add_argument("--title1", "-t1", default="Toxicity Detection Performance")
parser.add_argument("--title2", "-t2", default="Per-Class F1 Score Comparison")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# === Load models ===
models = []
with open(args.model1) as f:
    models.append(json.load(f))
with open(args.model2) as f:
    models.append(json.load(f))
if args.model3:
    with open(args.model3) as f:
        models.append(json.load(f))

# === Plot ===
if all(is_scalar_metrics(m) for m in models):
    plot_overall_metrics(
        models,
        os.path.join(args.output_dir, "metric_comparison.png"),
        args.title1
    )
elif all(not is_scalar_metrics(m) for m in models):
    plot_class_metrics(
        models,
        args.class_labels,
        os.path.join(args.output_dir, "per_class_f1_comparison.png"),
        args.title2
    )
else:
    raise ValueError("Mismatch: some models use scalar metrics, others use per-class metrics.")
