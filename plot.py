import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def is_scalar_metrics(results):
    return isinstance(results["precision"], float)

def plot_overall_metrics(model1, model2, output_path, title):
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35

    values1 = [model1[m] for m in metrics]
    values2 = [model2[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, values1, width, label=model1["model"])
    ax.bar(x + width/2, values2, width, label=model2["model"])
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

def plot_class_metrics(model1, model2, class_labels, output_path, title):
    x = np.arange(len(class_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, model1["f1"], width, label=model1["model"])
    ax.bar(x + width/2, model2["f1"], width, label=model2["model"])
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


parser = argparse.ArgumentParser()
parser.add_argument("--model1_results", "-m1", required=True, help="Path to model1_results result JSON")
parser.add_argument("--model2_results", "-m2", required=True, help="Path to model2_results result JSON")
parser.add_argument("--output_dir", "-o", default="plots", help="Directory to save plots")
parser.add_argument("--class_labels", "-c", nargs="+", default=["Explicitly Toxic", "Implicitly Toxic", "Action", "Other"])
parser.add_argument("--title1", "-t1", default="Toxicity Detection Performance")
parser.add_argument("--title2", "-t2", default="Per-Class F1 Score Comparison")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


with open(args.model1_results, encoding="utf-8") as f, open(args.model2_results, encoding="utf-8") as g:
    model1_results = json.load(f)
    model2_results = json.load(g)


if is_scalar_metrics(model1_results) and is_scalar_metrics(model2_results):
    plot_overall_metrics(
        model1_results,
        model2_results,
        os.path.join(args.output_dir, "metric_comparison.png"),
        args.title1
    )
elif not is_scalar_metrics(model1_results) and not is_scalar_metrics(model2_results):
    plot_class_metrics(
        model1_results,
        model2_results,
        args.class_labels,
        os.path.join(args.output_dir, "per_class_f1_comparison.png"),
        args.title2
    )
else:
    raise ValueError("Mismatch in input JSONs: one contains scalar metrics and the other per-class metrics.")
