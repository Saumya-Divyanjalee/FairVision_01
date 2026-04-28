"""
FairVision - Evaluation & Fairness Audit Module
=================================================
Handles:
- Full test set evaluation (accuracy, precision, recall, F1, confusion matrix)
- Subgroup accuracy by race and gender
- Fairness gap computation
- Comparison plots across models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List

AGE_NAMES    = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
GENDER_NAMES = ["Male", "Female"]
RACE_NAMES   = ["East Asian","Indian","Black","White","Middle Eastern","Latino_Hispanic","Southeast Asian"]


# ── Collect predictions ─────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over the full loader and collect labels + predictions.

    Returns:
        y_true, y_pred, genders, races  (all np.ndarray)
    """
    model.eval()
    all_labels, all_preds, all_genders, all_races = [], [], [], []

    for images, ages, genders, races in tqdm(loader, desc="  Evaluating"):
        images  = images.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = outputs.max(1)

        all_labels.extend(ages.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_genders.extend(genders.numpy())
        all_races.extend(races.numpy())

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_genders), np.array(all_races))


# ── Overall metrics ─────────────────────────────────────────────────────────────
def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    print_report: bool = True
) -> Dict[str, float]:
    """Compute and optionally print overall classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if print_report:
        print(f"\n{'='*50}")
        print(f"  Overall Performance")
        print(f"{'='*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}  (macro)")
        print(f"  Recall    : {rec:.4f}  (macro)")
        print(f"  F1-Score  : {f1:.4f}  (macro)")
        print()
        print(classification_report(y_true, y_pred, target_names=AGE_NAMES, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── Confusion matrix ────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=AGE_NAMES, yticklabels=AGE_NAMES)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved → {save_path}")
    plt.show()


# ── Subgroup accuracy ───────────────────────────────────────────────────────────
def compute_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    group_names: List[str]
) -> Dict[str, Optional[float]]:
    """Compute per-group accuracy."""
    results = {}
    for i, name in enumerate(group_names):
        mask = group_labels == i
        if mask.sum() == 0:
            results[name] = None
            continue
        results[name] = round(float(accuracy_score(y_true[mask], y_pred[mask])), 4)
    return results


def fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: np.ndarray,
    races: np.ndarray,
    model_name: str = "Model",
    save_dir: str = "outputs/plots"
) -> Tuple[Dict, Dict]:
    """
    Full fairness audit: race-wise and gender-wise accuracy.
    Prints gaps and saves bar charts.

    Returns:
        race_acc_dict, gender_acc_dict
    """
    race_acc   = compute_group_accuracy(y_true, y_pred, races,   RACE_NAMES)
    gender_acc = compute_group_accuracy(y_true, y_pred, genders, GENDER_NAMES)

    race_vals = [v for v in race_acc.values() if v is not None]
    best_race  = max(race_vals)
    worst_race = min(race_vals)
    gap_race   = round(best_race - worst_race, 4)

    print(f"\n{'='*50}")
    print(f"  Fairness Audit — {model_name}")
    print(f"{'='*50}")
    print(f"  Race-wise accuracy:")
    for name, acc in race_acc.items():
        marker = " ← best" if acc == best_race else (" ← worst" if acc == worst_race else "")
        print(f"    {name:22s}: {acc:.4f}{marker}")
    print(f"\n  Best  race accuracy : {best_race:.4f}")
    print(f"  Worst race accuracy : {worst_race:.4f}")
    print(f"  Race accuracy gap   : {gap_race:.4f}")

    print(f"\n  Gender-wise accuracy:")
    for name, acc in gender_acc.items():
        print(f"    {name:10s}: {acc:.4f}")

    # Plot race bar chart
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(11, 4))
    colors = [
        "green" if v == best_race else ("red" if v == worst_race else "steelblue")
        for v in race_acc.values()
    ]
    plt.bar(race_acc.keys(), race_acc.values(), color=colors, edgecolor="black")
    mean_acc = np.mean(race_vals)
    plt.axhline(mean_acc, color="orange", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_acc:.3f}")
    plt.title(f"Race-Wise Accuracy — {model_name}", fontsize=13)
    plt.xlabel("Race Group")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace(":", "")
    plt.savefig(f"{save_dir}/{fname}_race_accuracy.png", dpi=150)
    plt.show()

    # Plot gender bar chart
    plt.figure(figsize=(5, 4))
    plt.bar(gender_acc.keys(), gender_acc.values(),
            color=["royalblue", "hotpink"], edgecolor="black")
    plt.title(f"Gender-Wise Accuracy — {model_name}", fontsize=13)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fname}_gender_accuracy.png", dpi=150)
    plt.show()

    return race_acc, gender_acc


# ── Comparative analysis ────────────────────────────────────────────────────────
def compare_models(
    results: List[Dict],
    save_dir: str = "outputs"
):
    """
    Build comparison table and grouped bar charts.

    Args:
        results: list of dicts, each with keys:
                 name, overall, race_acc, gender_acc
    """
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/results", exist_ok=True)

    rows = []
    for r in results:
        race_vals = [v for v in r["race_acc"].values() if v is not None]
        rows.append({
            "Model":             r["name"],
            "Overall Accuracy":  round(r["overall"]["accuracy"], 4),
            "Macro F1":          round(r["overall"]["f1"], 4),
            "Best Race Acc":     round(max(race_vals), 4),
            "Worst Race Acc":    round(min(race_vals), 4),
            "Race Gap":          round(max(race_vals) - min(race_vals), 4),
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("  COMPARATIVE ANALYSIS")
    print("="*70)
    print(df.to_string(index=False))
    df.to_csv(f"{save_dir}/results/comparison_table.csv", index=False)
    print(f"\n  Saved → {save_dir}/results/comparison_table.csv")

    # Grouped bar — race accuracy
    race_names  = RACE_NAMES
    model_names = [r["name"] for r in results]
    x     = np.arange(len(race_names))
    width = 0.8 / len(results)
    palette = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, r in enumerate(results):
        vals = [r["race_acc"].get(rn, 0) or 0 for rn in race_names]
        ax.bar(x + i * width, vals, width, label=r["name"],
               color=palette[i % len(palette)], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Race Group")
    ax.set_ylabel("Accuracy")
    ax.set_title("Race-Wise Accuracy: Model Comparison", fontsize=13)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(race_names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/comparison_race_accuracy.png", dpi=150)
    plt.show()

    # Gender comparison
    gender_names = GENDER_NAMES
    x2 = np.arange(len(gender_names))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(results):
        vals = [r["gender_acc"].get(gn, 0) or 0 for gn in gender_names]
        ax.bar(x2 + i * width, vals, width, label=r["name"],
               color=palette[i % len(palette)], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Gender")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gender-Wise Accuracy: Model Comparison", fontsize=13)
    ax.set_xticks(x2 + width * (len(results) - 1) / 2)
    ax.set_xticklabels(gender_names)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/comparison_gender_accuracy.png", dpi=150)
    plt.show()

    return df
