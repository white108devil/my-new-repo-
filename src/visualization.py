from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _finalize_figure(fig: plt.Figure, output_path: str, show: bool) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_scatter(
    X_all: np.ndarray,
    scores: np.ndarray,
    predicted_anomaly: np.ndarray,
    labeled_mask: np.ndarray,
    X_pseudo: np.ndarray,
    output_path: str,
    sample_size: int,
    random_state: int | None = None,
    show: bool = False,
) -> None:
    rng = np.random.default_rng(random_state)
    n_samples = X_all.shape[0]
    idx = np.arange(n_samples)

    if sample_size and n_samples > sample_size:
        important = predicted_anomaly | labeled_mask
        keep_idx = idx[important]
        remaining = idx[~important]
        if len(keep_idx) < sample_size and len(remaining) > 0:
            extra = rng.choice(
                remaining, size=min(sample_size - len(keep_idx), len(remaining)), replace=False
            )
            idx = np.concatenate([keep_idx, extra])
        else:
            idx = keep_idx

    X_plot = X_all[idx]
    scores_plot = scores[idx]
    predicted_plot = predicted_anomaly[idx]
    labeled_plot = labeled_mask[idx]

    fit_data = np.vstack([X_all, X_pseudo]) if X_pseudo.size else X_all
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(fit_data)

    X_plot_2d = pca.transform(X_plot)
    X_pseudo_2d = pca.transform(X_pseudo) if X_pseudo.size else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    scatter = ax.scatter(
        X_plot_2d[:, 0],
        X_plot_2d[:, 1],
        c=scores_plot,
        cmap="turbo",
        s=12,
        alpha=0.85,
        linewidths=0,
        label="Transactions (colored by anomaly score)",
    )

    if predicted_plot.any():
        ax.scatter(
            X_plot_2d[predicted_plot, 0],
            X_plot_2d[predicted_plot, 1],
            facecolors="none",
            edgecolors="black",
            s=40,
            linewidths=0.7,
            label="Predicted fraud transactions",
        )

    if labeled_plot.any():
        ax.scatter(
            X_plot_2d[labeled_plot, 0],
            X_plot_2d[labeled_plot, 1],
            c="black",
            marker="x",
            s=60,
            linewidths=1.2,
            label="Labeled anomalies",
        )

    if X_pseudo_2d.size:
        ax.scatter(
            X_pseudo_2d[:, 0],
            X_pseudo_2d[:, 1],
            c="magenta",
            marker="D",
            s=24,
            alpha=0.7,
            label="NNG-Mix pseudo anomalies",
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Anomaly risk score")

    ax.set_title("Transaction Risk Map (NNG-Mix + KNN)")
    ax.set_xlabel("Behavior Axis 1 (PCA projection)")
    ax.set_ylabel("Behavior Axis 2 (PCA projection)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.2)

    _finalize_figure(fig, output_path, show)


def plot_confusion_matrices(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    output_path: str,
    labels: tuple[int, int] = (0, 1),
    show: bool = False,
) -> None:
    n_panels = len(preds)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=140)
    if n_panels == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=list(labels))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_xlabel("Predicted Transaction Status")
        ax.set_ylabel("Actual Transaction Status")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_yticklabels(["Normal", "Fraud"])

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _finalize_figure(fig, output_path, show)


def plot_roc_curves(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str,
    show: bool = False,
) -> None:
    n_panels = len(scores)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=140)
    if n_panels == 1:
        axes = [axes]

    for ax, (name, score_values) in zip(axes, scores.items()):
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, "ROC curve needs both classes", ha="center", va="center")
            ax.set_axis_off()
            continue

        fpr, tpr, _ = roc_curve(y_true, score_values)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Alarm Rate (Normal flagged as Fraud)")
        ax.set_ylabel("Fraud Detection Rate (Recall)")
        ax.set_title(f"{name}\nROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.2)

    _finalize_figure(fig, output_path, show)


def plot_score_distributions(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str,
    show: bool = False,
) -> None:
    n_panels = len(scores)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=140)
    if n_panels == 1:
        axes = [axes]

    for ax, (name, score_values) in zip(axes, scores.items()):
        normal_scores = score_values[y_true == 0]
        fraud_scores = score_values[y_true == 1]

        ax.hist(
            normal_scores,
            bins=50,
            alpha=0.7,
            color="steelblue",
            label="Normal",
            density=True,
        )
        if len(fraud_scores) > 0:
            ax.hist(
                fraud_scores,
                bins=50,
                alpha=0.7,
                color="crimson",
                label="Fraud",
                density=True,
            )

        ax.set_title(f"{name}\nScore Distribution")
        ax.set_xlabel("Anomaly Risk Score (0 = Normal, 1 = Fraud)")
        ax.set_ylabel("Relative Frequency")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)

    _finalize_figure(fig, output_path, show)
