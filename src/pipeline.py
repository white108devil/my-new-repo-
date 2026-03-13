from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import RobustScaler

from data_utils import BankFeatureEngineer, ensure_dirs, load_dataset, split_features_labels
from nng_mix import NNGMixGenerator
from visualization import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_score_distributions,
    plot_scatter,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_threshold(
    scores: np.ndarray,
    y_true: np.ndarray | None,
    method: str,
    expected_rate: float,
    anomaly_label: int,
) -> float:
    if method == "label_f1" and y_true is not None:
        y_binary = (y_true == 1).astype(int)
        candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, 60)))
        best_f1 = -1.0
        best_threshold = float(candidates[0])
        for threshold in candidates:
            preds = (scores >= threshold).astype(int)
            score = f1_score(y_binary, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold

    rate = min(max(expected_rate, 1e-6), 0.5)
    return float(np.quantile(scores, 1 - rate))


def coerce_binary_labels(
    y: np.ndarray, anomaly_label: int, normal_label: int
) -> np.ndarray:
    series = pd.Series(y)
    if pd.api.types.is_numeric_dtype(series):
        return (series == anomaly_label).astype(int).to_numpy()

    anomalies = {
        str(anomaly_label).lower(),
        "1",
        "true",
        "fraud",
        "anomaly",
        "yes",
        "y",
    }
    normals = {
        str(normal_label).lower(),
        "0",
        "false",
        "normal",
        "no",
        "n",
    }

    lower = series.astype(str).str.lower()
    mapped = lower.apply(lambda value: 1 if value in anomalies else 0 if value in normals else 0)
    return mapped.to_numpy()


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


class ConsolePrinter:
    def banner(self, title: str, subtitle: str) -> None:
        line = "=" * 70
        print(line)
        print(title.upper())
        print(line)
        print(subtitle)
        print(line)

    def step(self, number: int, title: str) -> None:
        line = "=" * 70
        print("\n" + line)
        print(f"STEP {number}: {title.upper()}")
        print(line)

    def info(self, text: str) -> None:
        print(text)

    def block(self, title: str, content: str) -> None:
        print(f"\n{title}")
        print(content)


class DatasetProfiler:
    def __init__(self, printer: ConsolePrinter) -> None:
        self.printer = printer

    def summarize(self, df: pd.DataFrame, y_binary: np.ndarray) -> None:
        total = len(df)
        fraud_count = int(y_binary.sum())
        ratio = fraud_count / total if total else 0.0
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)

        self.printer.block(
            "Dataset Summary:",
            "\n".join(
                [
                    f"Total transactions: {total:,}",
                    f"Columns: {df.shape[1]}",
                    f"Memory usage: {memory_mb:.2f} MB",
                    f"Fraud transactions: {fraud_count:,}",
                    f"Fraud ratio: {ratio:.4f}",
                ]
            ),
        )

    def preview(self, df: pd.DataFrame, rows: int = 5) -> None:
        self.printer.block("First 5 rows:", df.head(rows).to_string())

    def column_info(self, df: pd.DataFrame) -> None:
        lines = []
        for col in df.columns:
            dtype = df[col].dtype
            unique = df[col].nunique(dropna=True)
            missing = df[col].isna().sum()
            lines.append(f"{col}: {dtype}, {unique} unique, {missing} missing")
        self.printer.block("Column Information:", "\n".join(lines))

    def numeric_summary(self, df: pd.DataFrame) -> None:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            self.printer.block("Statistical Summary (numeric columns):", "No numeric columns found.")
            return
        summary = numeric.describe().T
        self.printer.block("Statistical Summary (numeric columns):", summary.to_string())


class FraudDetectionPipeline:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.printer = ConsolePrinter()

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Metrics:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        roc_auc = None
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, scores)
        return Metrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc) if roc_auc is not None else None,
        )

    def run(self) -> None:
        cfg = load_config(self.config_path)

        self.printer.banner(
            "Bank Transaction Anamoly Detection System",
            "Using NNG Mix with Pseudo Anomalies",
        )

        ensure_dirs(
            [
                "data/processed",
                "data/output",
                "outputs/plots",
                "models",
            ]
        )
        self.printer.info("[OK] Output directory ready")

        data_cfg = cfg["data"]
        label_column = data_cfg["label_column"]
        anomaly_label = int(data_cfg.get("anomaly_label", 1))
        normal_label = int(data_cfg.get("normal_label", 0))

        self.printer.step(1, "Loading Bank Transaction Dataset")
        self.printer.info(f"Loading dataset from: {data_cfg['raw_path']}")

        df = load_dataset(data_cfg["raw_path"])
        features_df, y_raw = split_features_labels(df, label_column)
        y_binary = coerce_binary_labels(y_raw, anomaly_label, normal_label)

        profiler = DatasetProfiler(self.printer)
        profiler.summarize(df, y_binary)

        self.printer.step(2, "Exploring Dataset Structure")
        profiler.preview(df)
        profiler.column_info(df)
        profiler.numeric_summary(df)

        self.printer.step(3, "Preprocessing Data")
        engineer = BankFeatureEngineer(
            drop_columns=data_cfg.get("drop_columns", []),
            fillna_strategy=data_cfg.get("fillna_strategy", "median"),
        )
        features_df = engineer.fit_transform(features_df)

        processed_path = cfg.get("outputs", {}).get(
            "processed_features_csv", "data/processed/processed_features.csv"
        )
        Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(processed_path, index=False)

        self.printer.info(f"Processed dataset shape: {features_df.shape}")
        self.printer.info(f"Number of features: {features_df.shape[1]}")
        if engineer.report.created_features:
            self.printer.info(
                f"Features created: {engineer.report.created_features}"
            )

        feature_names = list(features_df.columns)
        X = features_df.to_numpy(dtype=float)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        rng = np.random.default_rng(cfg["nng_mix"].get("random_state", 42))
        anomaly_indices = np.where(y_binary == 1)[0]
        if len(anomaly_indices) == 0:
            raise ValueError("No labeled anomalies found. Check the label column values.")

        labeled_fraction = float(cfg["nng_mix"].get("labeled_anomaly_fraction", 0.2))
        labeled_count = max(1, int(len(anomaly_indices) * labeled_fraction))
        labeled_indices = rng.choice(anomaly_indices, size=labeled_count, replace=False)
        unlabeled_indices = np.setdiff1d(np.arange(len(y_binary)), labeled_indices)

        X_labeled = X_scaled[labeled_indices]
        X_unlabeled = X_scaled[unlabeled_indices]

        self.printer.step(4, "Generating Pseudo Anomalies")
        mix_cfg = cfg["nng_mix"]
        generator = NNGMixGenerator(
            k_neighbors=mix_cfg["k_neighbors"],
            pseudo_per_anomaly=mix_cfg["pseudo_per_anomaly"],
            mix_mu=mix_cfg["mix_mu"],
            mix_sigma=mix_cfg["mix_sigma"],
            noise_std=mix_cfg["noise_std"],
            random_state=mix_cfg["random_state"],
        )
        X_pseudo = generator.generate(X_labeled, X_unlabeled)
        self.printer.info(f"Normal training samples: {len(X_unlabeled):,}")
        self.printer.info(f"Labeled anomalies: {len(X_labeled):,}")
        self.printer.info(f"Pseudo anomalies generated: {len(X_pseudo):,}")
        self.printer.info(
            f"Augmented training size: {len(X_unlabeled) + len(X_labeled) + len(X_pseudo):,}"
        )

        self.printer.step(5, "Training NNG Mix Model")
        self.printer.info("Training NNG Mix anomaly detector...")
        X_train = np.vstack([X_unlabeled, X_labeled, X_pseudo])
        y_train = np.concatenate(
            [
                np.zeros(len(X_unlabeled), dtype=int),
                np.ones(len(X_labeled), dtype=int),
                np.ones(len(X_pseudo), dtype=int),
            ]
        )

        model_cfg = cfg["model"]
        model = RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            min_samples_leaf=model_cfg["min_samples_leaf"],
            class_weight=model_cfg["class_weight"],
            random_state=model_cfg["random_state"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        nng_scores_raw = model.predict_proba(X_scaled)[:, 1]

        nng_scores = nng_scores_raw

        self.printer.step(6, "Scoring and Evaluation")
        threshold_cfg = cfg["threshold"]
        threshold_method = threshold_cfg.get("method", "expected_rate")
        expected_rate = threshold_cfg.get("expected_anomaly_rate", 0.05)

        nng_threshold = compute_threshold(
            nng_scores, y_binary, threshold_method, expected_rate, 1
        )
        nng_pred = (nng_scores >= nng_threshold).astype(int)
        nng_metrics = self._metrics(y_binary, nng_pred, nng_scores)

        self.printer.info(
            "NNG Mix Metrics "
            f"(P={nng_metrics.precision:.3f}, R={nng_metrics.recall:.3f}, "
            f"F1={nng_metrics.f1:.3f})"
        )

        self.printer.step(7, "Saving Outputs")
        predictions_df = df.copy()
        predictions_df["nng_mix_score"] = nng_scores
        predictions_df["predicted_anomaly"] = nng_pred
        predictions_df["predicted_label"] = np.where(
            nng_pred == 1, "Fraud", "Normal"
        )

        outputs_cfg = cfg["outputs"]
        predictions_df.to_csv(outputs_cfg["predictions_csv"], index=False)

        fraud_df = predictions_df.loc[nng_pred == 1].copy()
        fraud_df.to_csv(outputs_cfg["fraud_csv"], index=False)

        clean_df = predictions_df.loc[nng_pred == 0].copy()
        clean_df.to_csv(outputs_cfg.get("clean_csv", "data/output/clean_transactions.csv"), index=False)

        if X_pseudo.size:
            pseudo_df = pd.DataFrame(X_pseudo, columns=feature_names).assign(pseudo_anomaly=1)
            pseudo_df.to_csv(outputs_cfg["pseudo_csv"], index=False)

        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
                "thresholds": {
                    "nng_mix": float(nng_threshold),
                },
            },
            outputs_cfg["model_path"],
        )

        summary = {
            "rows": int(len(df)),
            "labeled_anomalies": int(len(labeled_indices)),
            "pseudo_anomalies": int(len(X_pseudo)),
            "predicted_anomalies": int(nng_pred.sum()),
            "thresholds": {
                "nng_mix": float(nng_threshold),
            },
            "metrics": {
                "nng_mix": nng_metrics.__dict__,
            },
        }
        summary_path = outputs_cfg.get("summary_path", "data/output/run_summary.json")
        Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self.printer.step(8, "Generating Visualizations")
        viz_cfg = cfg["visualization"]
        show_plots = bool(viz_cfg.get("show_plots", False))

        labeled_mask = np.zeros(len(y_binary), dtype=bool)
        labeled_mask[labeled_indices] = True

        plot_scatter(
            X_all=X_scaled,
            scores=nng_scores,
            predicted_anomaly=nng_pred.astype(bool),
            labeled_mask=labeled_mask,
            X_pseudo=X_pseudo,
            output_path=viz_cfg["output_path"],
            sample_size=viz_cfg["sample_size"],
            random_state=viz_cfg.get("random_state", 42),
            show=show_plots,
        )

        plot_confusion_matrices(
            y_true=y_binary,
            preds={
                "NNG Mix": nng_pred,
            },
            output_path=viz_cfg.get("confusion_matrix_path", "outputs/plots/confusion_matrices.png"),
            show=show_plots,
        )

        plot_roc_curves(
            y_true=y_binary,
            scores={
                "NNG Mix": nng_scores,
            },
            output_path=viz_cfg.get("roc_curve_path", "outputs/plots/roc_curves.png"),
            show=show_plots,
        )

        plot_score_distributions(
            y_true=y_binary,
            scores={
                "NNG Mix": nng_scores,
            },
            output_path=viz_cfg.get(
                "score_distribution_path", "outputs/plots/score_distributions.png"
            ),
            show=show_plots,
        )

        self.printer.info("All outputs saved successfully.")


def main(config_path: str) -> None:
    FraudDetectionPipeline(config_path).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NNG-Mix semi-supervised fraud detection pipeline"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
