import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


@dataclass
class AppPaths:
    base: Path
    config: Path
    data_raw: Path
    plot: Path
    confusion: Path
    roc: Path
    score_dist: Path
    fraud_csv: Path
    clean_csv: Path
    predictions_csv: Path
    summary_json: Path


class FraudDetectionApp:
    def __init__(self) -> None:
        base = Path(__file__).resolve().parent.parent
        self.paths = AppPaths(
            base=base,
            config=base / "config.yaml",
            data_raw=base / "data" / "raw",
            plot=base / "outputs" / "plots" / "scatter_nng_mix.png",
            confusion=base / "outputs" / "plots" / "confusion_matrices.png",
            roc=base / "outputs" / "plots" / "roc_curves.png",
            score_dist=base / "outputs" / "plots" / "score_distributions.png",
            fraud_csv=base / "data" / "output" / "fraud_transactions.csv",
            clean_csv=base / "data" / "output" / "clean_transactions.csv",
            predictions_csv=base / "data" / "output" / "anomaly_predictions.csv",
            summary_json=base / "data" / "output" / "run_summary.json",
        )

    def load_config(self) -> dict:
        with self.paths.config.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def save_config(self, cfg: dict) -> Path:
        temp_cfg = self.paths.base / "config.app.yaml"
        with temp_cfg.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        return temp_cfg

    def upload_dataset(self, filename: str, sample_rows: int) -> Path:
        uploaded = st.file_uploader("Upload Bank_Transaction_Fraud_Detection.csv", type=["csv"])
        if not uploaded:
            return Path()

        self.paths.data_raw.mkdir(parents=True, exist_ok=True)
        target_path = self.paths.data_raw / filename

        if sample_rows > 0:
            df = pd.read_csv(uploaded, nrows=sample_rows)
            df.to_csv(target_path, index=False)
            st.info(f"Saved a sample of {len(df)} rows to {target_path.name}.")
        else:
            target_path.write_bytes(uploaded.getbuffer())
            st.info(f"Saved full dataset to {target_path.name}.")

        return target_path

    def render_sidebar(self, cfg: dict) -> dict:
        st.sidebar.header("Options")

        data_cfg = cfg["data"]
        nng_cfg = cfg["nng_mix"]
        model_cfg = cfg["model"]
        threshold_cfg = cfg["threshold"]
        viz_cfg = cfg["visualization"]

        data_cfg["raw_path"] = st.sidebar.text_input(
            "Raw CSV path", value=data_cfg["raw_path"]
        )
        data_cfg["label_column"] = st.sidebar.text_input(
            "Label column", value=data_cfg["label_column"]
        )

        data_cfg["fillna_strategy"] = st.sidebar.selectbox(
            "Fill missing values",
            options=["median", "zero", "ffill"],
            index=["median", "zero", "ffill"].index(data_cfg.get("fillna_strategy", "median")),
        )

        nng_cfg["labeled_anomaly_fraction"] = st.sidebar.slider(
            "Labeled anomaly fraction",
            min_value=0.05,
            max_value=0.9,
            value=float(nng_cfg["labeled_anomaly_fraction"]),
            step=0.05,
        )
        nng_cfg["k_neighbors"] = st.sidebar.slider(
            "NNG neighbors (k)", min_value=1, max_value=30, value=int(nng_cfg["k_neighbors"])
        )
        nng_cfg["pseudo_per_anomaly"] = st.sidebar.slider(
            "Pseudo anomalies per labeled",
            min_value=1,
            max_value=20,
            value=int(nng_cfg["pseudo_per_anomaly"]),
        )
        nng_cfg["mix_mu"] = st.sidebar.slider(
            "Mix mu", min_value=0.0, max_value=1.5, value=float(nng_cfg["mix_mu"]), step=0.05
        )
        nng_cfg["mix_sigma"] = st.sidebar.slider(
            "Mix sigma", min_value=0.0, max_value=1.0, value=float(nng_cfg["mix_sigma"]), step=0.05
        )
        nng_cfg["noise_std"] = st.sidebar.slider(
            "Noise std", min_value=0.0, max_value=0.2, value=float(nng_cfg["noise_std"]), step=0.01
        )

        model_cfg["n_estimators"] = st.sidebar.slider(
            "Trees (n_estimators)", min_value=50, max_value=500, value=int(model_cfg["n_estimators"]), step=50
        )
        model_cfg["max_depth"] = st.sidebar.slider(
            "Max depth", min_value=4, max_value=30, value=int(model_cfg["max_depth"]), step=2
        )
        model_cfg["min_samples_leaf"] = st.sidebar.slider(
            "Min samples leaf",
            min_value=1,
            max_value=10,
            value=int(model_cfg["min_samples_leaf"]),
            step=1,
        )

        threshold_cfg["method"] = st.sidebar.selectbox(
            "Threshold method",
            options=["expected_rate", "label_f1"],
            index=["expected_rate", "label_f1"].index(threshold_cfg.get("method", "expected_rate")),
        )
        threshold_cfg["expected_anomaly_rate"] = st.sidebar.slider(
            "Expected anomaly rate",
            min_value=0.0005,
            max_value=0.02,
            value=float(threshold_cfg.get("expected_anomaly_rate", 0.002)),
            step=0.0005,
            format="%.4f",
        )

        viz_cfg["sample_size"] = st.sidebar.slider(
            "Plot sample size",
            min_value=1000,
            max_value=20000,
            value=int(viz_cfg.get("sample_size", 8000)),
            step=1000,
        )
        viz_cfg["show_plots"] = st.sidebar.checkbox(
            "Show plot windows during run",
            value=bool(viz_cfg.get("show_plots", False)),
        )

        cfg["data"] = data_cfg
        cfg["nng_mix"] = nng_cfg
        cfg["model"] = model_cfg
        cfg["threshold"] = threshold_cfg
        cfg["visualization"] = viz_cfg
        return cfg

    def run_pipeline(self, cfg_path: Path) -> None:
        cmd = [sys.executable, str(self.paths.base / "src" / "pipeline.py"), "--config", str(cfg_path)]
        subprocess.run(cmd, check=True)

    def render_outputs(self) -> None:
        if self.paths.plot.exists():
            st.image(str(self.paths.plot), caption="NNG-Mix Scatter Plot")
        if self.paths.confusion.exists():
            st.image(str(self.paths.confusion), caption="Confusion Matrix")
        if self.paths.roc.exists():
            st.image(str(self.paths.roc), caption="ROC Curve")
        if self.paths.score_dist.exists():
            st.image(str(self.paths.score_dist), caption="Score Distribution")

        if self.paths.summary_json.exists():
            summary = self.paths.summary_json.read_text(encoding="utf-8")
            st.code(summary, language="json")

        if self.paths.fraud_csv.exists():
            with self.paths.fraud_csv.open("rb") as f:
                st.download_button("Download Fraud CSV", f, file_name="fraud_transactions.csv")

        if self.paths.clean_csv.exists():
            with self.paths.clean_csv.open("rb") as f:
                st.download_button("Download Clean CSV", f, file_name="clean_transactions.csv")

        if self.paths.predictions_csv.exists():
            with self.paths.predictions_csv.open("rb") as f:
                st.download_button("Download Predictions CSV", f, file_name="anomaly_predictions.csv")

    def render(self) -> None:
        st.title("Bank Transaction Anamoly Detection (NNG-Mix)")
        st.write("Upload the dataset, tune options, and run detection.")

        cfg = self.load_config()
        cfg = self.render_sidebar(cfg)

        sample_rows = st.number_input(
            "Optional: use only the first N rows (0 = full dataset)",
            min_value=0,
            max_value=500000,
            value=0,
            step=50000,
        )

        filename = Path(cfg["data"]["raw_path"]).name
        dataset_path = self.upload_dataset(filename, int(sample_rows))

        if st.button("Run Detection"):
            if not dataset_path:
                st.error("Please upload the dataset CSV first.")
                return
            cfg_path = self.save_config(cfg)
            with st.spinner("Running pipeline..."):
                self.run_pipeline(cfg_path)
            st.success("Detection complete.")
            self.render_outputs()

        st.divider()
        st.subheader("Latest Outputs")
        self.render_outputs()


def main() -> None:
    FraudDetectionApp().render()


if __name__ == "__main__":
    main()
