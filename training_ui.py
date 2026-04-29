from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from gru_common import preprocess_data, set_seed, train_with_history


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    data_path = Path(dataset_path)
    if data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    return pd.read_excel(data_path)


def _format_time(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_training_ui(
    dataset_name: str,
    dataset_path: str,
    target_col: str,
    activation_name: str,
    show_header: bool = True,
) -> None:
    if show_header:
        st.set_page_config(page_title=f"GRU {dataset_name} - {activation_name}", layout="wide")
        st.title(f"GRU Training UI - {dataset_name} ({activation_name})")
    else:
        st.subheader(f"GRU Training UI - {dataset_name} ({activation_name})")

    df = _load_dataset(dataset_path)
    st.caption(f"Dataset: `{dataset_path}` | Baris: {len(df)} | Kolom: {len(df.columns)}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Panel Input Konfigurasi")
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
        epochs = st.slider("Epoch", min_value=10, max_value=300, value=100, step=10)
        optimizer_name = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)

    with right:
        st.subheader("Dataset Splitter + Model Depth")
        test_size = st.slider("Test Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        hidden_size = st.selectbox("Hidden Size", [16, 32, 64, 128], index=1)
        model_depth = st.slider("Model Depth (GRU Layers)", min_value=1, max_value=4, value=1, step=1)
        st.markdown(f"- Aktivasi: `{activation_name}`")
        st.markdown(f"- Target: `{target_col}`")

    if st.button("Start Training", type="primary"):
        set_seed()
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col, test_size=test_size)

        st.subheader("Progress Bar")
        progress = st.progress(0)
        eta_text = st.empty()

        st.subheader("Live Training Graphs")
        chart_placeholder = st.empty()

        live_rows: Dict[str, list] = {"epoch": [], "train_loss": [], "val_loss": []}

        def _on_progress(payload: Dict[str, float]) -> None:
            epoch = int(payload["epoch"])
            total_epochs = int(payload["epochs"])
            live_rows["epoch"].append(epoch)
            live_rows["train_loss"].append(payload["train_loss"])
            live_rows["val_loss"].append(payload["val_loss"])
            progress.progress(min(epoch / total_epochs, 1.0))
            eta_text.write(
                f"Estimated Time: {_format_time(payload['eta_seconds'])} | Elapsed: {_format_time(payload['elapsed_seconds'])}"
            )
            chart_df = pd.DataFrame(live_rows).set_index("epoch")
            chart_placeholder.line_chart(chart_df)

        result = train_with_history(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            activation_name=activation_name,
            epochs=epochs,
            lr=float(learning_rate),
            hidden_size=int(hidden_size),
            num_layers=int(model_depth),
            batch_size=int(batch_size),
            optimizer_name=optimizer_name,
            progress_callback=_on_progress,
        )

        st.success("Training selesai.")
        st.subheader("Grafik Perubahan Loss")
        loss_history = result["regression_loss_history"]
        final_loss_df = pd.DataFrame({"epoch": list(range(1, len(loss_history["mse"]) + 1))})
        final_loss_df["MSE"] = loss_history["mse"]
        final_loss_df["MAE"] = loss_history["mae"]
        final_loss_df["Huber Loss"] = loss_history["huber_loss"]
        final_loss_df["Log-Cosh Loss"] = loss_history["log_cosh_loss"]
        final_loss_df["Quantile Loss (q=0.5)"] = loss_history["quantile_loss"]
        final_loss_df["MBE"] = loss_history["mbe"]
        final_loss_df["RMSE"] = loss_history["rmse"]
        final_loss_df = final_loss_df.set_index("epoch")
        st.line_chart(final_loss_df)

        st.subheader("Final Metrics Table")
        cls = result["classification_metrics"]
        reg = result["regression_metrics"]
        table_df = pd.DataFrame(
            [
                {
                    "MSE": round(reg["mse"], 6),
                    "MAE": round(reg["mae"], 6),
                    "Huber Loss": round(reg["huber_loss"], 6),
                    "Log-Cosh Loss": round(reg["log_cosh_loss"], 6),
                    "Quantile Loss (q=0.5)": round(reg["quantile_loss"], 6),
                    "MBE": round(reg["mbe"], 6),
                    "RMSE": round(reg["rmse"], 6),
                    "R2": round(reg["r2"], 6),
                    "Akurasi Akhir (%)": round(cls["accuracy_percent"], 4),
                    "F1-Score": round(cls["f1_score"], 4),
                    "Precision": round(cls["precision"], 4),
                    "Recall": round(cls["recall"], 4),
                    "Epoch Selesai": int(result["epochs_completed"]),
                    "Total Waktu": _format_time(result["total_seconds"]),
                }
            ]
        )
        st.dataframe(table_df, use_container_width=True)
