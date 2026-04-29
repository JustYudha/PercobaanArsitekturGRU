import streamlit as st

from training_ui import run_training_ui


st.set_page_config(page_title="Tugas DeepLearning Menggunakan GRU", layout="wide")
st.title("Inang Eksperimen GRU")
st.caption("Pilih pemilik dataset dan aktivasi, lalu jalankan training dari satu UI.")

dataset_options = {
    "Cleo": {
        "dataset_path": r"C:\Users\Yudha02\Documents\Itenas\Semester Pendek\Big Data\tim_nhl.csv",
        "target_col": "win_pct",
        "folder": "cleo",
    },
    "Yudha": {
        "dataset_path": r"C:\Users\Yudha02\Documents\Itenas\Semester 5\Data Mining\auto-mpg.xlsx",
        "target_col": "mpg",
        "folder": "yudha",
    },
}

activation_options = [
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "softmax",
    "swish",
    "gelu",
]

col1, col2 = st.columns(2)
with col1:
    selected_owner = st.selectbox("Pilih Dataset", list(dataset_options.keys()))
with col2:
    selected_activation = st.selectbox("Pilih Aktivasi", activation_options)

cfg = dataset_options[selected_owner]
source_file = f"{cfg['folder']}/gru_{selected_activation}.py"
st.info(f"File code yang dipanggil: `{source_file}`")

run_training_ui(
    dataset_name=selected_owner,
    dataset_path=cfg["dataset_path"],
    target_col=cfg["target_col"],
    activation_name=selected_activation,
    show_header=False,
)
