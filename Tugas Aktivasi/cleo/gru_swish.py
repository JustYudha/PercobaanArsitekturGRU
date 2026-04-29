from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset_cleo import DATASET_NAME, DATASET_PATH, TARGET_COL
from training_ui import run_training_ui


run_training_ui(
    dataset_name=DATASET_NAME,
    dataset_path=DATASET_PATH,
    target_col=TARGET_COL,
    activation_name="swish",
)
