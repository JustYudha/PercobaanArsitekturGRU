import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "swish": Swish(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Aktivasi tidak dikenal: {name}")
    return activations[name]


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation_name: str):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.activation = build_activation(activation_name)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        gru_out, _ = self.gru(x)
        last_timestep = gru_out[:, -1, :]
        activated = self.activation(last_timestep)
        return self.output(activated)


@dataclass
class ExperimentConfig:
    name: str
    file_path: Path
    target_col: str


def preprocess_data(
    df: pd.DataFrame, target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[target_col], errors="coerce").values.reshape(-1, 1)
    X = df.drop(columns=[target_col])

    # Tangani kolom object campuran angka-teks (mis. "horsepower" pada auto-mpg).
    for col in X.select_dtypes(include=["object"]).columns:
        numeric_candidate = pd.to_numeric(X[col], errors="coerce")
        if numeric_candidate.notna().mean() >= 0.7:
            X[col] = numeric_candidate

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    y_imputer = SimpleImputer(strategy="median")
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_imputer.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    return X_train, X_test, y_train, y_test


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    activation_name: str,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_size: int = 32,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    model = GRURegressor(
        input_size=X_train.shape[1], hidden_size=hidden_size, activation_name=activation_name
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 12
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        avg_val_loss = float(np.mean(val_losses))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().flatten()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "activation": activation_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(config.file_path) if config.file_path.suffix == ".csv" else pd.read_excel(config.file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, config.target_col)

    activations = ["relu", "leaky_relu", "sigmoid", "tanh", "softmax", "swish", "gelu"]
    results: List[Dict[str, float]] = []

    for act in activations:
        metrics = train_one_model(X_train, y_train, X_test, y_test, act)
        results.append(metrics)
        print(f"[{config.name}] selesai aktivasi: {act} | RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    result_df = pd.DataFrame(results).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    return result_df


def main() -> None:
    set_seed(SEED)

    experiments = [
        ExperimentConfig(
            name="Cleo",
            file_path=Path(r"C:\Users\Yudha02\Documents\Itenas\Semester Pendek\Big Data\tim_nhl.csv"),
            target_col="win_pct",
        ),
        ExperimentConfig(
            name="Yudha",
            file_path=Path(r"C:\Users\Yudha02\Documents\Itenas\Semester 5\Data Mining\auto-mpg.xlsx"),
            target_col="mpg",
        ),
    ]

    for exp in experiments:
        print(f"\n=== Eksperimen {exp.name} | Dataset: {exp.file_path.name} ===")
        result_df = run_experiment(exp)
        output_file = Path(f"hasil_gru_{exp.name.lower()}.csv")
        result_df.to_csv(output_file, index=False)
        best = result_df.iloc[0]
        print(result_df.to_string(index=False))
        print(
            f"Aktivasi terbaik ({exp.name}): {best['activation']} "
            f"(RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, R2={best['r2']:.4f})"
        )
        print(f"Hasil disimpan ke: {output_file.resolve()}")


if __name__ == "__main__":
    main()
