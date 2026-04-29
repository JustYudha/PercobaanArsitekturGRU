import random
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
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
    def __init__(
        self, input_size: int, hidden_size: int, activation_name: str, num_layers: int = 1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.activation = build_activation(activation_name)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_timestep = gru_out[:, -1, :]
        activated = self.activation(last_timestep)
        return self.output(activated)


def preprocess_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[target_col], errors="coerce").values.reshape(-1, 1)
    X = df.drop(columns=[target_col])

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
        X, y, test_size=test_size, random_state=SEED
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
    num_layers: int = 1,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    model = GRURegressor(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        activation_name=activation_name,
        num_layers=num_layers,
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


def build_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    raise ValueError(f"Optimizer tidak dikenal: {name}")


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Konversi target regresi ke 3 kelas berbasis kuantil agar metrik klasifikasi dapat dihitung.
    quantiles = np.quantile(y_true, [0.33, 0.66])
    bins = np.array([-np.inf, quantiles[0], quantiles[1], np.inf])
    true_cls = np.digitize(y_true, bins[1:-1], right=False)
    pred_cls = np.digitize(y_pred, bins[1:-1], right=False)

    accuracy = accuracy_score(true_cls, pred_cls)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_cls, pred_cls, average="weighted", zero_division=0
    )
    return {
        "accuracy_percent": float(accuracy * 100.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def train_with_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    activation_name: str,
    epochs: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    optimizer_name: str,
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = GRURegressor(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        activation_name=activation_name,
        num_layers=num_layers,
    ).to(device)
    optimizer = build_optimizer(optimizer_name, model, lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    start = time.time()

    best_val_loss = float("inf")
    best_state = None
    patience = 12
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_losses))

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_batch_losses.append(criterion(pred, yb).item())
        epoch_val_loss = float(np.mean(val_batch_losses))

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - start
        avg_epoch_time = elapsed / epoch
        eta_seconds = max(0.0, avg_epoch_time * (epochs - epoch))
        if progress_callback:
            progress_callback(
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta_seconds,
                }
            )

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

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    regression_metrics = {
        "mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
        "r2": float(r2_score(y_true_np, y_pred_np)),
    }
    cls_metrics = compute_classification_metrics(y_true_np, y_pred_np)

    return {
        "regression_metrics": regression_metrics,
        "classification_metrics": cls_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_completed": len(train_losses),
        "total_seconds": float(time.time() - start),
    }


def run_single_activation(
    dataset_name: str, dataset_path: str, target_col: str, activation_name: str
) -> Dict[str, float]:
    set_seed(SEED)
    data_path = Path(dataset_path)
    df = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_excel(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
    metrics = train_one_model(X_train, y_train, X_test, y_test, activation_name)

    print(f"Dataset     : {dataset_name}")
    print(f"Aktivasi    : {activation_name}")
    print(f"MAE         : {metrics['mae']:.6f}")
    print(f"RMSE        : {metrics['rmse']:.6f}")
    print(f"R2          : {metrics['r2']:.6f}")
    return metrics
