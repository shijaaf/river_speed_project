import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PHASE3_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_prediction_table,
    calculate_metrics,
    clean_target,
    set_seed,
    split_features_target,
)


class RiverSpeedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_feature_dataset():
    return pd.read_csv(PHASE3_FEATURE_FILE)


def split_features_and_target(feature_df):
    dataset_names = feature_df["dataset_name"].values
    y = feature_df["real_speed"].values.astype(np.float32)

    X = feature_df.drop(
        columns=["dataset_name", "real_speed"],
    ).values.astype(np.float32)

    return X, y, dataset_names


def train_one_model(
    X_train,
    y_train,
    input_size,
    epochs=300,
    learning_rate=0.001,
):
    model = RiverSpeedMLP(input_size=input_size)

    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.001,
    )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).view(-1, 1)

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        predictions = model(X_train_tensor)

        loss = criterion(
            predictions,
            y_train_tensor,
        )

        loss.backward()
        optimizer.step()

    return model


def evaluate_with_leave_one_out(X, y):
    loo = LeaveOneOut()
    predictions = []

    for train_index, test_index in loo.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = train_one_model(
            X_train=X_train_scaled,
            y_train=y_train,
            input_size=X.shape[1],
            epochs=300,
            learning_rate=0.001,
        )

        model.eval()

        X_test_tensor = torch.tensor(
            X_test_scaled,
            dtype=torch.float32,
        )

        with torch.no_grad():
            predicted_speed = model(X_test_tensor).item()

        predictions.append(predicted_speed)

    return np.array(predictions)


def train_final_model(X, y):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = train_one_model(
        X_train=X_scaled,
        y_train=y,
        input_size=X.shape[1],
        epochs=500,
        learning_rate=0.001,
    )

    return model, scaler


def save_outputs(model, scaler, result_df, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "mlp_deep_learning_model.pth"
    scaler_path = MODELS_DIR / "mlp_scaler.pkl"
    result_path = RESULTS_DIR / "phase5_dl_prediction_results.xlsx"
    metrics_path = RESULTS_DIR / "phase5_dl_metrics.csv"

    torch.save(model.state_dict(), model_path)

    pd.to_pickle(scaler, scaler_path)

    result_df.to_excel(result_path, index=False)

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Deep learning model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Prediction table saved to: {result_path}")
    print(f"Metrics saved to: {metrics_path}")


def run_dl_training():
    set_seed(42)

    feature_df = clean_target(load_feature_dataset())

    print(f"\nDL training samples: {len(feature_df)}")

    X, y, dataset_names = split_features_and_target(feature_df)

    loo_predictions = evaluate_with_leave_one_out(X, y)

    metrics = calculate_metrics(y, loo_predictions)

    final_model, scaler = train_final_model(X, y)

    result_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=loo_predictions,
    )

    print("Deep Learning Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nPrediction Results:")
    print(result_df)

    save_outputs(
        model=final_model,
        scaler=scaler,
        result_df=result_df,
        metrics=metrics,
    )
