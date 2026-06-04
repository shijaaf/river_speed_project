import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Phase 13 Deep Features
PHASE13_FILE = "outputs/results/phase13_deep_features.csv"

METRICS_FILE = "outputs/results/phase16_cnnlstm_metrics.csv"
PREDICTIONS_FILE = "outputs/results/phase16_cnnlstm_predictions.xlsx"
MODEL_FILE = "outputs/models/phase16_cnnlstm_model.pth"

SEED = 42
EPOCHS = 400
LEARNING_RATE = 0.001
SEQ_LEN = 5  # تعداد فریم در توالی


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNNLSTMRegressor(nn.Module):
    """
    Tiny CNN-LSTM regression model for river speed estimation.
    """

    def __init__(self, feature_dim, hidden_dim=32):
        super(CNNLSTMRegressor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, feature_dim]
        batch, seq_len, feature_dim = x.shape

        x = x.view(batch * seq_len, 1, feature_dim)
        x = self.cnn(x)
        x = torch.mean(x, dim=2)  # global average pooling
        x = x.view(batch, seq_len, -1)
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]
        out = self.regressor(hn)
        return out


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"model": "CNNLSTM", "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def load_sequences():
    """
    Convert Phase13 deep features into sequences for CNN-LSTM
    """

    df = pd.read_csv(PHASE13_FILE)
    df = df.dropna(subset=["real_speed"]).reset_index(drop=True)

    # Features
    feature_cols = [c for c in df.columns if c.startswith("deep_feature_")]

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["real_speed"].values.astype(np.float32)
    dataset_names = df["dataset_name"].values

    sequences = []
    for i in range(len(X_all)):
        seq = []
        for j in range(SEQ_LEN):
            idx = max(0, i - j)
            seq.append(X_all[idx])
        sequences.append(np.array(seq[::-1]))  # reverse for temporal order

    X_seq = np.stack(sequences)
    return X_seq, y_all, dataset_names


def train_one_fold(X_train, y_train):
    feature_dim = X_train.shape[2]
    model = CNNLSTMRegressor(feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
    return model


def evaluate_loo(X, y):
    loo = LeaveOneOut()
    predictions = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        model = train_one_fold(X_train, y_train)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            pred = model(X_test_tensor).item()
        predictions.append(pred)
    return np.array(predictions)


def main():
    set_seed()
    X_seq, y, dataset_names = load_sequences()
    loo_predictions = evaluate_loo(X_seq, y)
    metrics = calculate_metrics(y, loo_predictions)
    print("CNN-LSTM Metrics:")
    for k, v in metrics.items():
        if k == "model":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")

    # Train final model on full dataset
    feature_dim = X_seq.shape[2]
    final_model = CNNLSTMRegressor(feature_dim)
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    final_model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = final_model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

    # Save model and predictions
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    torch.save(final_model.state_dict(), MODEL_FILE)
    pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y,
        "مقدار پیشبینی شده": loo_predictions,
        "میزان خطا": np.abs(y - loo_predictions)
    }).to_excel(PREDICTIONS_FILE, index=False)

    pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()