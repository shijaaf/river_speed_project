import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import LeaveOneOut

from src.config import MODELS_DIR, PHASE13_DEEP_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_prediction_table,
    calculate_metrics,
    clean_target,
    set_seed,
)


METRICS_FILE = RESULTS_DIR / "phase16_cnnlstm_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "phase16_cnnlstm_predictions.xlsx"
MODEL_FILE = MODELS_DIR / "phase16_cnnlstm_model.pth"

SEED = 42
EPOCHS = 400
LEARNING_RATE = 0.001
SEQ_LEN = 5


class CNNLSTMRegressor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        batch, seq_len, feature_dim = x.shape

        x = x.view(batch * seq_len, 1, feature_dim)

        x = self.cnn(x)

        x = torch.mean(x, dim=2)

        x = x.view(batch, seq_len, -1)

        _, (hn, _) = self.lstm(x)

        hn = hn[-1]

        return self.regressor(hn)


def load_sequences():
    df = clean_target(pd.read_csv(PHASE13_DEEP_FEATURE_FILE))

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("deep_feature_")
    ]

    X_all = df[feature_columns].values.astype(np.float32)
    y_all = df["real_speed"].values.astype(np.float32)
    dataset_names = df["dataset_name"].values

    sequences = []

    for index in range(len(X_all)):
        sequence = []

        for step in range(SEQ_LEN):
            source_index = max(0, index - step)
            sequence.append(X_all[source_index])

        sequences.append(np.array(sequence[::-1]))

    X_seq = np.stack(sequences)

    return X_seq, y_all, dataset_names


def train_one_fold(X_train, y_train):
    feature_dim = X_train.shape[2]

    model = CNNLSTMRegressor(feature_dim)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.SmoothL1Loss()

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).view(-1, 1)

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        output = model(X_train_tensor)

        loss = criterion(
            output,
            y_train_tensor,
        )

        loss.backward()
        optimizer.step()

    return model


def evaluate_loo(X, y):
    loo = LeaveOneOut()
    predictions = []

    for train_index, test_index in loo.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]

        model = train_one_fold(
            X_train=X_train,
            y_train=y_train,
        )

        model.eval()

        with torch.no_grad():
            X_test_tensor = torch.tensor(
                X_test,
                dtype=torch.float32,
            )

            prediction = model(X_test_tensor).item()

        predictions.append(prediction)

    return np.array(predictions)


def train_final_model(X, y):
    feature_dim = X.shape[2]

    model = CNNLSTMRegressor(feature_dim)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.SmoothL1Loss()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    ).view(-1, 1)

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        output = model(X_tensor)

        loss = criterion(
            output,
            y_tensor,
        )

        loss.backward()
        optimizer.step()

    return model


def main():
    set_seed(SEED)

    X_seq, y, dataset_names = load_sequences()

    loo_predictions = evaluate_loo(
        X=X_seq,
        y=y,
    )

    metrics = calculate_metrics(
        y_true=y,
        y_pred=loo_predictions,
        include_mse=False,
    )

    metrics["model"] = "CNNLSTM"

    print("CNN-LSTM Metrics:")
    for key, value in metrics.items():
        if key == "model":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    final_model = train_final_model(
        X=X_seq,
        y=y,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        final_model.state_dict(),
        MODEL_FILE,
    )

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=loo_predictions,
    )

    prediction_df.to_excel(
        PREDICTIONS_FILE,
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
