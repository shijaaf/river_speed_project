import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PHASE6_FILE = "outputs/results/phase6_improved_features.csv"
PHASE9_FILE = "outputs/results/phase9_advanced_motion_features.csv"
PHASE13_FILE = "outputs/results/phase13_deep_features.csv"

METRICS_FILE = "outputs/results/phase15_tinycnn_metrics.csv"
PREDICTIONS_FILE = "outputs/results/phase15_tinycnn_predictions.xlsx"
MODEL_FILE = "outputs/models/phase15_tinycnn_model.pth"

SEED = 42
SELECTED_FEATURE_COUNT = 40
EPOCHS = 600
LEARNING_RATE = 0.001


def set_seed(seed=SEED):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyCNNRegressor(nn.Module):
    """
    Tiny 1D CNN regression model for river speed estimation.
    """

    def __init__(self, input_length):
        super(TinyCNNRegressor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        Forward pass.
        """

        x = self.feature_extractor(x)

        x = x.view(x.size(0), -1)

        x = self.regressor(x)

        return x


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        "model": "TinyCNNRegressor",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def load_fused_dataset():
    """
    Load and merge Phase 6, Phase 9, and Phase 13 features.
    """

    phase6_df = pd.read_csv(PHASE6_FILE)
    phase9_df = pd.read_csv(PHASE9_FILE)
    phase13_df = pd.read_csv(PHASE13_FILE)

    phase9_df = phase9_df.drop(
        columns=["real_speed"],
        errors="ignore"
    )

    phase13_df = phase13_df.drop(
        columns=["real_speed"],
        errors="ignore"
    )

    fused_df = pd.merge(
        phase6_df,
        phase9_df,
        on="dataset_name",
        how="inner"
    )

    fused_df = pd.merge(
        fused_df,
        phase13_df,
        on="dataset_name",
        how="inner"
    )

    fused_df["real_speed"] = pd.to_numeric(
        fused_df["real_speed"],
        errors="coerce"
    )

    fused_df = fused_df.dropna(
        subset=["real_speed"]
    ).reset_index(drop=True)

    return fused_df


def split_data(df):
    """
    Split dataframe into features, target, and dataset names.
    """

    dataset_names = df["dataset_name"].values

    y = df["real_speed"].values.astype(np.float32)

    X = df.drop(
        columns=[
            "dataset_name",
            "real_speed"
        ]
    )

    return X, y, dataset_names


def prepare_fold_features(X_train, X_test, y_train):
    """
    Apply variance filtering, feature selection, and scaling inside one fold.
    """

    variance_filter = VarianceThreshold(
        threshold=1e-8
    )

    X_train_var = variance_filter.fit_transform(
        X_train
    )

    X_test_var = variance_filter.transform(
        X_test
    )

    selector = SelectKBest(
        score_func=f_regression,
        k=min(
            SELECTED_FEATURE_COUNT,
            X_train_var.shape[1]
        )
    )

    X_train_selected = selector.fit_transform(
        X_train_var,
        y_train
    )

    X_test_selected = selector.transform(
        X_test_var
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_selected
    )

    X_test_scaled = scaler.transform(
        X_test_selected
    )

    return X_train_scaled, X_test_scaled


def train_one_tinycnn(X_train, y_train):
    """
    Train one Tiny CNN model for one fold.
    """

    input_length = X_train.shape[1]

    model = TinyCNNRegressor(
        input_length=input_length
    )

    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.005
    )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    ).unsqueeze(1)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    ).view(-1, 1)

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        predictions = model(
            X_train_tensor
        )

        loss = criterion(
            predictions,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

    return model


def evaluate_leave_one_out(X, y):
    """
    Evaluate Tiny CNN using Leave-One-Out Cross Validation.
    """

    loo = LeaveOneOut()

    predictions = []

    for fold_index, (train_index, test_index) in enumerate(loo.split(X), start=1):

        print(f"Training fold {fold_index}/{len(y)}")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y[train_index]

        X_train_ready, X_test_ready = prepare_fold_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train
        )

        model = train_one_tinycnn(
            X_train=X_train_ready,
            y_train=y_train
        )

        model.eval()

        X_test_tensor = torch.tensor(
            X_test_ready,
            dtype=torch.float32
        ).unsqueeze(1)

        with torch.no_grad():
            predicted_speed = model(
                X_test_tensor
            ).item()

        predictions.append(predicted_speed)

    return np.array(predictions)


def train_final_model(X, y):
    """
    Train final Tiny CNN model on the full labeled dataset.
    """

    variance_filter = VarianceThreshold(
        threshold=1e-8
    )

    X_var = variance_filter.fit_transform(X)

    remaining_features = X.columns[
        variance_filter.get_support()
    ]

    selector = SelectKBest(
        score_func=f_regression,
        k=min(
            SELECTED_FEATURE_COUNT,
            X_var.shape[1]
        )
    )

    X_selected = selector.fit_transform(
        X_var,
        y
    )

    selected_features = remaining_features[
        selector.get_support()
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_selected
    )

    model = train_one_tinycnn(
        X_train=X_scaled,
        y_train=y
    )

    return model, variance_filter, selector, scaler, selected_features


def save_outputs(metrics, dataset_names, y_true, y_pred, model, variance_filter, selector, scaler, selected_features):
    """
    Save metrics, prediction table, and final trained model.
    """

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        METRICS_FILE,
        index=False
    )

    prediction_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y_true,
        "مقدار پیشبینی شده": y_pred,
        "میزان خطا": np.abs(y_true - y_pred)
    })

    prediction_df.to_excel(
        PREDICTIONS_FILE,
        index=False
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variance_filter": variance_filter,
            "selector": selector,
            "scaler": scaler,
            "selected_features": list(selected_features),
            "selected_feature_count": SELECTED_FEATURE_COUNT
        },
        MODEL_FILE
    )

    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Tiny CNN model saved to: {MODEL_FILE}")


def main():
    """
    Run Phase 15 Tiny CNN regression.
    """

    set_seed()

    df = load_fused_dataset()

    print(f"Training samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    X, y, dataset_names = split_data(df)

    loo_predictions = evaluate_leave_one_out(
        X=X,
        y=y
    )

    metrics = calculate_metrics(
        y_true=y,
        y_pred=loo_predictions
    )

    print()
    print("Tiny CNN Metrics:")
    for key, value in metrics.items():
        if key == "model":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    final_model, variance_filter, selector, scaler, selected_features = train_final_model(
        X=X,
        y=y
    )

    print()
    print("Selected features for final Tiny CNN:")
    for feature in selected_features:
        print(feature)

    save_outputs(
        metrics=metrics,
        dataset_names=dataset_names,
        y_true=y,
        y_pred=loo_predictions,
        model=final_model,
        variance_filter=variance_filter,
        selector=selector,
        scaler=scaler,
        selected_features=selected_features
    )


if __name__ == "__main__":
    main()
