import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def load_dataset():

    df = pd.read_csv(
        "outputs/results/phase7_klt_tracking_features.csv"
    )

    df["real_speed"] = pd.to_numeric(
        df["real_speed"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["real_speed"]
    ).reset_index(drop=True)

    return df


def create_models():

    models = {

        "LinearRegression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),

        "Ridge":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),

        "SVR":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(
                    C=10,
                    epsilon=0.05,
                    kernel="rbf"
                ))
            ]),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42
            ),

        "ExtraTrees":
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=42
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                random_state=42
            )
    }

    return models


def main():

    df = load_dataset()

    dataset_names = df["dataset_name"]

    X = df.drop(
        columns=[
            "dataset_name",
            "real_speed"
        ]
    )

    y = df["real_speed"]

    loo = LeaveOneOut()

    metrics_rows = []

    best_mae = 999
    best_model_name = None
    best_predictions = None

    models = create_models()

    for model_name, model in models.items():

        predictions = cross_val_predict(
            estimator=model,
            X=X,
            y=y,
            cv=loo
        )

        metrics = calculate_metrics(
            y,
            predictions
        )

        row = {
            "model": model_name,
            **metrics
        }

        metrics_rows.append(row)

        print()
        print(model_name)
        print(metrics)

        if metrics["MAE"] < best_mae:

            best_mae = metrics["MAE"]

            best_model_name = model_name

            best_predictions = predictions

    print()
    print("BEST MODEL:", best_model_name)
    print("BEST MAE:", best_mae)

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        "outputs/results/phase8_klt_metrics.csv",
        index=False
    )

    prediction_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y,
        "مقدار پیشبینی شده": best_predictions,
        "میزان خطا":
            np.abs(
                y - best_predictions
            )
    })

    prediction_df.to_excel(
        "outputs/results/phase8_klt_predictions.xlsx",
        index=False
    )

    print(
        "Results saved successfully."
    )


if __name__ == "__main__":
    main()