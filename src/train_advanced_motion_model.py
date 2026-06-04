import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
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


FEATURE_FILE = "outputs/results/phase9_advanced_motion_features.csv"
METRICS_FILE = "outputs/results/phase10_advanced_metrics.csv"
PREDICTION_FILE = "outputs/results/phase10_advanced_predictions.xlsx"
MODEL_FILE = "outputs/models/phase10_best_advanced_model.pkl"


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """

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
    """
    Load advanced motion feature dataset.
    """

    df = pd.read_csv(FEATURE_FILE)

    df["real_speed"] = pd.to_numeric(
        df["real_speed"],
        errors="coerce"
    )

    missing_rows = df[df["real_speed"].isna()]

    if len(missing_rows) > 0:
        print("Samples removed because real_speed is missing:")
        print(missing_rows[["dataset_name", "real_speed"]])

    df = df.dropna(
        subset=["real_speed"]
    ).reset_index(drop=True)

    return df


def split_features_and_target(df):
    """
    Split dataframe into input features, target values, and dataset names.
    """

    dataset_names = df["dataset_name"]

    y = df["real_speed"]

    X = df.drop(
        columns=[
            "dataset_name",
            "real_speed"
        ]
    )

    return X, y, dataset_names


def select_best_features(X, y, k=10):
    """
    Select top K features using univariate regression score.
    """

    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X.shape[1])
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    print()
    print("Selected Features:")
    for feature in selected_features:
        print(feature)

    return X_selected, selected_features, selector


def create_models():
    """
    Create regression models for comparison.
    """

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                C=10,
                epsilon=0.05,
                kernel="rbf"
            ))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=42
        )
    }

    return models


def evaluate_models(X, y, models):
    """
    Evaluate all models using Leave-One-Out Cross Validation.
    """

    loo = LeaveOneOut()

    results = {}

    for model_name, model in models.items():

        predictions = cross_val_predict(
            estimator=model,
            X=X,
            y=y,
            cv=loo
        )

        metrics = calculate_metrics(
            y_true=y,
            y_pred=predictions
        )

        results[model_name] = {
            "predictions": predictions,
            "metrics": metrics
        }

        print()
        print(f"{model_name} Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    return results


def find_best_model(results):
    """
    Find best model based on MAE.
    """

    best_model_name = None
    best_mae = float("inf")

    for model_name, result in results.items():
        mae = result["metrics"]["MAE"]

        if mae < best_mae:
            best_mae = mae
            best_model_name = model_name

    return best_model_name, best_mae


def build_metrics_table(results):
    """
    Build metrics dataframe.
    """

    rows = []

    for model_name, result in results.items():
        row = {
            "model": model_name
        }

        row.update(result["metrics"])

        rows.append(row)

    return pd.DataFrame(rows)


def build_prediction_table(dataset_names, y_true, y_pred):
    """
    Build final prediction table.
    """

    result_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y_true,
        "مقدار پیشبینی شده": y_pred,
        "میزان خطا": np.abs(y_true - y_pred)
    })

    return result_df


def save_outputs(metrics_df, prediction_df, best_model, selector, selected_features):
    """
    Save metrics, predictions, and final trained model.
    """

    import os

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    prediction_df.to_excel(
        PREDICTION_FILE,
        index=False
    )

    joblib.dump(
        {
            "model": best_model,
            "selector": selector,
            "selected_features": list(selected_features)
        },
        MODEL_FILE
    )

    print()
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTION_FILE}")
    print(f"Best model saved to: {MODEL_FILE}")


def main():
    """
    Run Phase 10 advanced motion model training.
    """

    df = load_dataset()

    print()
    print(f"Training samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    X, y, dataset_names = split_features_and_target(df)

    X_selected, selected_features, selector = select_best_features(
        X=X,
        y=y,
        k=10
    )

    models = create_models()

    results = evaluate_models(
        X=X_selected,
        y=y,
        models=models
    )

    best_model_name, best_mae = find_best_model(results)

    print()
    print(f"Best model: {best_model_name}")
    print(f"Best MAE: {best_mae:.4f}")

    best_predictions = results[best_model_name]["predictions"]

    metrics_df = build_metrics_table(results)

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=best_predictions
    )

    best_model = models[best_model_name]

    best_model.fit(
        X_selected,
        y
    )

    save_outputs(
        metrics_df=metrics_df,
        prediction_df=prediction_df,
        best_model=best_model,
        selector=selector,
        selected_features=selected_features
    )


if __name__ == "__main__":
    main()