import os
import joblib
import numpy as np
import pandas as pd

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


PHASE6_FILE = "outputs/results/phase6_improved_features.csv"
PHASE9_FILE = "outputs/results/phase9_advanced_motion_features.csv"
PHASE13_FILE = "outputs/results/phase13_deep_features.csv"

METRICS_FILE = "outputs/results/phase14_hybrid_deep_metrics.csv"
PREDICTIONS_FILE = "outputs/results/phase14_hybrid_deep_predictions.xlsx"
MODEL_FILE = "outputs/models/phase14_hybrid_deep_model.pkl"


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
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def load_fused_dataset():
    """
    Load and merge motion features and deep learning features.
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

    missing_rows = fused_df[
        fused_df["real_speed"].isna()
    ]

    if len(missing_rows) > 0:
        print("Samples removed because real_speed is missing:")
        print(missing_rows[["dataset_name", "real_speed"]])

    fused_df = fused_df.dropna(
        subset=["real_speed"]
    ).reset_index(drop=True)

    return fused_df


def split_data(df):
    """
    Split dataset into features, target, and names.
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


def select_features(X, y, k=20):
    """
    Select top K features to reduce overfitting risk.
    """

    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X.shape[1])
    )

    X_selected = selector.fit_transform(
        X,
        y
    )

    selected_features = X.columns[
        selector.get_support()
    ]

    print()
    print("Selected Features:")
    print("------------------")

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
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=700,
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
    Evaluate models using Leave-One-Out Cross Validation.
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

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return results


def build_metrics_table(results):
    """
    Convert metrics dictionary to dataframe.
    """

    rows = []

    for model_name, result in results.items():
        row = {
            "model": model_name
        }

        row.update(result["metrics"])

        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(results, metrics_df, dataset_names, y, best_model_name, best_model, selector, selected_features):
    """
    Save metrics, predictions, and final model.
    """

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    best_predictions = results[
        best_model_name
    ]["predictions"]

    prediction_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y,
        "مقدار پیشبینی شده": best_predictions,
        "میزان خطا": np.abs(y - best_predictions)
    })

    prediction_df.to_excel(
        PREDICTIONS_FILE,
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
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Model saved to: {MODEL_FILE}")


def main():
    """
    Run Phase 14 hybrid-deep model training.
    """

    fused_df = load_fused_dataset()

    print()
    print(f"Fused samples: {len(fused_df)}")
    print(f"Fused columns: {len(fused_df.columns)}")

    X, y, dataset_names = split_data(fused_df)

    X_selected, selected_features, selector = select_features(
        X=X,
        y=y,
        k=20
    )

    models = create_models()

    results = evaluate_models(
        X=X_selected,
        y=y,
        models=models
    )

    metrics_df = build_metrics_table(results)

    best_model_name = (
        metrics_df
        .sort_values("MAE")
        .iloc[0]["model"]
    )

    best_mae = (
        metrics_df
        .sort_values("MAE")
        .iloc[0]["MAE"]
    )

    print()
    print(f"Best model: {best_model_name}")
    print(f"Best MAE: {best_mae:.4f}")

    best_model = models[best_model_name]

    best_model.fit(
        X_selected,
        y
    )

    save_outputs(
        results=results,
        metrics_df=metrics_df,
        dataset_names=dataset_names,
        y=y,
        best_model_name=best_model_name,
        best_model=best_model,
        selector=selector,
        selected_features=selected_features
    )


if __name__ == "__main__":
    main()