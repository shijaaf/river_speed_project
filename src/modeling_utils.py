import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import torch
except Exception:
    torch = None


NAME_COLUMN = "dataset_name"
TARGET_COLUMN = "real_speed"
EPSILON = 1e-8


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_joblib(obj, path):
    ensure_parent_dir(path)
    joblib.dump(obj, path)


def load_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def clean_target(df, target_col=TARGET_COLUMN, verbose=True):
    df = df.copy()

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    missing_rows = df[df[target_col].isna()]

    if verbose and not missing_rows.empty:
        print("\nSamples removed because real_speed is missing:")

        shown_columns = [
            column
            for column in [NAME_COLUMN, target_col]
            if column in missing_rows.columns
        ]

        print(missing_rows[shown_columns])

    return df.dropna(subset=[target_col]).reset_index(drop=True)


def split_features_target(df, name_col=NAME_COLUMN, target_col=TARGET_COLUMN):
    dataset_names = df[name_col]
    y = df[target_col]
    X = df.drop(columns=[name_col, target_col])

    return X, y, dataset_names


def calculate_metrics(y_true, y_pred, include_mse=True):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    denominator = np.where(
        np.abs(y_true) < EPSILON,
        np.nan,
        y_true,
    )

    mape = np.nanmean(
        np.abs((y_true - y_pred) / denominator)
    ) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }

    if include_mse:
        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
        }

    return metrics


def build_prediction_table(dataset_names, y_true, y_pred):
    return pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y_true,
        "مقدار پیشبینی شده": y_pred,
        "میزان خطا": np.abs(
            np.asarray(y_true) - np.asarray(y_pred)
        ),
    })


def evaluate_model_loo(model, X, y):
    loo = LeaveOneOut()

    return cross_val_predict(
        estimator=model,
        X=X,
        y=y,
        cv=loo,
    )


def evaluate_models_loo(X, y, models):
    results = {}

    for model_name, model in models.items():
        predictions = evaluate_model_loo(
            model=model,
            X=X,
            y=y,
        )

        metrics = calculate_metrics(
            y_true=y,
            y_pred=predictions,
        )

        results[model_name] = {
            "predictions": predictions,
            "metrics": metrics,
        }

        print(f"\n{model_name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return results


def build_metrics_table(results):
    rows = []

    for model_name, result in results.items():
        row = {"model": model_name}
        row.update(result["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)


def find_best_model_name(metrics_df, metric="MAE"):
    return metrics_df.sort_values(metric).iloc[0]["model"]


def select_k_best_features(X, y, k):
    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X.shape[1]),
    )

    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    print("\nSelected Features:")
    print("------------------")
    for feature in selected_features:
        print(feature)

    return X_selected, selected_features, selector


def variance_then_select_k_best(X, y, k, threshold=1e-8):
    variance_filter = VarianceThreshold(threshold=threshold)

    X_var = variance_filter.fit_transform(X)

    remaining_features = X.columns[
        variance_filter.get_support()
    ]

    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X_var.shape[1]),
    )

    X_selected = selector.fit_transform(X_var, y)

    selected_features = remaining_features[
        selector.get_support()
    ]

    print("\nSelected Features:")
    print("------------------")
    for feature in selected_features:
        print(feature)

    return X_selected, variance_filter, selector, selected_features


def create_standard_models(
    include_linear=False,
    random_forest_n=300,
    extra_trees_n=500,
    gradient_n=300,
    max_depth=4,
    min_samples_leaf=2,
    ridge_alpha=1.0,
    svr_c=10,
    svr_epsilon=0.05,
):
    models = {}

    if include_linear:
        models["LinearRegression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    models.update({
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha)),
        ]),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                C=svr_c,
                epsilon=svr_epsilon,
                kernel="rbf",
            )),
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=random_forest_n,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=extra_trees_n,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=gradient_n,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
    })

    return models


def load_merged_feature_dataset(files, merge_on=NAME_COLUMN):
    merged_df = None

    for index, file_path in enumerate(files):
        df = load_csv(file_path)

        if index > 0:
            df = df.drop(
                columns=[TARGET_COLUMN],
                errors="ignore",
            )

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_on,
                how="inner",
            )

    if merged_df is None:
        raise ValueError("No dataset files were provided.")

    return clean_target(merged_df)


def save_metrics_and_predictions(
    metrics_df,
    prediction_df,
    metrics_file,
    prediction_file,
):
    ensure_parent_dir(metrics_file)
    ensure_parent_dir(prediction_file)

    metrics_df.to_csv(metrics_file, index=False)
    prediction_df.to_excel(prediction_file, index=False)

