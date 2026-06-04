import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PHASE6_FILE = "outputs/results/phase6_improved_features.csv"
PHASE9_FILE = "outputs/results/phase9_advanced_motion_features.csv"
PHASE13_FILE = "outputs/results/phase13_deep_features.csv"

METRICS_FILE = "outputs/results/phase17_tuned_ensemble_metrics.csv"
PREDICTIONS_FILE = "outputs/results/phase17_tuned_ensemble_predictions.xlsx"
MODEL_FILE = "outputs/models/phase17_tuned_ensemble.pkl"


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def load_fused_dataset():
    phase6_df = pd.read_csv(PHASE6_FILE)
    phase9_df = pd.read_csv(PHASE9_FILE)
    phase13_df = pd.read_csv(PHASE13_FILE)

    phase9_df = phase9_df.drop(columns=["real_speed"], errors="ignore")
    phase13_df = phase13_df.drop(columns=["real_speed"], errors="ignore")

    df = pd.merge(phase6_df, phase9_df, on="dataset_name", how="inner")
    df = pd.merge(df, phase13_df, on="dataset_name", how="inner")

    df["real_speed"] = pd.to_numeric(df["real_speed"], errors="coerce")
    df = df.dropna(subset=["real_speed"]).reset_index(drop=True)

    return df


def split_data(df):
    dataset_names = df["dataset_name"]

    y = df["real_speed"]

    X = df.drop(
        columns=[
            "dataset_name",
            "real_speed"
        ]
    )

    return X, y, dataset_names


def create_models():
    models = {
        "ExtraTrees_A": ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=3,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features="sqrt",
            random_state=42
        ),

        "ExtraTrees_B": ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=3,
            max_features=0.5,
            random_state=7
        ),

        "ExtraTrees_C": ExtraTreesRegressor(
            n_estimators=1200,
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features=0.7,
            random_state=21
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=800,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.025,
            max_depth=2,
            subsample=0.8,
            random_state=42
        ),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                C=5,
                epsilon=0.05,
                kernel="rbf"
            ))
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0))
        ])
    }

    return models


def prepare_features(X, y, k=20):
    variance_filter = VarianceThreshold(
        threshold=1e-8
    )

    X_var = variance_filter.fit_transform(X)

    remaining_features = X.columns[
        variance_filter.get_support()
    ]

    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X_var.shape[1])
    )

    X_selected = selector.fit_transform(
        X_var,
        y
    )

    selected_features = remaining_features[
        selector.get_support()
    ]

    print("Selected Features:")
    for feature in selected_features:
        print(feature)

    return X_selected, variance_filter, selector, selected_features


def evaluate_models(X, y, models):
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
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return results


def build_weighted_ensemble(results):
    model_names = []
    prediction_list = []
    weights = []

    for model_name, result in results.items():
        mae = result["metrics"]["MAE"]

        if mae <= 0:
            continue

        model_names.append(model_name)
        prediction_list.append(result["predictions"])

        weights.append(1.0 / mae)

    prediction_matrix = np.vstack(prediction_list)

    weights = np.array(weights)
    weights = weights / weights.sum()

    ensemble_predictions = np.average(
        prediction_matrix,
        axis=0,
        weights=weights
    )

    return ensemble_predictions, model_names, weights


def save_outputs(metrics_rows, prediction_df, trained_package):
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    prediction_df.to_excel(
        PREDICTIONS_FILE,
        index=False
    )

    joblib.dump(
        trained_package,
        MODEL_FILE
    )

    print()
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Model package saved to: {MODEL_FILE}")


def main():
    df = load_fused_dataset()

    print(f"Training samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    X, y, dataset_names = split_data(df)

    X_selected, variance_filter, selector, selected_features = prepare_features(
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

    ensemble_predictions, ensemble_model_names, ensemble_weights = build_weighted_ensemble(
        results
    )

    ensemble_metrics = calculate_metrics(
        y_true=y,
        y_pred=ensemble_predictions
    )

    print()
    print("WeightedEnsemble:")
    for metric_name, value in ensemble_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    metrics_rows = []

    for model_name, result in results.items():
        row = {
            "model": model_name
        }
        row.update(result["metrics"])
        metrics_rows.append(row)

    ensemble_row = {
        "model": "WeightedEnsemble"
    }
    ensemble_row.update(ensemble_metrics)
    metrics_rows.append(ensemble_row)

    prediction_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y,
        "مقدار پیشبینی شده": ensemble_predictions,
        "میزان خطا": np.abs(y - ensemble_predictions)
    })

    final_models = create_models()

    for model_name, model in final_models.items():
        model.fit(X_selected, y)

    trained_package = {
        "models": final_models,
        "variance_filter": variance_filter,
        "selector": selector,
        "selected_features": list(selected_features),
        "ensemble_model_names": ensemble_model_names,
        "ensemble_weights": ensemble_weights
    }

    save_outputs(
        metrics_rows=metrics_rows,
        prediction_df=prediction_df,
        trained_package=trained_package
    )


if __name__ == "__main__":
    main()
