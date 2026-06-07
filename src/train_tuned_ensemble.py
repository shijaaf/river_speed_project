import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import (
    MODELS_DIR,
    PHASE6_FEATURE_FILE,
    PHASE9_ADVANCED_FEATURE_FILE,
    PHASE13_DEEP_FEATURE_FILE,
    RESULTS_DIR,
)
from src.modeling_utils import (
    build_prediction_table,
    calculate_metrics,
    evaluate_models_loo,
    load_merged_feature_dataset,
    split_features_target,
    variance_then_select_k_best,
)


METRICS_FILE = RESULTS_DIR / "phase17_tuned_ensemble_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "phase17_tuned_ensemble_predictions.xlsx"
MODEL_FILE = MODELS_DIR / "phase17_tuned_ensemble.pkl"


def load_fused_dataset():
    return load_merged_feature_dataset([
        PHASE6_FEATURE_FILE,
        PHASE9_ADVANCED_FEATURE_FILE,
        PHASE13_DEEP_FEATURE_FILE,
    ])


def create_models():
    return {
        "ExtraTrees_A": ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=3,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features="sqrt",
            random_state=42,
        ),

        "ExtraTrees_B": ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=3,
            max_features=0.5,
            random_state=7,
        ),

        "ExtraTrees_C": ExtraTreesRegressor(
            n_estimators=1200,
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features=0.7,
            random_state=21,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=800,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.025,
            max_depth=2,
            subsample=0.8,
            random_state=42,
        ),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                C=5,
                epsilon=0.05,
                kernel="rbf",
            )),
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]),
    }


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
        weights=weights,
    )

    return ensemble_predictions, model_names, weights


def build_metrics_rows(results, ensemble_metrics):
    rows = []

    for model_name, result in results.items():
        row = {"model": model_name}
        row.update(result["metrics"])
        rows.append(row)

    ensemble_row = {"model": "WeightedEnsemble"}
    ensemble_row.update(ensemble_metrics)
    rows.append(ensemble_row)

    return rows


def save_outputs(metrics_rows, prediction_df, trained_package):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics_rows).to_csv(
        METRICS_FILE,
        index=False,
    )

    prediction_df.to_excel(
        PREDICTIONS_FILE,
        index=False,
    )

    joblib.dump(
        trained_package,
        MODEL_FILE,
    )

    print()
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Model package saved to: {MODEL_FILE}")


def main():
    df = load_fused_dataset()

    print(f"Training samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    X, y, dataset_names = split_features_target(df)

    (
        X_selected,
        variance_filter,
        selector,
        selected_features,
    ) = variance_then_select_k_best(
        X=X,
        y=y,
        k=20,
    )

    models = create_models()

    results = evaluate_models_loo(
        X=X_selected,
        y=y,
        models=models,
    )

    (
        ensemble_predictions,
        ensemble_model_names,
        ensemble_weights,
    ) = build_weighted_ensemble(results)

    ensemble_metrics = calculate_metrics(
        y_true=y,
        y_pred=ensemble_predictions,
    )

    print()
    print("WeightedEnsemble:")
    for metric_name, value in ensemble_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    metrics_rows = build_metrics_rows(
        results=results,
        ensemble_metrics=ensemble_metrics,
    )

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=ensemble_predictions,
    )

    final_models = create_models()

    for model in final_models.values():
        model.fit(X_selected, y)

    trained_package = {
        "models": final_models,
        "variance_filter": variance_filter,
        "selector": selector,
        "selected_features": list(selected_features),
        "ensemble_model_names": ensemble_model_names,
        "ensemble_weights": ensemble_weights,
    }

    save_outputs(
        metrics_rows=metrics_rows,
        prediction_df=prediction_df,
        trained_package=trained_package,
    )
