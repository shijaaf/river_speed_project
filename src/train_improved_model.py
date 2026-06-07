import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression

from src.config import MODELS_DIR, PHASE6_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_metrics_table,
    build_prediction_table,
    calculate_metrics,
    clean_target,
    evaluate_models_loo,
    split_features_target,
)


MODEL_FILE = MODELS_DIR / "phase6_best_model.pkl"
PREDICTION_FILE = RESULTS_DIR / "phase6_final_prediction_results.xlsx"
METRICS_FILE = RESULTS_DIR / "phase6_all_metrics.csv"
IMPORTANCE_FILE = RESULTS_DIR / "phase6_feature_importance.csv"


def load_improved_features():
    return pd.read_csv(PHASE6_FEATURE_FILE)


def create_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
    }


def build_ensemble_prediction(results):
    prediction_array = np.vstack([
        model_result["predictions"]
        for model_result in results.values()
    ])

    return np.mean(prediction_array, axis=0)


def save_feature_importance(model, selected_features):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(IMPORTANCE_FILE, index=False)

    print("\nFeature Importance:")
    print(importance_df.head(10))


def run_improved_training():
    df = clean_target(load_improved_features())

    print("Rows after removing NaN real_speed:", len(df))

    X, y, dataset_names = split_features_target(df)

    selector = SelectKBest(
        score_func=f_regression,
        k=min(10, X.shape[1]),
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    models = create_models()

    results = evaluate_models_loo(
        X=X_selected,
        y=y,
        models=models,
    )

    ensemble_prediction = build_ensemble_prediction(results)

    ensemble_metrics = calculate_metrics(
        y_true=y,
        y_pred=ensemble_prediction,
    )

    print("\nEnsemble Metrics:")
    for metric_name, metric_value in ensemble_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    metrics_df = build_metrics_table(results)

    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame([
                {
                    "model": "Ensemble",
                    **ensemble_metrics,
                }
            ]),
        ],
        ignore_index=True,
    )

    result_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=ensemble_prediction,
    )

    best_model_name = metrics_df.sort_values("MAE").iloc[0]["model"]

    if best_model_name == "Ensemble":
        best_model = models["ExtraTrees"]
    else:
        best_model = models[best_model_name]

    best_model.fit(X_selected, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_FILE)

    result_df.to_excel(PREDICTION_FILE, index=False)
    metrics_df.to_csv(METRICS_FILE, index=False)

    save_feature_importance(best_model, selected_features)

    print(f"Best model saved to: {MODEL_FILE}")
    print(f"Final prediction table saved to: {PREDICTION_FILE}")
    print(f"All metrics saved to: {METRICS_FILE}")
