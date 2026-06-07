import joblib
import pandas as pd

from src.config import MODELS_DIR, PHASE9_ADVANCED_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_metrics_table,
    build_prediction_table,
    clean_target,
    create_standard_models,
    evaluate_models_loo,
    find_best_model_name,
    select_k_best_features,
    split_features_target,
)


METRICS_FILE = RESULTS_DIR / "phase10_advanced_metrics.csv"
PREDICTION_FILE = RESULTS_DIR / "phase10_advanced_predictions.xlsx"
MODEL_FILE = MODELS_DIR / "phase10_best_advanced_model.pkl"


def load_dataset():
    return clean_target(pd.read_csv(PHASE9_ADVANCED_FEATURE_FILE))


def main():
    df = load_dataset()

    print()
    print(f"Training samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    X, y, dataset_names = split_features_target(df)

    X_selected, selected_features, selector = select_k_best_features(
        X=X,
        y=y,
        k=10,
    )

    models = create_standard_models(
        random_forest_n=300,
        extra_trees_n=500,
        gradient_n=300,
        max_depth=4,
        min_samples_leaf=2,
    )

    results = evaluate_models_loo(
        X=X_selected,
        y=y,
        models=models,
    )

    metrics_df = build_metrics_table(results)

    best_model_name = find_best_model_name(metrics_df)
    best_mae = metrics_df.sort_values("MAE").iloc[0]["MAE"]

    print()
    print(f"Best model: {best_model_name}")
    print(f"Best MAE: {best_mae:.4f}")

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=results[best_model_name]["predictions"],
    )

    best_model = models[best_model_name]

    best_model.fit(X_selected, y)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_FILE, index=False)
    prediction_df.to_excel(PREDICTION_FILE, index=False)

    joblib.dump(
        {
            "model": best_model,
            "selector": selector,
            "selected_features": list(selected_features),
        },
        MODEL_FILE,
    )

    print()
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTION_FILE}")
    print(f"Best model saved to: {MODEL_FILE}")

