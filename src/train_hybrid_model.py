import pandas as pd

from src.config import PHASE6_FEATURE_FILE, PHASE9_ADVANCED_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_metrics_table,
    build_prediction_table,
    create_standard_models,
    evaluate_models_loo,
    find_best_model_name,
    load_merged_feature_dataset,
    select_k_best_features,
    split_features_target,
)


METRICS_FILE = RESULTS_DIR / "phase11_hybrid_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "phase11_hybrid_predictions.xlsx"


def load_hybrid_dataset():
    return load_merged_feature_dataset([
        PHASE6_FEATURE_FILE,
        PHASE9_ADVANCED_FEATURE_FILE,
    ])


def main():
    df = load_hybrid_dataset()

    print()
    print(f"Hybrid samples: {len(df)}")
    print(f"Hybrid columns: {len(df.columns)}")

    X, y, dataset_names = split_features_target(df)

    X_selected, selected_features, selector = select_k_best_features(
        X=X,
        y=y,
        k=15,
    )

    models = create_standard_models(
        random_forest_n=500,
        extra_trees_n=700,
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

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=results[best_model_name]["predictions"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_FILE, index=False)
    prediction_df.to_excel(PREDICTIONS_FILE, index=False)

    print()
    print(f"Best Model: {best_model_name}")
    print("Results saved.")
