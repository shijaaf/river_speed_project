import pandas as pd

from src.modeling_utils import (
    build_metrics_table,
    build_prediction_table,
    clean_target,
    create_standard_models,
    evaluate_models_loo,
    find_best_model_name,
    split_features_target,
)


FEATURE_FILE = "outputs/results/phase7_klt_tracking_features.csv"
METRICS_FILE = "outputs/results/phase8_klt_metrics.csv"
PREDICTION_FILE = "outputs/results/phase8_klt_predictions.xlsx"


def load_dataset():
    return clean_target(pd.read_csv(FEATURE_FILE))


def main():
    df = load_dataset()

    X, y, dataset_names = split_features_target(df)

    models = create_standard_models(
        include_linear=True,
        random_forest_n=300,
        extra_trees_n=500,
        gradient_n=300,
        max_depth=None,
        min_samples_leaf=1,
    )

    results = evaluate_models_loo(
        X=X,
        y=y,
        models=models,
    )

    metrics_df = build_metrics_table(results)

    best_model_name = find_best_model_name(metrics_df)

    best_mae = metrics_df.sort_values("MAE").iloc[0]["MAE"]

    print()
    print("BEST MODEL:", best_model_name)
    print("BEST MAE:", best_mae)

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=results[best_model_name]["predictions"],
    )

    metrics_df.to_csv(METRICS_FILE, index=False)
    prediction_df.to_excel(PREDICTION_FILE, index=False)

    print("Results saved successfully.")
