import time
import joblib
import pandas as pd

from src.config import MODELS_DIR, PHASE13_DEEP_FEATURE_FILE, RESULTS_DIR, LOGS_DIR
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


METRICS_FILE = RESULTS_DIR / "phase14_deep_feature_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "phase14_deep_feature_predictions.xlsx"
MODEL_FILE = MODELS_DIR / "phase14_deep_feature_model.pkl"
PREDICTION_TIME_FILE = LOGS_DIR / "phase14_prediction_time.txt"


def load_dataset():
    return clean_target(pd.read_csv(PHASE13_DEEP_FEATURE_FILE))


def calculate_average_prediction_time(model, X):
    """
    Calculate average prediction time per video/sample.
    """

    prediction_times = []

    for sample_index in range(len(X)):
        single_sample = X[sample_index:sample_index + 1]

        start_time = time.perf_counter()

        model.predict(single_sample)

        end_time = time.perf_counter()

        prediction_times.append(end_time - start_time)

    average_time = sum(prediction_times) / len(prediction_times)
    total_time = sum(prediction_times)

    return average_time, total_time, prediction_times


def save_prediction_time_report(
    average_time,
    total_time,
    sample_count,
):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PREDICTION_TIME_FILE, "w", encoding="utf-8") as file:
        file.write("Phase 14 Deep Feature Model Prediction Time Report\n")
        file.write("=" * 60)
        file.write("\n\n")

        file.write(f"Total samples/videos: {sample_count}\n")
        file.write(f"Total prediction time: {total_time:.6f} seconds\n")
        file.write(f"Average prediction time per video: {average_time:.6f} seconds\n")
        file.write(f"Average prediction time per video: {average_time * 1000:.6f} ms\n")

    print(f"Prediction time report saved to: {PREDICTION_TIME_FILE}")
    print(f"Average prediction time per video: {average_time:.6f} seconds")


def main():
    df = load_dataset()

    print(f"Deep feature samples: {len(df)}")
    print(f"Deep feature columns: {len(df.columns)}")

    X, y, dataset_names = split_features_target(df)

    X_selected, selected_features, selector = select_k_best_features(
        X=X,
        y=y,
        k=20,
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
    best_model = models[best_model_name]

    best_model.fit(X_selected, y)

    average_prediction_time, total_prediction_time, prediction_times = (
        calculate_average_prediction_time(
            model=best_model,
            X=X_selected,
        )
    )

    prediction_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=results[best_model_name]["predictions"],
    )

    prediction_df["زمان پیشبینی هر ویدیو - ثانیه"] = prediction_times

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_FILE, index=False)
    prediction_df.to_excel(PREDICTIONS_FILE, index=False)

    save_prediction_time_report(
        average_time=average_prediction_time,
        total_time=total_prediction_time,
        sample_count=len(X_selected),
    )

    joblib.dump(
        {
            "model": best_model,
            "selector": selector,
            "selected_features": list(selected_features),
            "average_prediction_time_seconds": average_prediction_time,
            "total_prediction_time_seconds": total_prediction_time,
        },
        MODEL_FILE,
    )

    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Model saved to: {MODEL_FILE}")

