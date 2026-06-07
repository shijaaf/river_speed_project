import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR, PHASE3_FEATURE_FILE, RESULTS_DIR
from src.modeling_utils import (
    build_prediction_table,
    calculate_metrics,
    clean_target,
    evaluate_model_loo,
    split_features_target,
)


def load_feature_dataset():
    return pd.read_csv(PHASE3_FEATURE_FILE)


def train_random_forest_model(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        min_samples_leaf=1,
    )

    model.fit(X, y)

    return model


def save_outputs(model, result_df, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "random_forest_model.pkl"
    result_path = RESULTS_DIR / "phase4_ml_prediction_results.xlsx"
    metrics_path = RESULTS_DIR / "phase4_ml_metrics.csv"

    joblib.dump(model, model_path)

    result_df.to_excel(result_path, index=False)

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Model saved to: {model_path}")
    print(f"Prediction table saved to: {result_path}")
    print(f"Metrics saved to: {metrics_path}")


def run_ml_training():
    feature_df = clean_target(load_feature_dataset())

    print(f"\nTraining samples: {len(feature_df)}")

    X, y, dataset_names = split_features_target(feature_df)

    evaluation_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        min_samples_leaf=1,
    )

    loo_predictions = evaluate_model_loo(
        model=evaluation_model,
        X=X,
        y=y,
    )

    metrics = calculate_metrics(y, loo_predictions)

    final_model = train_random_forest_model(X, y)

    result_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=loo_predictions,
    )

    print("Machine Learning Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nPrediction Results:")
    print(result_df)

    save_outputs(
        model=final_model,
        result_df=result_df,
        metrics=metrics,
    )
