import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from src.config import RESULTS_DIR, MODELS_DIR


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """

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


def load_improved_features():
    """
    Load improved feature dataset.
    """

    feature_path = RESULTS_DIR / "phase6_improved_features.csv"

    return pd.read_csv(feature_path)


def split_data(df):
    """
    Split dataframe into X, y, and dataset names.
    """

    dataset_names = df["dataset_name"]

    y = df["real_speed"]

    X = df.drop(columns=["dataset_name", "real_speed"])

    return X, y, dataset_names


def create_models():
    """
    Create multiple regression models for comparison.
    """

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
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
    Evaluate all models using Leave-One-Out Cross Validation.
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

        metrics = calculate_metrics(y, predictions)

        results[model_name] = {
            "predictions": predictions,
            "metrics": metrics
        }

        print(f"\n{model_name} Metrics:")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    return results


def build_ensemble_prediction(results):
    """
    Build ensemble prediction from all model predictions.
    """

    prediction_list = []

    for model_result in results.values():
        prediction_list.append(model_result["predictions"])

    prediction_array = np.vstack(prediction_list)

    ensemble_prediction = np.mean(prediction_array, axis=0)

    return ensemble_prediction


def save_phase6_outputs(best_model, result_df, metrics_df):
    """
    Save phase 6 outputs.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "phase6_best_model.pkl"
    result_path = RESULTS_DIR / "phase6_final_prediction_results.xlsx"
    metrics_path = RESULTS_DIR / "phase6_all_metrics.csv"

    joblib.dump(best_model, model_path)

    result_df.to_excel(result_path, index=False)

    metrics_df.to_csv(metrics_path, index=False)

    print(f"Best model saved to: {model_path}")
    print(f"Final prediction table saved to: {result_path}")
    print(f"All metrics saved to: {metrics_path}")


def run_improved_training():
    """
    Run improved model training pipeline.
    """

    df = load_improved_features()

    df["real_speed"] = pd.to_numeric(df["real_speed"], errors="coerce")

    print("Rows before removing NaN real_speed:", len(df))
    print("NaN real_speed count:", df["real_speed"].isna().sum())

    df = df.dropna(subset=["real_speed"]).reset_index(drop=True)

    print("Rows after removing NaN real_speed:", len(df))

    X, y, dataset_names = split_data(df)
    selector = SelectKBest(
        score_func=f_regression,
        k=min(10, X.shape[1])
    )

    X = selector.fit_transform(X, y)

    selected_features = (
        df.drop(columns=["dataset_name", "real_speed"])
        .columns[selector.get_support()]
    )

    print("\nSelected Features:")
    for feature in selected_features:
        print(feature)

    models = create_models()

    results = evaluate_models(X, y, models)

    ensemble_prediction = build_ensemble_prediction(results)

    ensemble_metrics = calculate_metrics(y, ensemble_prediction)

    print("\nEnsemble Metrics:")

    for metric_name, metric_value in ensemble_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    final_result_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y,
        "مقدار پیشبینی شده": ensemble_prediction,
        "میزان خطا": np.abs(y - ensemble_prediction)
    })

    metrics_rows = []

    for model_name, model_result in results.items():
        row = {"model": model_name}
        row.update(model_result["metrics"])
        metrics_rows.append(row)

    ensemble_row = {"model": "Ensemble"}
    ensemble_row.update(ensemble_metrics)
    metrics_rows.append(ensemble_row)

    metrics_df = pd.DataFrame(metrics_rows)

    best_model_name = metrics_df.sort_values("MAE").iloc[0]["model"]

    if best_model_name == "Ensemble":
        best_model = models["ExtraTrees"]
    else:
        best_model = models[best_model_name]

    best_model.fit(X, y)

    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": selected_features,
            "importance": best_model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

        importance_df.to_csv(
            RESULTS_DIR / "phase6_feature_importance.csv",
            index=False
        )

        print("\nFeature Importance:")
        print(importance_df.head(10))

    save_phase6_outputs(
        best_model=best_model,
        result_df=final_result_df,
        metrics_df=metrics_df
    )
