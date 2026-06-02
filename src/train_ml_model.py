import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RESULTS_DIR, MODELS_DIR


def load_feature_dataset():
    """
    Load optical flow feature dataset from CSV file.

    Returns:
        pandas.DataFrame: Feature dataset.
    """

    feature_path = RESULTS_DIR / "phase3_optical_flow_features.csv"

    feature_df = pd.read_csv(feature_path)

    return feature_df


def split_features_and_target(feature_df):
    """
    Split dataframe into input features and target values.

    Args:
        feature_df: Dataframe containing dataset name, real speed, and features.

    Returns:
        X: Input feature matrix.
        y: Target speed values.
        dataset_names: Dataset names.
    """

    dataset_names = feature_df["dataset_name"]

    y = feature_df["real_speed"]

    X = feature_df.drop(columns=["dataset_name", "real_speed"])

    return X, y, dataset_names


def train_random_forest_model(X, y):
    """
    Train Random Forest regression model.

    Args:
        X: Input feature matrix.
        y: Target speed values.

    Returns:
        model: Trained Random Forest model.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        min_samples_leaf=1
    )

    model.fit(X, y)

    return model


def evaluate_with_leave_one_out(X, y):
    """
    Evaluate model using Leave-One-Out Cross Validation.

    Args:
        X: Input features.
        y: Target values.

    Returns:
        numpy.ndarray: Predicted values for each sample.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        min_samples_leaf=1
    )

    loo = LeaveOneOut()

    predictions = cross_val_predict(
        estimator=model,
        X=X,
        y=y,
        cv=loo
    )

    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Args:
        y_true: Real speed values.
        y_pred: Predicted speed values.

    Returns:
        dict: Evaluation metrics.
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


def build_prediction_table(dataset_names, y_true, y_pred):
    """
    Build final prediction table.

    Args:
        dataset_names: Dataset names.
        y_true: Real speed values.
        y_pred: Predicted speed values.

    Returns:
        pandas.DataFrame: Prediction result table.
    """

    result_df = pd.DataFrame({
        "نام دیتاست": dataset_names,
        "مقدار واقعی": y_true,
        "مقدار پیشبینی شده": y_pred,
        "میزان خطا": np.abs(y_true - y_pred)
    })

    return result_df


def save_outputs(model, result_df, metrics):
    """
    Save trained model, prediction table, and metrics.

    Args:
        model: Trained Random Forest model.
        result_df: Prediction result dataframe.
        metrics: Evaluation metric dictionary.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "random_forest_model.pkl"
    result_path = RESULTS_DIR / "phase4_ml_prediction_results.xlsx"
    metrics_path = RESULTS_DIR / "phase4_ml_metrics.csv"

    joblib.dump(model, model_path)

    result_df.to_excel(result_path, index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Model saved to: {model_path}")
    print(f"Prediction table saved to: {result_path}")
    print(f"Metrics saved to: {metrics_path}")


def run_ml_training():

    feature_df = load_feature_dataset()

    # Convert target to numeric
    feature_df["real_speed"] = pd.to_numeric(
        feature_df["real_speed"],
        errors="coerce"
    )

    # Show samples with missing labels
    missing_rows = feature_df[feature_df["real_speed"].isna()]

    if len(missing_rows) > 0:
        print("\nSamples removed because real_speed is missing:")
        print(missing_rows[["dataset_name", "real_speed"]])

    # Remove rows with missing target
    feature_df = feature_df.dropna(subset=["real_speed"])
    print(f"\nTraining samples: {len(feature_df)}")

    X, y, dataset_names = split_features_and_target(feature_df)

    loo_predictions = evaluate_with_leave_one_out(X, y)

    metrics = calculate_metrics(y, loo_predictions)

    final_model = train_random_forest_model(X, y)

    result_df = build_prediction_table(
        dataset_names=dataset_names,
        y_true=y,
        y_pred=loo_predictions
    )

    print("Machine Learning Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nPrediction Results:")
    print(result_df)

    save_outputs(
        model=final_model,
        result_df=result_df,
        metrics=metrics
    )