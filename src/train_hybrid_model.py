import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict

from sklearn.feature_selection import (
    SelectKBest,
    f_regression
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PHASE6_FILE = "outputs/results/phase6_improved_features.csv"
PHASE9_FILE = "outputs/results/phase9_advanced_motion_features.csv"


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def load_hybrid_dataset():

    phase6_df = pd.read_csv(
        PHASE6_FILE
    )

    phase9_df = pd.read_csv(
        PHASE9_FILE
    )

    phase9_df = phase9_df.drop(
        columns=["real_speed"],
        errors="ignore"
    )

    hybrid_df = pd.merge(
        phase6_df,
        phase9_df,
        on="dataset_name",
        how="inner"
    )

    hybrid_df["real_speed"] = pd.to_numeric(
        hybrid_df["real_speed"],
        errors="coerce"
    )

    hybrid_df = hybrid_df.dropna(
        subset=["real_speed"]
    )

    hybrid_df = hybrid_df.reset_index(
        drop=True
    )

    return hybrid_df


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


def select_features(X, y):

    selector = SelectKBest(
        score_func=f_regression,
        k=min(15, X.shape[1])
    )

    X_selected = selector.fit_transform(
        X,
        y
    )

    selected_features = (
        X.columns[
            selector.get_support()
        ]
    )

    print()
    print("Selected Features:")
    print("------------------")

    for feature in selected_features:
        print(feature)

    return (
        X_selected,
        selected_features,
        selector
    )


def create_models():

    return {

        "Ridge":
            Pipeline([
                ("scaler",
                 StandardScaler()),

                ("model",
                 Ridge())
            ]),

        "SVR":
            Pipeline([
                ("scaler",
                 StandardScaler()),

                ("model",
                 SVR(
                     C=10,
                     epsilon=0.05
                 ))
            ]),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42
            ),

        "ExtraTrees":
            ExtraTreesRegressor(
                n_estimators=700,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                random_state=42
            )
    }


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
            y,
            predictions
        )

        results[model_name] = {
            "predictions": predictions,
            "metrics": metrics
        }

        print()
        print(model_name)

        for metric_name, value in metrics.items():
            print(
                f"{metric_name}: "
                f"{value:.4f}"
            )

    return results


def build_metrics_table(results):

    rows = []

    for model_name, result in results.items():

        row = {
            "model": model_name
        }

        row.update(
            result["metrics"]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    df = load_hybrid_dataset()

    print()
    print(
        f"Hybrid samples: "
        f"{len(df)}"
    )

    print(
        f"Hybrid columns: "
        f"{len(df.columns)}"
    )

    X, y, dataset_names = split_data(df)

    (
        X_selected,
        selected_features,
        selector
    ) = select_features(X, y)

    models = create_models()

    results = evaluate_models(
        X,
        y,
        models
    )

    metrics_df = build_metrics_table(
        results
    )

    metrics_df.to_csv(
        "outputs/results/phase11_hybrid_metrics.csv",
        index=False
    )

    best_model_name = (
        metrics_df
        .sort_values("MAE")
        .iloc[0]["model"]
    )

    best_predictions = (
        results[
            best_model_name
        ]["predictions"]
    )

    prediction_df = pd.DataFrame({
        "نام دیتاست":
            dataset_names,

        "مقدار واقعی":
            y,

        "مقدار پیشبینی شده":
            best_predictions,

        "میزان خطا":
            np.abs(
                y -
                best_predictions
            )
    })

    prediction_df.to_excel(
        "outputs/results/phase11_hybrid_predictions.xlsx",
        index=False
    )

    print()
    print(
        f"Best Model: "
        f"{best_model_name}"
    )

    print(
        "Results saved."
    )


if __name__ == "__main__":
    main()