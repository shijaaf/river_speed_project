import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor


PHASE6_FILE = "outputs/results/phase6_improved_features.csv"
PHASE9_FILE = "outputs/results/phase9_advanced_motion_features.csv"
PHASE11_PREDICTIONS_FILE = "outputs/results/phase11_hybrid_predictions.xlsx"
PHASE11_METRICS_FILE = "outputs/results/phase11_hybrid_metrics.csv"

FINAL_MODEL_FILE = "outputs/models/final_hybrid_extratrees_model.pkl"
FINAL_TABLE_FILE = "outputs/results/final_prediction_table.xlsx"
FEATURE_IMPORTANCE_FILE = "outputs/results/final_feature_importance.csv"
FINAL_REPORT_FILE = "outputs/results/final_project_summary.txt"

ERROR_PLOT_FILE = "outputs/results/final_error_plot.png"
TRUE_VS_PRED_FILE = "outputs/results/final_true_vs_predicted.png"


def load_hybrid_dataset():
    """
    Load and merge Phase 6 and Phase 9 feature datasets.
    """

    phase6_df = pd.read_csv(PHASE6_FILE)

    phase9_df = pd.read_csv(PHASE9_FILE)

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
    ).reset_index(drop=True)

    return hybrid_df


def train_final_model(df):
    """
    Train final ExtraTrees model on the full labeled dataset.
    """

    dataset_names = df["dataset_name"]

    y = df["real_speed"]

    X = df.drop(
        columns=[
            "dataset_name",
            "real_speed"
        ]
    )

    selector = SelectKBest(
        score_func=f_regression,
        k=min(15, X.shape[1])
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    model = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_selected, y)

    return model, selector, selected_features, X_selected, y, dataset_names


def save_final_model(model, selector, selected_features):
    """
    Save final trained model and feature selector.
    """

    os.makedirs("outputs/models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "selector": selector,
            "selected_features": list(selected_features)
        },
        FINAL_MODEL_FILE
    )

    print(f"Final model saved to: {FINAL_MODEL_FILE}")


def save_feature_importance(model, selected_features):
    """
    Save feature importance table.
    """

    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_FILE}")

    return importance_df


def save_final_prediction_table():
    """
    Copy Phase 11 best prediction table as final project output table.
    """

    prediction_df = pd.read_excel(PHASE11_PREDICTIONS_FILE)

    prediction_df.to_excel(
        FINAL_TABLE_FILE,
        index=False
    )

    print(f"Final prediction table saved to: {FINAL_TABLE_FILE}")

    return prediction_df


def plot_error(prediction_df):
    """
    Plot absolute prediction error for each dataset.
    """

    plt.figure(figsize=(12, 6))

    plt.bar(
        prediction_df["نام دیتاست"],
        prediction_df["میزان خطا"]
    )

    plt.xticks(rotation=90)

    plt.xlabel("Dataset Name")
    plt.ylabel("Absolute Error (m/s)")
    plt.title("Final Prediction Error per Dataset")

    plt.tight_layout()

    plt.savefig(ERROR_PLOT_FILE, dpi=300)

    plt.close()

    print(f"Error plot saved to: {ERROR_PLOT_FILE}")


def plot_true_vs_predicted(prediction_df):
    """
    Plot true speed values against predicted speed values.
    """

    y_true = prediction_df["مقدار واقعی"]
    y_pred = prediction_df["مقدار پیشبینی شده"]

    plt.figure(figsize=(7, 7))

    plt.scatter(y_true, y_pred)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Real Speed (m/s)")
    plt.ylabel("Predicted Speed (m/s)")
    plt.title("Real Speed vs Predicted Speed")

    plt.tight_layout()

    plt.savefig(TRUE_VS_PRED_FILE, dpi=300)

    plt.close()

    print(f"True vs predicted plot saved to: {TRUE_VS_PRED_FILE}")


def save_summary_report(metrics_df, prediction_df, importance_df):
    """
    Save a short Persian final summary report.
    """

    best_row = metrics_df.sort_values("MAE").iloc[0]

    mean_error = prediction_df["میزان خطا"].mean()
    max_error = prediction_df["میزان خطا"].max()

    worst_sample = prediction_df.sort_values(
        "میزان خطا",
        ascending=False
    ).iloc[0]

    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("گزارش خلاصه نهایی پروژه تخمین سرعت جریان آب رودخانه\n")
        file.write("=" * 60)
        file.write("\n\n")

        file.write("روش نهایی انتخاب‌شده:\n")
        file.write("Hybrid Feature Fusion + ExtraTrees Regressor\n\n")

        file.write("دلیل انتخاب روش:\n")
        file.write(
            "در این پروژه چند روش شامل Optical Flow ساده، KLT Tracking، "
            "ویژگی‌های پیشرفته حرکتی و ترکیب ویژگی‌ها بررسی شد. "
            "بهترین نتیجه با ترکیب ویژگی‌های Phase 6 و Phase 9 به دست آمد.\n\n"
        )

        file.write("بهترین مدل:\n")
        file.write(f"{best_row['model']}\n\n")

        file.write("معیارهای ارزیابی:\n")
        file.write(f"MAE: {best_row['MAE']:.4f}\n")
        file.write(f"RMSE: {best_row['RMSE']:.4f}\n")
        file.write(f"R2: {best_row['R2']:.4f}\n")
        file.write(f"MAPE: {best_row['MAPE']:.4f}\n\n")

        file.write("تحلیل خطا:\n")
        file.write(f"میانگین خطا: {mean_error:.4f} m/s\n")
        file.write(f"بیشترین خطا: {max_error:.4f} m/s\n")
        file.write(f"بدترین نمونه: {worst_sample['نام دیتاست']}\n\n")

        file.write("مهم‌ترین ویژگی‌ها:\n")
        for _, row in importance_df.head(10).iterrows():
            file.write(
                f"{row['feature']}: {row['importance']:.4f}\n"
            )

        file.write("\nمشخصات سیستم مورد استفاده:\n")
        file.write("CPU: Intel Core i7-8750H\n")
        file.write("GPU: NVIDIA GeForce GTX 1050 4GB\n")
        file.write("RAM: 8GB\n")
        file.write("OS: Windows 11\n")
        file.write("Python: 3.12\n")

    print(f"Final summary report saved to: {FINAL_REPORT_FILE}")


def main():
    """
    Run Phase 12 finalization pipeline.
    """

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    hybrid_df = load_hybrid_dataset()

    model, selector, selected_features, X_selected, y, dataset_names = train_final_model(
        hybrid_df
    )

    save_final_model(
        model=model,
        selector=selector,
        selected_features=selected_features
    )

    importance_df = save_feature_importance(
        model=model,
        selected_features=selected_features
    )

    prediction_df = save_final_prediction_table()

    metrics_df = pd.read_csv(PHASE11_METRICS_FILE)

    plot_error(prediction_df)

    plot_true_vs_predicted(prediction_df)

    save_summary_report(
        metrics_df=metrics_df,
        prediction_df=prediction_df,
        importance_df=importance_df
    )

    print("Phase 12 completed successfully.")


if __name__ == "__main__":
    main()
