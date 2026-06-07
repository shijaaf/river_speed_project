import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression

from src.config import (
    MODELS_DIR,
    PHASE6_FEATURE_FILE,
    PHASE9_ADVANCED_FEATURE_FILE,
    RESULTS_DIR,
)
from src.modeling_utils import load_merged_feature_dataset, split_features_target


PHASE11_PREDICTIONS_FILE = RESULTS_DIR / "phase11_hybrid_predictions.xlsx"
PHASE11_METRICS_FILE = RESULTS_DIR / "phase11_hybrid_metrics.csv"

FINAL_MODEL_FILE = MODELS_DIR / "final_hybrid_extratrees_model.pkl"
FINAL_TABLE_FILE = RESULTS_DIR / "final_prediction_table.xlsx"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "final_feature_importance.csv"
FINAL_REPORT_FILE = RESULTS_DIR / "final_project_summary.txt"

ERROR_PLOT_FILE = RESULTS_DIR / "final_error_plot.png"
TRUE_VS_PRED_FILE = RESULTS_DIR / "final_true_vs_predicted.png"


def load_hybrid_dataset():
    return load_merged_feature_dataset([
        PHASE6_FEATURE_FILE,
        PHASE9_ADVANCED_FEATURE_FILE,
    ])


def train_final_model(df):
    X, y, dataset_names = split_features_target(df)

    selector = SelectKBest(
        score_func=f_regression,
        k=min(15, X.shape[1]),
    )

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    model = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )

    model.fit(X_selected, y)

    return model, selector, selected_features


def save_final_model(model, selector, selected_features):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "selector": selector,
            "selected_features": list(selected_features),
        },
        FINAL_MODEL_FILE,
    )

    print(f"Final model saved to: {FINAL_MODEL_FILE}")


def save_feature_importance(model, selected_features):
    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_FILE}")

    return importance_df


def save_final_prediction_table():
    prediction_df = pd.read_excel(PHASE11_PREDICTIONS_FILE)

    prediction_df.to_excel(
        FINAL_TABLE_FILE,
        index=False,
    )

    print(f"Final prediction table saved to: {FINAL_TABLE_FILE}")

    return prediction_df


def plot_error(prediction_df):
    plt.figure(figsize=(12, 6))

    plt.bar(
        prediction_df["نام دیتاست"],
        prediction_df["میزان خطا"],
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
    y_true = prediction_df["مقدار واقعی"]
    y_pred = prediction_df["مقدار پیشبینی شده"]

    plt.figure(figsize=(7, 7))

    plt.scatter(y_true, y_pred)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
    )

    plt.xlabel("Real Speed (m/s)")
    plt.ylabel("Predicted Speed (m/s)")
    plt.title("Real Speed vs Predicted Speed")

    plt.tight_layout()

    plt.savefig(TRUE_VS_PRED_FILE, dpi=300)
    plt.close()

    print(f"True vs predicted plot saved to: {TRUE_VS_PRED_FILE}")


def save_summary_report(metrics_df, prediction_df, importance_df):
    best_row = metrics_df.sort_values("MAE").iloc[0]

    mean_error = prediction_df["میزان خطا"].mean()
    max_error = prediction_df["میزان خطا"].max()

    worst_sample = prediction_df.sort_values(
        "میزان خطا",
        ascending=False,
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

        if "RMSE" in best_row:
            file.write(f"RMSE: {best_row['RMSE']:.4f}\n")

        if "R2" in best_row:
            file.write(f"R2: {best_row['R2']:.4f}\n")

        if "MAPE" in best_row:
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_df = load_hybrid_dataset()

    model, selector, selected_features = train_final_model(hybrid_df)

    save_final_model(
        model=model,
        selector=selector,
        selected_features=selected_features,
    )

    importance_df = save_feature_importance(
        model=model,
        selected_features=selected_features,
    )

    prediction_df = save_final_prediction_table()

    metrics_df = pd.read_csv(PHASE11_METRICS_FILE)

    plot_error(prediction_df)

    plot_true_vs_predicted(prediction_df)

    save_summary_report(
        metrics_df=metrics_df,
        prediction_df=prediction_df,
        importance_df=importance_df,
    )

    print("Phase 12 completed successfully.")


if __name__ == "__main__":
    main()
