"""
Main entry point for the River Speed Estimation Project.
This file is used to test whether the project environment works correctly.
"""

import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch

# --------- phase1
from src.read_labels import read_labels
from src.video_info import build_dataset_overview
from src.config import RESULTS_DIR , LOGS_DIR

# --------- phase2
from src.preprocess import extract_preprocessed_frames, save_sample_frames

# --------- phase3
from src.optical_flow_features import build_feature_dataset

# --------- phase4
from src.train_ml_model import run_ml_training

# --------- phase5
from src.train_dl_model import run_dl_training

# --------- phase6
from src.improved_features import build_improved_feature_dataset
from src.train_improved_model import run_improved_training

# ==========================
# new pipeline - phase7
from src.tracking_features import build_klt_feature_dataset

# --------- phase8
from src.train_klt_models import main as run_phase8_klt

# --------- phase9
from src.advanced_motion_features import build_advanced_feature_dataset

# --------- phase10
from src.train_advanced_motion_model import main as run_phase10_advanced

# --------- phase11
from src.train_hybrid_model import main as run_phase11_hybrid

# --------- deep learning method - phase13
from src.deep_feature_extraction import build_deep_feature_dataset

# --------- phase14
from src.train_deep_feature_model import main as run_phase14_deep

# --------- phase15
from src.train_hybrid_deep_tinycnn import main as run_phase15_tinycnn

# --------- phase16
from src.train_hybrid_deep_cnnlstm import main as run_phase16_cnnlstm

# --------- phase17
from src.train_tuned_ensemble import main as run_phase17_ensemble


TIME_LOG_FILE = LOGS_DIR / "processing_time_log.txt"


def log_processing_time(phase_name, start_time, end_time):
    """
    Save processing time of one phase in one line.
    Each execution appends a new line to the text file.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_line = (
        f"{timestamp} | "
        f"{phase_name} | "
        f"{elapsed_seconds:.2f} seconds | "
        f"{elapsed_minutes:.2f} minutes\n"
    )

    with open(TIME_LOG_FILE, "a", encoding="utf-8") as file:
        file.write(log_line)

    print(f"Processing time saved to: {TIME_LOG_FILE}")
    print(f"{phase_name} time: {elapsed_seconds:.2f} seconds")


def run_phase_with_timer(phase_name, phase_function):
    """
    Run a phase and record its processing time.
    """

    start_time = time.perf_counter()

    try:
        phase_function()
    finally:
        end_time = time.perf_counter()
        log_processing_time(
            phase_name=phase_name,
            start_time=start_time,
            end_time=end_time,
        )


def test_libraries():
    """
    Print installed package versions to verify the environment setup.
    """

    print("River Speed Estimation Project")
    print("--------------------------------")

    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. CPU will be used.")


def load_overview():
    labels_df = read_labels()
    overview_df = build_dataset_overview(labels_df)

    return labels_df, overview_df


def phase1_dataset_overview():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df, overview_df = load_overview()

    print("Labels loaded successfully.")
    print(labels_df)

    print("\nDataset overview:")
    print(overview_df)

    output_path = RESULTS_DIR / "phase1_dataset_overview.csv"
    overview_df.to_csv(output_path, index=False)

    text_output = RESULTS_DIR / "phase1_dataset_overview.txt"

    with open(text_output, "w", encoding="utf-8") as file:
        file.write("Labels loaded successfully.\n\n")
        file.write(labels_df.to_string())

        file.write("\n\nDataset overview:\n\n")
        file.write(overview_df.to_string())

        file.write(f"\n\nDataset overview saved to: {output_path}\n")

    print(f"\nDataset overview saved to: {output_path}")
    print(f"Text report saved to: {text_output}")


def phase2_preprocess():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    for _, row in overview_df.iterrows():
        dataset_name = row["dataset_name"]
        video_path = row["video_path"]

        if not row["video_exists"]:
            print(f"Video not found: {dataset_name}")
            continue

        if not row["is_readable"]:
            print(f"Video is not readable: {dataset_name}")
            continue

        frames = extract_preprocessed_frames(
            video_path=video_path,
            dataset_name=dataset_name,
            max_frames=None,
        )

        save_sample_frames(
            frames=frames,
            dataset_name=dataset_name,
            max_samples=5,
        )

        print(f"{dataset_name}: {len(frames)} frames extracted.")


def phase3_optical_flow_features():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    feature_df = build_feature_dataset(
        overview_df=overview_df,
        frame_extractor_function=extract_preprocessed_frames,
    )

    print("Optical flow feature dataset:")
    print(feature_df)

    output_path = RESULTS_DIR / "phase3_optical_flow_features.csv"

    feature_df.to_csv(output_path, index=False)

    print(f"Feature dataset saved to: {output_path}")


def phase4_train_ml_model():
    run_ml_training()


def phase5_train_dl_model():
    run_dl_training()


def phase6_improve_training():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    improved_feature_df = build_improved_feature_dataset(
        overview_df=overview_df,
        frame_extractor_function=extract_preprocessed_frames,
    )

    feature_path = RESULTS_DIR / "phase6_improved_features.csv"

    improved_feature_df.to_csv(feature_path, index=False)

    print(f"Improved features saved to: {feature_path}")

    run_improved_training()


def phase7_klt_tracking_features():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    klt_feature_df = build_klt_feature_dataset(overview_df)

    output_path = RESULTS_DIR / "phase7_klt_tracking_features.csv"

    klt_feature_df.to_csv(output_path, index=False)

    print("KLT tracking feature dataset:")
    print(klt_feature_df)

    print(f"KLT tracking features saved to: {output_path}")


def phase8_train_klt_models():
    run_phase8_klt()


def phase9_advanced_motion_features():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    advanced_feature_df = build_advanced_feature_dataset(
        overview_df=overview_df,
        frame_extractor_function=extract_preprocessed_frames,
    )

    output_path = RESULTS_DIR / "phase9_advanced_motion_features.csv"

    advanced_feature_df.to_csv(output_path, index=False)

    print("Advanced motion feature dataset:")
    print(advanced_feature_df)

    print(f"Advanced motion features saved to: {output_path}")


def phase10_train_advanced_motion_model():
    run_phase10_advanced()


def phase11_train_hybrid_model():
    run_phase11_hybrid()


def phase13_deep_feature_extraction():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, overview_df = load_overview()

    deep_feature_df = build_deep_feature_dataset(
        overview_df=overview_df,
    )

    output_path = RESULTS_DIR / "phase13_deep_features.csv"

    deep_feature_df.to_csv(
        output_path,
        index=False,
    )

    print("Deep feature dataset:")
    print(deep_feature_df)

    print(f"Deep features saved to: {output_path}")


def phase14_train_deep_feature_model():
    run_phase14_deep()


def phase15_train_tinycnn_model():
    run_phase15_tinycnn()


def phase16_train_cnnlstm_model():
    run_phase16_cnnlstm()


def phase17_train_tuned_ensemble_model():
    run_phase17_ensemble()


if __name__ == "__main__":
    run_phase_with_timer("test_libraries", test_libraries)

    run_phase_with_timer("phase1_dataset_overview", phase1_dataset_overview)
    run_phase_with_timer("phase2_preprocess", phase2_preprocess)
    run_phase_with_timer("phase3_optical_flow_features", phase3_optical_flow_features)

    run_phase_with_timer("phase4_train_ml_model", phase4_train_ml_model)
    run_phase_with_timer("phase5_train_dl_model", phase5_train_dl_model)

    run_phase_with_timer("phase6_improve_training", phase6_improve_training)

    run_phase_with_timer("phase7_klt_tracking_features", phase7_klt_tracking_features)
    run_phase_with_timer("phase8_train_klt_models", phase8_train_klt_models)

    run_phase_with_timer("phase9_advanced_motion_features", phase9_advanced_motion_features)
    run_phase_with_timer("phase10_train_advanced_motion_model", phase10_train_advanced_motion_model)

    run_phase_with_timer("phase11_train_hybrid_model", phase11_train_hybrid_model)

    run_phase_with_timer("phase13_deep_feature_extraction", phase13_deep_feature_extraction)
    run_phase_with_timer("phase14_train_deep_feature_model", phase14_train_deep_feature_model)

    run_phase_with_timer("phase15_train_tinycnn_model", phase15_train_tinycnn_model)

    run_phase_with_timer("phase16_train_cnnlstm_model", phase16_train_cnnlstm_model)

    run_phase_with_timer("phase17_train_tuned_ensemble_model", phase17_train_tuned_ensemble_model)
