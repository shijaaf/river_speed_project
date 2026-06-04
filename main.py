"""
Main entry point for the River Speed Estimation Project.
This file is used to test whether the project environment works correctly.
"""

import sys
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
# ---------phase1
from src.read_labels import read_labels
from src.video_info import build_dataset_overview
from src.config import RESULTS_DIR
# ---------phase2
from src.read_labels import read_labels
from src.video_info import build_dataset_overview
from src.preprocess import extract_preprocessed_frames, save_sample_frames
from src.config import RESULTS_DIR
# ---------phase3
from src.read_labels import read_labels
from src.video_info import build_dataset_overview
from src.preprocess import extract_preprocessed_frames
from src.optical_flow_features import build_feature_dataset
from src.config import RESULTS_DIR
# ---------phase4
from src.train_ml_model import run_ml_training
# ---------phase5
from src.train_dl_model import run_dl_training
# ---------phase6
from src.read_labels import read_labels
from src.video_info import build_dataset_overview
from src.preprocess import extract_preprocessed_frames
from src.improved_features import build_improved_feature_dataset
from src.train_improved_model import run_improved_training
from src.config import RESULTS_DIR


def test_libraries():
    # Print installed package versions to verify the environment setup.

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


def phase1_dataset_overview():
    """
    Run phase 1:
    1. Read labels
    2. Check video files
    3. Save dataset overview
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = read_labels()

    print("Labels loaded successfully.")
    print(labels_df)

    overview_df = build_dataset_overview(labels_df)

    print("\nDataset overview:")
    print(overview_df)

    # save CSV
    output_path = RESULTS_DIR / "phase1_dataset_overview.csv"
    overview_df.to_csv(output_path, index=False)

    # save TXT
    text_output = RESULTS_DIR / "phase1_dataset_overview.txt"

    with open(text_output, "w", encoding="utf-8") as f:
        f.write("Labels loaded successfully.\n\n")
        f.write(labels_df.to_string())

        f.write("\n\nDataset overview:\n\n")
        f.write(overview_df.to_string())

        f.write(f"\n\nDataset overview saved to: {output_path}\n")

    print(f"\nDataset overview saved to: {output_path}")
    print(f"Text report saved to: {text_output}")


def phase2_preprocess():
    """
    Run phase 2:
    1. Read labels
    2. Read dataset overview
    3. Extract preprocessed frames
    4. Save sample frames
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = read_labels()
    overview_df = build_dataset_overview(labels_df)

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
            max_frames=None
        )

        save_sample_frames(
            frames=frames,
            dataset_name=dataset_name,
            max_samples=5
        )

        print(f"{dataset_name}: {len(frames)} frames extracted.")


def phase3_optical_flow_features():
    """
    Run phase 3:
    1. Read labels
    2. Build dataset overview
    3. Extract optical flow features
    4. Save feature dataset
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = read_labels()

    overview_df = build_dataset_overview(labels_df)

    feature_df = build_feature_dataset(
        overview_df=overview_df,
        frame_extractor_function=extract_preprocessed_frames
    )

    print("Optical flow feature dataset:")
    print(feature_df)

    output_path = RESULTS_DIR / "phase3_optical_flow_features.csv"

    feature_df.to_csv(output_path, index=False)

    print(f"Feature dataset saved to: {output_path}")


def phase4_train_ml_model():
    """
    Run phase 4:
    Train and evaluate Machine Learning model.
    """

    run_ml_training()


def phase5_train_dl_model():
    """
    Run phase 5:
    Train and evaluate Deep Learning model.
    """

    run_dl_training()


def phase6_improve_training():
    """
        Run phase 6:
        1. Build improved features
        2. Save improved feature dataset
        3. Train improved ensemble models
        """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = read_labels()

    overview_df = build_dataset_overview(labels_df)

    improved_feature_df = build_improved_feature_dataset(
        overview_df=overview_df,
        frame_extractor_function=extract_preprocessed_frames
    )

    feature_path = RESULTS_DIR / "phase6_improved_features.csv"

    improved_feature_df.to_csv(feature_path, index=False)

    print(f"Improved features saved to: {feature_path}")

    run_improved_training()


if __name__ == "__main__":
    # test_libraries()
    # phase1_dataset_overview()
    # phase2_preprocess()
    # phase3_optical_flow_features()
    # phase4_train_ml_model()
    # phase5_train_dl_model()
    phase6_improve_training()
