import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_optical_flow(prev_frame, next_frame):
    """
    Calculate dense optical flow between two consecutive frames.

    Args:
        prev_frame: Previous grayscale frame.
        next_frame: Next grayscale frame.

    Returns:
        magnitude: Motion strength for each pixel.
        angle: Motion direction for each pixel.
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame,
        next=next_frame,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    flow_x = flow[..., 0]  # pixel horizontal movement
    flow_y = flow[..., 1]  # pixel vertical movement

    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    return magnitude, angle


def extract_motion_features_from_frames(frames):
    """
    Extract statistical motion features from preprocessed frames.

    Args:
        frames: List of preprocessed grayscale frames.

    Returns:
        dict: Motion features extracted from the whole video.
    """

    all_magnitudes = []
    all_angles = []

    for i in range(len(frames) - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]

        magnitude, angle = calculate_optical_flow(prev_frame, next_frame)

        all_magnitudes.append(magnitude.flatten())
        all_angles.append(angle.flatten())

    all_magnitudes = np.concatenate(all_magnitudes)
    all_angles = np.concatenate(all_angles)

    features = {
        "mag_mean": np.mean(all_magnitudes),
        "mag_median": np.median(all_magnitudes),
        "mag_std": np.std(all_magnitudes),
        "mag_min": np.min(all_magnitudes),
        "mag_max": np.max(all_magnitudes),
        "mag_p25": np.percentile(all_magnitudes, 25),
        "mag_p75": np.percentile(all_magnitudes, 75),
        "mag_p90": np.percentile(all_magnitudes, 90),
        "angle_mean": np.mean(all_angles),
        "angle_median": np.median(all_angles),
        "angle_std": np.std(all_angles)
    }

    return features


def build_feature_dataset(overview_df, frame_extractor_function):
    """
    Build a machine learning dataset using optical flow features.

    Args:
        overview_df: Dataset overview dataframe.
        frame_extractor_function: Function that extracts preprocessed frames.

    Returns:
        pandas.DataFrame: Feature dataset.
    """

    rows = []

    for _, row in tqdm(
        overview_df.iterrows(),
        total=len(overview_df),
        desc="Extracting optical flow features"
    ):
        dataset_name = row["dataset_name"]
        real_speed = row["real_speed"]
        video_path = row["video_path"]

        if not row["video_exists"] or not row["is_readable"]:
            print(f"Skipping invalid video: {dataset_name}")
            continue

        frames = frame_extractor_function(
            video_path=video_path,
            dataset_name=dataset_name,
            max_frames=None
        )

        if len(frames) < 2:
            print(f"Skipping video with insufficient frames: {dataset_name}")
            continue

        features = extract_motion_features_from_frames(frames)

        features["dataset_name"] = dataset_name
        features["real_speed"] = real_speed

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    columns_order = [
        "dataset_name",
        "real_speed",
        "mag_mean",
        "mag_median",
        "mag_std",
        "mag_min",
        "mag_max",
        "mag_p25",
        "mag_p75",
        "mag_p90",
        "angle_mean",
        "angle_median",
        "angle_std"
    ]

    feature_df = feature_df[columns_order]

    return feature_df
