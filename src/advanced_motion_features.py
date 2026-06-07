import numpy as np
import pandas as pd

from src.video_feature_utils import (
    calculate_entropy,
    calculate_farneback_flow,
    iter_valid_videos,
    summarize_frame_feature_rows,
)


ACTIVE_PERCENTILE = 80
HIGH_PERCENTILE = 95


def calculate_flow(prev_frame, next_frame):
    _, _, magnitude, angle = calculate_farneback_flow(
        prev_frame=prev_frame,
        next_frame=next_frame,
    )

    return magnitude, angle


def calculate_direction_consistency(angle, active_mask):
    active_angles = angle[active_mask]

    if len(active_angles) == 0:
        return 0.0

    cos_mean = np.cos(active_angles).mean()
    sin_mean = np.sin(active_angles).mean()

    return cos_mean ** 2 + sin_mean ** 2


def extract_single_pair_features(magnitude, angle):
    active_threshold = np.percentile(magnitude, ACTIVE_PERCENTILE)
    high_threshold = np.percentile(magnitude, HIGH_PERCENTILE)

    active_mask = magnitude > active_threshold
    active_values = magnitude[active_mask]

    if len(active_values) == 0:
        return None

    return {
        "mag_mean": np.mean(magnitude),
        "mag_median": np.median(magnitude),
        "mag_std": np.std(magnitude),
        "mag_p75": np.percentile(magnitude, 75),
        "mag_p90": np.percentile(magnitude, 90),
        "mag_p95": np.percentile(magnitude, 95),

        "active_mag_mean": np.mean(active_values),
        "active_mag_median": np.median(active_values),
        "active_mag_std": np.std(active_values),

        "active_motion_ratio": active_mask.mean(),
        "high_motion_ratio": (magnitude > high_threshold).mean(),

        "motion_energy": np.mean(magnitude ** 2),
        "active_motion_energy": np.mean(active_values ** 2),

        "direction_consistency": calculate_direction_consistency(
            angle=angle,
            active_mask=active_mask,
        ),

        "motion_entropy": calculate_entropy(magnitude.flatten()),
        "active_motion_entropy": calculate_entropy(active_values.flatten()),
    }


def extract_motion_features(frames):
    if frames is None or len(frames) < 2:
        return None

    frame_features = []

    for index in range(len(frames) - 1):
        magnitude, angle = calculate_flow(
            prev_frame=frames[index],
            next_frame=frames[index + 1],
        )

        features = extract_single_pair_features(
            magnitude=magnitude,
            angle=angle,
        )

        if features is not None:
            frame_features.append(features)

    if len(frame_features) == 0:
        return None

    final_features = summarize_frame_feature_rows(frame_features)
    final_features["processed_flow_pairs"] = len(frame_features)

    return final_features


def build_advanced_feature_dataset(overview_df, frame_extractor_function):
    rows = []

    for row in iter_valid_videos(
        overview_df=overview_df,
        desc="Extracting advanced motion features",
    ):
        dataset_name = row["dataset_name"]

        frames = frame_extractor_function(
            video_path=row["video_path"],
            dataset_name=dataset_name,
            max_frames=None,
        )

        if frames is None or len(frames) < 2:
            print(f"Skipping video with insufficient frames: {dataset_name}")
            continue

        features = extract_motion_features(frames)

        if features is None:
            print(f"No valid motion features extracted: {dataset_name}")
            continue

        features["dataset_name"] = dataset_name
        features["real_speed"] = row["real_speed"]

        rows.append(features)

    return pd.DataFrame(rows)
