import numpy as np
import pandas as pd

from src.video_feature_utils import calculate_farneback_flow, iter_valid_videos


def calculate_optical_flow(prev_frame, next_frame):
    _, _, magnitude, angle = calculate_farneback_flow(
        prev_frame=prev_frame,
        next_frame=next_frame,
    )

    return magnitude, angle


def extract_motion_features_from_frames(frames):
    all_magnitudes = []
    all_angles = []

    for index in range(len(frames) - 1):
        magnitude, angle = calculate_optical_flow(
            prev_frame=frames[index],
            next_frame=frames[index + 1],
        )

        all_magnitudes.append(magnitude.flatten())
        all_angles.append(angle.flatten())

    all_magnitudes = np.concatenate(all_magnitudes)
    all_angles = np.concatenate(all_angles)

    return {
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
        "angle_std": np.std(all_angles),
    }


def build_feature_dataset(overview_df, frame_extractor_function):
    rows = []

    for row in iter_valid_videos(
        overview_df=overview_df,
        desc="Extracting optical flow features",
    ):
        dataset_name = row["dataset_name"]

        frames = frame_extractor_function(
            video_path=row["video_path"],
            dataset_name=dataset_name,
            max_frames=None,
        )

        if len(frames) < 2:
            print(f"Skipping video with insufficient frames: {dataset_name}")
            continue

        features = extract_motion_features_from_frames(frames)

        features["dataset_name"] = dataset_name
        features["real_speed"] = row["real_speed"]

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
        "angle_std",
    ]

    return feature_df[columns_order]
