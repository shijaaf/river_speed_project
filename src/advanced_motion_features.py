import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_entropy(values, bins=20):
    """
    Calculate entropy of numeric values.
    """

    hist, _ = np.histogram(
        values,
        bins=bins
    )

    hist = hist.astype(float)

    hist = hist / (hist.sum() + 1e-8)

    entropy = -np.sum(
        hist * np.log2(hist + 1e-8)
    )

    return entropy


def calculate_flow(prev_frame, next_frame):
    """
    Calculate dense optical flow between two grayscale frames.
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

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    return magnitude, angle


def extract_motion_features(frames):
    """
    Extract advanced dense motion features from preprocessed frames.
    """

    frame_features = []

    for i in range(len(frames) - 1):

        magnitude, angle = calculate_flow(
            frames[i],
            frames[i + 1]
        )

        active_threshold = np.percentile(
            magnitude,
            80
        )

        high_threshold = np.percentile(
            magnitude,
            95
        )

        active_mask = magnitude > active_threshold

        active_values = magnitude[active_mask]

        if len(active_values) == 0:
            continue

        active_motion_ratio = active_mask.mean()

        high_motion_ratio = (
            magnitude > high_threshold
        ).mean()

        motion_energy = np.mean(
            magnitude ** 2
        )

        active_motion_energy = np.mean(
            active_values ** 2
        )

        direction_consistency = (
            np.cos(angle[active_mask]).mean() ** 2
            +
            np.sin(angle[active_mask]).mean() ** 2
        )

        motion_entropy = calculate_entropy(
            magnitude.flatten()
        )

        active_motion_entropy = calculate_entropy(
            active_values.flatten()
        )

        frame_features.append({
            "mag_mean": np.mean(magnitude),
            "mag_median": np.median(magnitude),
            "mag_std": np.std(magnitude),
            "mag_p75": np.percentile(magnitude, 75),
            "mag_p90": np.percentile(magnitude, 90),
            "mag_p95": np.percentile(magnitude, 95),

            "active_mag_mean": np.mean(active_values),
            "active_mag_median": np.median(active_values),
            "active_mag_std": np.std(active_values),

            "active_motion_ratio": active_motion_ratio,
            "high_motion_ratio": high_motion_ratio,

            "motion_energy": motion_energy,
            "active_motion_energy": active_motion_energy,

            "direction_consistency": direction_consistency,

            "motion_entropy": motion_entropy,
            "active_motion_entropy": active_motion_entropy
        })

    if len(frame_features) == 0:
        return None

    feature_df = pd.DataFrame(frame_features)

    final_features = {}

    for column in feature_df.columns:

        final_features[f"{column}_mean"] = feature_df[column].mean()
        final_features[f"{column}_median"] = feature_df[column].median()
        final_features[f"{column}_std"] = feature_df[column].std()
        final_features[f"{column}_min"] = feature_df[column].min()
        final_features[f"{column}_max"] = feature_df[column].max()

    final_features["processed_flow_pairs"] = len(feature_df)

    return final_features


def build_advanced_feature_dataset(overview_df, frame_extractor_function):
    """
    Build advanced dense motion feature dataset for all videos.
    """

    rows = []

    for _, row in tqdm(
        overview_df.iterrows(),
        total=len(overview_df),
        desc="Extracting advanced motion features"
    ):
        dataset_name = row["dataset_name"]

        if not row["video_exists"] or not row["is_readable"]:
            print(f"Skipping invalid video: {dataset_name}")
            continue

        frames = frame_extractor_function(
            video_path=row["video_path"],
            dataset_name=dataset_name,
            max_frames=None
        )

        if len(frames) < 2:
            print(f"Skipping video with insufficient frames: {dataset_name}")
            continue

        features = extract_motion_features(
            frames=frames
        )

        if features is None:
            print(f"No valid motion features extracted: {dataset_name}")
            continue

        features["dataset_name"] = dataset_name
        features["real_speed"] = row["real_speed"]

        rows.append(features)

    return pd.DataFrame(rows)