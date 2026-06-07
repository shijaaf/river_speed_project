import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


EPSILON = 1e-8


def safe_fps(fps, default=1.0):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return default

    return fps if fps > 0 else default


def calculate_farneback_flow(
    prev_frame,
    next_frame,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
):
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame,
        next=next_frame,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        flow_x,
        flow_y,
    )

    return flow_x, flow_y, magnitude, angle


def calculate_entropy(values, bins=20):
    hist, _ = np.histogram(values, bins=bins)

    hist = hist.astype(float)
    hist = hist / (hist.sum() + EPSILON)

    return -np.sum(hist * np.log2(hist + EPSILON))


def summarize_array(values, prefix, percentiles=None):
    values = np.asarray(values).flatten()

    if percentiles is None:
        percentiles = [10, 25, 75, 90, 95]

    summary = {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
    }

    for percentile in percentiles:
        summary[f"{prefix}_p{percentile}"] = np.percentile(
            values,
            percentile,
        )

    return summary


def summarize_frame_feature_rows(frame_features):
    feature_df = pd.DataFrame(frame_features)

    final_features = {}

    for column in feature_df.columns:
        final_features[f"{column}_mean"] = feature_df[column].mean()
        final_features[f"{column}_median"] = feature_df[column].median()
        final_features[f"{column}_std"] = feature_df[column].std()
        final_features[f"{column}_min"] = feature_df[column].min()
        final_features[f"{column}_max"] = feature_df[column].max()

    return final_features


def iter_valid_videos(overview_df, desc):
    for _, row in tqdm(
        overview_df.iterrows(),
        total=len(overview_df),
        desc=desc,
    ):
        dataset_name = row["dataset_name"]

        if not row["video_exists"] or not row["is_readable"]:
            print(f"Skipping invalid video: {dataset_name}")
            continue

        yield row
