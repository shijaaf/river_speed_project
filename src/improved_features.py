import numpy as np
import pandas as pd

from src.video_feature_utils import (
    calculate_farneback_flow,
    iter_valid_videos,
    safe_fps,
)


def calculate_optical_flow_components(prev_frame, next_frame):
    return calculate_farneback_flow(
        prev_frame=prev_frame,
        next_frame=next_frame,
        levels=4,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
    )


def extract_name_features(dataset_name):
    name_lower = dataset_name.lower()

    return {
        "is_stabilised": int(
            "stabilised" in name_lower
            and "notstabilised" not in name_lower
        ),
        "is_not_stabilised": int(
            "notstabilised" in name_lower
            or "nonstabilised" in name_lower
        ),
        "is_seeded": int(
            "seeded" in name_lower
            and "unseeded" not in name_lower
        ),
        "is_unseeded": int("unseeded" in name_lower),
        "is_uas": int("uas" in name_lower),
        "is_gopro": int("gopro" in name_lower),
        "is_orthorectified": int("orthorectified" in name_lower),
        "has_fps_in_name": int("fps" in name_lower),
    }


def extract_improved_motion_features(frames, fps):
    all_flow_x = []
    all_flow_y = []
    all_magnitudes = []
    all_angles = []
    moving_ratios = []

    for index in range(len(frames) - 1):
        flow_x, flow_y, magnitude, angle = calculate_optical_flow_components(
            prev_frame=frames[index],
            next_frame=frames[index + 1],
        )

        threshold = np.percentile(magnitude, 75)
        moving_area_ratio = np.mean(magnitude > threshold)

        all_flow_x.append(flow_x.flatten())
        all_flow_y.append(flow_y.flatten())
        all_magnitudes.append(magnitude.flatten())
        all_angles.append(angle.flatten())
        moving_ratios.append(moving_area_ratio)

    all_flow_x = np.concatenate(all_flow_x)
    all_flow_y = np.concatenate(all_flow_y)
    all_magnitudes = np.concatenate(all_magnitudes)
    all_angles = np.concatenate(all_angles)

    fps = safe_fps(fps)

    return {
        "flow_x_mean": np.mean(all_flow_x),
        "flow_x_median": np.median(all_flow_x),
        "flow_x_std": np.std(all_flow_x),
        "flow_x_abs_mean": np.mean(np.abs(all_flow_x)),

        "flow_y_mean": np.mean(all_flow_y),
        "flow_y_median": np.median(all_flow_y),
        "flow_y_std": np.std(all_flow_y),
        "flow_y_abs_mean": np.mean(np.abs(all_flow_y)),

        "mag_mean": np.mean(all_magnitudes),
        "mag_median": np.median(all_magnitudes),
        "mag_std": np.std(all_magnitudes),
        "mag_min": np.min(all_magnitudes),
        "mag_max": np.max(all_magnitudes),
        "mag_p10": np.percentile(all_magnitudes, 10),
        "mag_p25": np.percentile(all_magnitudes, 25),
        "mag_p75": np.percentile(all_magnitudes, 75),
        "mag_p90": np.percentile(all_magnitudes, 90),
        "mag_p95": np.percentile(all_magnitudes, 95),

        "angle_mean": np.mean(all_angles),
        "angle_median": np.median(all_angles),
        "angle_std": np.std(all_angles),

        "moving_area_ratio_mean": np.mean(moving_ratios),
        "moving_area_ratio_std": np.std(moving_ratios),

        "motion_per_second_mean": np.mean(all_magnitudes) * fps,
        "motion_per_second_p90": np.percentile(all_magnitudes, 90) * fps,
    }


def build_improved_feature_dataset(overview_df, frame_extractor_function):
    rows = []

    for row in iter_valid_videos(
        overview_df=overview_df,
        desc="Extracting improved features",
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

        motion_features = extract_improved_motion_features(
            frames=frames,
            fps=row["fps"],
        )

        name_features = extract_name_features(dataset_name)

        rows.append({
            "dataset_name": dataset_name,
            "real_speed": row["real_speed"],
            "fps": row["fps"],
            "frame_count": row["frame_count"],
            "duration_sec": row["duration_sec"],
            "width": row["width"],
            "height": row["height"],
            **motion_features,
            **name_features,
        })

    return pd.DataFrame(rows)
