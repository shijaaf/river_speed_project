from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.video_feature_utils import iter_valid_videos, safe_fps


VIDEO_START_SECONDS = {
    "1_Castor_Canada": 3.0,
    "2_Castor_Canada": 3.0,
    "3_Castor_Canada": 3.0,
    "AlpineStabilised": 3.0,
}

ROI_TOP_RATIO = 0.20
ROI_BOTTOM_RATIO = 0.85
ROI_LEFT_RATIO = 0.05
ROI_RIGHT_RATIO = 0.95

MAX_CORNERS = 300
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
MAX_FRAMES = 300


def apply_water_roi(frame):
    height, width = frame.shape

    top = int(height * ROI_TOP_RATIO)
    bottom = int(height * ROI_BOTTOM_RATIO)
    left = int(width * ROI_LEFT_RATIO)
    right = int(width * ROI_RIGHT_RATIO)

    return frame[top:bottom, left:right]


def save_roi_preview(frame, dataset_name):
    output_dir = Path("outputs/roi_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}.png"
    cv2.imwrite(str(output_path), frame)


def preprocess_tracking_frame(frame):
    resized_frame = cv2.resize(
        frame,
        (RESIZE_WIDTH, RESIZE_HEIGHT),
    )

    gray_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2GRAY,
    )

    enhanced_frame = cv2.equalizeHist(gray_frame)

    return apply_water_roi(enhanced_frame)


def read_tracking_frames(video_path, dataset_name, fps, max_frames=MAX_FRAMES):
    frames = []

    fps = safe_fps(fps)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    start_second = VIDEO_START_SECONDS.get(dataset_name, 0.0)
    start_frame = int(start_second * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(preprocess_tracking_frame(frame))

        if len(frames) >= max_frames:
            break

    cap.release()

    return frames


def detect_initial_points(first_frame):
    return cv2.goodFeaturesToTrack(
        image=first_frame,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
    )


def track_points_between_frames(prev_frame, next_frame, prev_points):
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prevImg=prev_frame,
        nextImg=next_frame,
        prevPts=prev_points,
        nextPts=None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )

    if next_points is None or status is None:
        return None, None

    status = status.reshape(-1)

    good_prev_points = prev_points[status == 1]
    good_next_points = next_points[status == 1]

    return good_prev_points, good_next_points


def calculate_displacement_features(prev_points, next_points, fps):
    displacement = next_points - prev_points

    dx = displacement[:, 0, 0]
    dy = displacement[:, 0, 1]

    distance_px = np.sqrt(dx ** 2 + dy ** 2)

    fps = safe_fps(fps)

    speed_px_per_sec = distance_px * fps
    angle = np.arctan2(dy, dx)

    return {
        "track_speed_mean": np.mean(speed_px_per_sec),
        "track_speed_median": np.median(speed_px_per_sec),
        "track_speed_std": np.std(speed_px_per_sec),
        "track_speed_p75": np.percentile(speed_px_per_sec, 75),
        "track_speed_p90": np.percentile(speed_px_per_sec, 90),
        "track_dx_mean": np.mean(dx),
        "track_dx_abs_mean": np.mean(np.abs(dx)),
        "track_dy_mean": np.mean(dy),
        "track_dy_abs_mean": np.mean(np.abs(dy)),
        "track_angle_mean": np.mean(angle),
        "valid_track_count": len(speed_px_per_sec),
    }


def extract_klt_features_from_video(video_path, fps):
    dataset_name = Path(video_path).stem

    frames = read_tracking_frames(
        video_path=video_path,
        dataset_name=dataset_name,
        fps=fps,
    )

    if len(frames) < 2:
        return None

    initial_points = detect_initial_points(frames[0])

    if initial_points is None or len(initial_points) < 5:
        return None

    all_feature_rows = []

    prev_frame = frames[0]
    prev_points = initial_points

    for index in range(1, len(frames)):
        good_prev_points, good_next_points = track_points_between_frames(
            prev_frame=prev_frame,
            next_frame=frames[index],
            prev_points=prev_points,
        )

        if good_prev_points is None or len(good_prev_points) < 5:
            break

        frame_features = calculate_displacement_features(
            prev_points=good_prev_points,
            next_points=good_next_points,
            fps=fps,
        )

        all_feature_rows.append(frame_features)

        prev_frame = frames[index]
        prev_points = good_next_points.reshape(-1, 1, 2)

    if len(all_feature_rows) == 0:
        return None

    feature_df = pd.DataFrame(all_feature_rows)

    final_features = {}

    for column in feature_df.columns:
        final_features[f"{column}_mean"] = feature_df[column].mean()
        final_features[f"{column}_median"] = feature_df[column].median()
        final_features[f"{column}_std"] = feature_df[column].std()

    final_features["tracking_frame_count"] = len(feature_df)

    return final_features


def build_klt_feature_dataset(overview_df):
    rows = []

    for row in iter_valid_videos(
        overview_df=overview_df,
        desc="Extracting KLT tracking features",
    ):
        dataset_name = row["dataset_name"]

        features = extract_klt_features_from_video(
            video_path=row["video_path"],
            fps=row["fps"],
        )

        if features is None:
            print(f"No valid tracking features extracted: {dataset_name}")
            continue

        features["dataset_name"] = dataset_name
        features["real_speed"] = row["real_speed"]

        rows.append(features)

    return pd.DataFrame(rows)
