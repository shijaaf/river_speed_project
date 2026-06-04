import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


MAX_CORNERS = 300
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01
)

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
FRAME_STEP = 1
MAX_FRAMES = 300


def preprocess_tracking_frame(frame):
    """
    Convert input frame to a grayscale resized frame for feature tracking.
    """

    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    enhanced_frame = cv2.equalizeHist(gray_frame)

    return enhanced_frame


def read_tracking_frames(video_path, max_frames=MAX_FRAMES):
    """
    Read video frames for tracking.
    """

    frames = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % FRAME_STEP == 0:
            processed_frame = preprocess_tracking_frame(frame)
            frames.append(processed_frame)

        frame_index += 1

        if len(frames) >= max_frames:
            break

    cap.release()

    return frames


def detect_initial_points(first_frame):
    """
    Detect strong trackable points in the first frame.
    """

    points = cv2.goodFeaturesToTrack(
        image=first_frame,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE
    )

    return points


def track_points_between_frames(prev_frame, next_frame, prev_points):
    """
    Track points from previous frame to next frame using Lucas-Kanade optical flow.
    """

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prevImg=prev_frame,
        nextImg=next_frame,
        prevPts=prev_points,
        nextPts=None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )

    if next_points is None or status is None:
        return None, None

    status = status.reshape(-1)

    good_prev_points = prev_points[status == 1]

    good_next_points = next_points[status == 1]

    return good_prev_points, good_next_points


def calculate_displacement_features(prev_points, next_points, fps):
    """
    Calculate displacement-based features between two tracked point sets.
    """

    displacement = next_points - prev_points

    dx = displacement[:, 0, 0]
    dy = displacement[:, 0, 1]

    distance_px = np.sqrt(dx ** 2 + dy ** 2)

    fps = fps if fps and fps > 0 else 1.0

    speed_px_per_sec = distance_px * fps

    angle = np.arctan2(dy, dx)

    features = {
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
        "valid_track_count": len(speed_px_per_sec)
    }

    return features


def extract_klt_features_from_video(video_path, fps):
    """
    Extract KLT tracking features from one video.
    """

    frames = read_tracking_frames(video_path)

    if len(frames) < 2:
        return None

    initial_points = detect_initial_points(frames[0])

    if initial_points is None or len(initial_points) < 5:
        return None

    all_feature_rows = []

    prev_frame = frames[0]
    prev_points = initial_points

    for i in range(1, len(frames)):
        next_frame = frames[i]

        good_prev_points, good_next_points = track_points_between_frames(
            prev_frame=prev_frame,
            next_frame=next_frame,
            prev_points=prev_points
        )

        if good_prev_points is None:
            break

        if len(good_prev_points) < 5:
            break

        frame_features = calculate_displacement_features(
            prev_points=good_prev_points,
            next_points=good_next_points,
            fps=fps
        )

        all_feature_rows.append(frame_features)

        prev_frame = next_frame
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
    """
    Build a dataset of KLT tracking features for all videos.
    """

    rows = []

    for _, row in tqdm(
        overview_df.iterrows(),
        total=len(overview_df),
        desc="Extracting KLT tracking features"
    ):
        dataset_name = row["dataset_name"]

        if not row["video_exists"] or not row["is_readable"]:
            print(f"Skipping invalid video: {dataset_name}")
            continue

        features = extract_klt_features_from_video(
            video_path=row["video_path"],
            fps=row["fps"]
        )

        if features is None:
            print(f"No valid tracking features extracted: {dataset_name}")
            continue

        features["dataset_name"] = dataset_name
        features["real_speed"] = row["real_speed"]

        rows.append(features)

    return pd.DataFrame(rows)