import cv2
import pandas as pd
from src.config import VIDEO_DIR, VIDEO_EXTENSIONS


def find_video_file(dataset_name):
    """
    Find a video file by dataset name.

    Args:
        dataset_name (str): Name of the dataset from label file.

    Returns:
        Path or None: Video file path if found, otherwise None.
    """

    for extension in VIDEO_EXTENSIONS:
        video_path = VIDEO_DIR / f"{dataset_name}{extension}"

        if video_path.exists():
            return video_path

    return None


def get_video_info(video_path):
    """
    Extract basic information from a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: Video information.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "is_readable": False,
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
            "width": None,
            "height": None
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_sec = frame_count / fps if fps > 0 else None

    cap.release()

    return {
        "is_readable": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height
    }


def build_dataset_overview(labels_df):
    """
    Build an overview table by combining labels and video metadata.

    Args:
        labels_df (pandas.DataFrame): Label dataframe.

    Returns:
        pandas.DataFrame: Dataset overview table.
    """

    rows = []

    for _, row in labels_df.iterrows():
        dataset_name = row["dataset_name"]
        real_speed = row["real_speed"]

        video_path = find_video_file(dataset_name)

        if video_path is None:
            rows.append({
                "dataset_name": dataset_name,
                "real_speed": real_speed,
                "video_path": None,
                "video_exists": False,
                "is_readable": False,
                "fps": None,
                "frame_count": None,
                "duration_sec": None,
                "width": None,
                "height": None
            })

            continue

        info = get_video_info(video_path)

        rows.append({
            "dataset_name": dataset_name,
            "real_speed": real_speed,
            "video_path": str(video_path),
            "video_exists": True,
            **info
        })

    return pd.DataFrame(rows)
