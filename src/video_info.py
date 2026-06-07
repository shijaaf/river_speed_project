import cv2
import pandas as pd

from src.config import VIDEO_DIR, VIDEO_EXTENSIONS


def find_video_file(dataset_name):
    for extension in VIDEO_EXTENSIONS:
        video_path = VIDEO_DIR / f"{dataset_name}{extension}"

        if video_path.exists():
            return video_path

    return None


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "is_readable": False,
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
            "width": None,
            "height": None,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_sec = frame_count / fps if fps > 0 else None

    return {
        "is_readable": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def build_dataset_overview(labels_df):
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
                "height": None,
            })

            continue

        video_info = get_video_info(video_path)

        rows.append({
            "dataset_name": dataset_name,
            "real_speed": real_speed,
            "video_path": str(video_path),
            "video_exists": True,
            **video_info,
        })

    return pd.DataFrame(rows)

