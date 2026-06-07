import cv2
from tqdm import tqdm

from src.config import FRAME_HEIGHT, FRAME_STEP, FRAME_WIDTH, OUTPUT_DIR


def preprocess_frame(frame):
    resized_frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    gray_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2GRAY,
    )

    blurred_frame = cv2.GaussianBlur(
        gray_frame,
        (5, 5),
        0,
    )

    return blurred_frame


def extract_preprocessed_frames(video_path, dataset_name, max_frames=None):
    frames = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frame_index = 0

    with tqdm(total=total_frames, desc=f"Processing {dataset_name}") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if max_frames is not None and frame_index >= max_frames:
                break

            if frame_index % FRAME_STEP == 0:
                frames.append(preprocess_frame(frame))

            frame_index += 1
            pbar.update(1)

    cap.release()

    return frames


def save_sample_frames(frames, dataset_name, max_samples=5):
    sample_dir = OUTPUT_DIR / "frames" / dataset_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(len(frames), max_samples)

    for index in range(sample_count):
        output_path = sample_dir / f"sample_{index + 1}.png"
        cv2.imwrite(str(output_path), frames[index])
