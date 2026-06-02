import cv2
from tqdm import tqdm
from src.config import OUTPUT_DIR


# Target frame size for all videos
FRAME_WIDTH = 224
FRAME_HEIGHT = 224

# Save one frame every N frames
FRAME_STEP = 5


def preprocess_frame(frame):
    """
    Preprocess a single video frame.

    Steps:
    1. Resize frame
    2. Convert to grayscale
    3. Apply Gaussian blur

    Args:
        frame: Input BGR image from OpenCV.

    Returns:
        processed_frame: Preprocessed grayscale frame.
    """

    # Resize frame to fixed size
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Convert BGR frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise using Gaussian blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    return blurred_frame


def extract_preprocessed_frames(video_path, dataset_name, max_frames=None):
    """
    Extract and preprocess frames from a video.

    Args:
        video_path: Path to input video.
        dataset_name: Dataset/video name.
        max_frames: Maximum number of frames to process.

    Returns:
        list: List of preprocessed frames.
    """

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
                processed_frame = preprocess_frame(frame)
                frames.append(processed_frame)

            frame_index += 1
            pbar.update(1)

    cap.release()

    return frames


def save_sample_frames(frames, dataset_name, max_samples=5):
    """
    Save a few sample preprocessed frames for visual inspection.

    Args:
        frames: List of preprocessed frames.
        dataset_name: Dataset/video name.
        max_samples: Maximum number of sample frames to save.
    """

    sample_dir = OUTPUT_DIR / "frames" / dataset_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(len(frames), max_samples)

    for i in range(sample_count):
        output_path = sample_dir / f"sample_{i + 1}.png"

        cv2.imwrite(str(output_path), frames[i])