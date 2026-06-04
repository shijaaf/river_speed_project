import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torchvision.models as models
import torchvision.transforms as transforms


MAX_FRAMES_PER_VIDEO = 20
FRAME_SIZE = 224
OUTPUT_FILE = "outputs/results/phase13_deep_features.csv"


def load_resnet18_feature_extractor():
    """
    Load pretrained ResNet18 and remove the final classification layer.
    """

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    model.fc = torch.nn.Identity()

    model.eval()

    return model


def get_image_transform():
    """
    Create image preprocessing transform required by ResNet18.
    """

    return transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def sample_video_frames(video_path, max_frames=MAX_FRAMES_PER_VIDEO):
    """
    Sample frames uniformly from a video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num=min(max_frames, total_frames),
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)

        frames.append(pil_image)

    cap.release()

    return frames


def extract_deep_features_from_video(video_path, model, transform, device):
    """
    Extract one deep feature vector from a video using ResNet18.
    """

    frames = sample_video_frames(video_path)

    if len(frames) == 0:
        return None

    feature_vectors = []

    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame)

            input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(device)

            feature_vector = model(input_tensor)

            feature_vector = feature_vector.cpu().numpy().flatten()

            feature_vectors.append(feature_vector)

    video_feature = np.mean(
        np.array(feature_vectors),
        axis=0
    )

    return video_feature


def build_deep_feature_dataset(overview_df):
    """
    Build deep feature dataset for all videos.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = load_resnet18_feature_extractor()

    model = model.to(device)

    transform = get_image_transform()

    rows = []

    for _, row in tqdm(
        overview_df.iterrows(),
        total=len(overview_df),
        desc="Extracting deep features"
    ):
        dataset_name = row["dataset_name"]

        if not row["video_exists"] or not row["is_readable"]:
            print(f"Skipping invalid video: {dataset_name}")
            continue

        feature_vector = extract_deep_features_from_video(
            video_path=row["video_path"],
            model=model,
            transform=transform,
            device=device
        )

        if feature_vector is None:
            print(f"No deep features extracted: {dataset_name}")
            continue

        feature_dict = {
            f"deep_feature_{i}": value
            for i, value in enumerate(feature_vector)
        }

        feature_dict["dataset_name"] = dataset_name
        feature_dict["real_speed"] = row["real_speed"]

        rows.append(feature_dict)

    feature_df = pd.DataFrame(rows)

    return feature_df

