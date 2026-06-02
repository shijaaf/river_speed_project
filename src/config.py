from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
LABEL_DIR = DATA_DIR / "labels"

# Output directories
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"

# Label file path
LABEL_FILE = LABEL_DIR / "Validation Numbers.ods"

# Supported video formats
VIDEO_EXTENSIONS = [".avi", ".mp4"]