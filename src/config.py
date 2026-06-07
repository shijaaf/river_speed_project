from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
LABEL_DIR = DATA_DIR / "labels"

OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"

LABEL_FILE = LABEL_DIR / "Validation Numbers.ods"

VIDEO_EXTENSIONS = [".avi", ".mp4"]

RANDOM_STATE = 42

FRAME_WIDTH = 224
FRAME_HEIGHT = 224
FRAME_STEP = 5

DEFAULT_MAX_FRAMES = None

RF_ESTIMATORS = 300
ET_ESTIMATORS = 500
GB_ESTIMATORS = 300

DEFAULT_MAX_DEPTH = 4
DEFAULT_MIN_SAMPLES_LEAF = 2

PHASE3_FEATURE_FILE = RESULTS_DIR / "phase3_optical_flow_features.csv"
PHASE6_FEATURE_FILE = RESULTS_DIR / "phase6_improved_features.csv"
PHASE7_KLT_FEATURE_FILE = RESULTS_DIR / "phase7_klt_tracking_features.csv"
PHASE9_ADVANCED_FEATURE_FILE = RESULTS_DIR / "phase9_advanced_motion_features.csv"
PHASE13_DEEP_FEATURE_FILE = RESULTS_DIR / "phase13_deep_features.csv"
