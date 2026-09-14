import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LEAD_DATA_ROOT", PROJECT_ROOT / "data"))
BASE_MODEL_PATH = os.environ.get("LEAD_BASE_MODEL", str(PROJECT_ROOT / "models" / "base"))
TRAIN_FILE = os.environ.get("LEAD_TRAIN_FILE", str(DATA_ROOT / "train.json"))
VAL_FILE = os.environ.get("LEAD_VAL_FILE", str(DATA_ROOT / "val.json"))
TEST_FILE = os.environ.get("LEAD_TEST_FILE", str(DATA_ROOT / "test.json"))
LABEL_FILE = os.environ.get("LEAD_LABEL_FILE", str(DATA_ROOT / "labels.csv"))
IMAGE_ROOT = os.environ.get("LEAD_IMAGE_ROOT", str(DATA_ROOT))

