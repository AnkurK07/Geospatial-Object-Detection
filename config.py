"""
config.py
---------
Every path, dataset id, and hyperparameter used by the project lives
here. Edit this one file to change how things run - no command-line
flags needed anywhere.
"""

# ----------------------------------------------------------------------
# Raw dataset download (same Google Drive files as the original notebook)
# ----------------------------------------------------------------------
RAW_DATA_DIR = "raw_data"
DIOR_DATA_DIR = "dior_data"

GDRIVE_FILES = {
    "Annotations.zip":          "1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw",
    "JPEGImages-test.zip":      "11SXPqcESez9qTn4Z5Q3v35K9hRwO_epr",
    "JPEGImages-trainval.zip":  "1ZHbHDM6hYAEGDC_K5eiW0yF_lzVgpuir",
}

ANNOTATIONS_DIR = f"{DIOR_DATA_DIR}/Annotations/Horizontal Bounding Boxes"
TRAINVAL_IMAGES_DIR = f"{DIOR_DATA_DIR}/JPEGImages-trainval"
TEST_IMAGES_DIR = f"{DIOR_DATA_DIR}/JPEGImages-test"
YOLO_ANNOTATIONS_DIR = f"{DIOR_DATA_DIR}/yolo_annotations"

# ----------------------------------------------------------------------
# Organized YOLO dataset layout (what ultralytics expects)
# ----------------------------------------------------------------------
DATASET_ROOT = "datasets"        # <-- change this if you want it elsewhere
IMG_TRAIN_DIR = f"{DATASET_ROOT}/images/train"
IMG_VAL_DIR = f"{DATASET_ROOT}/images/val"
IMG_TEST_DIR = f"{DATASET_ROOT}/images/test"
LABEL_TRAIN_DIR = f"{DATASET_ROOT}/labels/train"
LABEL_VAL_DIR = f"{DATASET_ROOT}/labels/val"
LABEL_TEST_DIR = f"{DATASET_ROOT}/labels/test"

# Fraction of the held-out test images to carve out as a validation split
# during training (the rest is used purely for final testing).
VAL_SPLIT_FROM_TEST = 0.2

TRAIN_YAML_PATH = "config_train.yaml"
TEST_YAML_PATH = "config_test.yaml"

# ----------------------------------------------------------------------
# Classes (DIOR dataset, 20 classes - order must match the label ids
# baked into every YOLO .txt annotation)
# ----------------------------------------------------------------------
CLASS_NAMES = [
    "Expressway-Service-area",
    "Expressway-toll-station",
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------
PRETRAINED = False                          # True -> fine-tune yolov8n.pt, False -> train from scratch
MODEL_CONFIG = "yolov8n.pt" if PRETRAINED else "yolov8n.yaml"

IMAGE_SIZE = 800
EPOCHS = 50
BATCH_SIZE = 16
RUN_NAME = "yolov8n_epochs50_batch16"

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
CONF_THRESHOLD = 0.5
BEST_WEIGHTS_PATH = f"runs/detect/{RUN_NAME}/weights/best.pt"
