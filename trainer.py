"""
trainer.py
----------
Trains the YOLOv8 model with ultralytics, using every setting from
config.py.
"""

from ultralytics import YOLO

import config


def train_model():
    model = YOLO(config.MODEL_CONFIG)
    results = model.train(
        data=config.TRAIN_YAML_PATH,
        imgsz=config.IMAGE_SIZE,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        name=config.RUN_NAME,
    )
    print(f"[INFO] Training complete. Best weights: {config.BEST_WEIGHTS_PATH}")
    return results
