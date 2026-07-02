"""
evaluator.py
------------
Runs validation/testing on a trained YOLOv8 checkpoint against the held
out test set defined in config_test.yaml.
"""

from ultralytics import YOLO

import config


def evaluate_model(weights_path: str = None, run_name: str = "yolov8n_val_on_test"):
    weights_path = weights_path or config.BEST_WEIGHTS_PATH
    model = YOLO(weights_path)
    results = model.val(data=config.TEST_YAML_PATH, imgsz=config.IMAGE_SIZE, name=run_name)
    return results
