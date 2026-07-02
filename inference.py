"""
inference.py
------------
Load a trained checkpoint and run prediction on a single image or
video file, with an optional side-by-side ground-truth comparison.
"""

import os
import matplotlib.pyplot as plt

from ultralytics import YOLO

import config
from annotations import plot_bboxes


def load_model(weights_path: str = None) -> YOLO:
    weights_path = weights_path or config.BEST_WEIGHTS_PATH
    return YOLO(weights_path)


def predict_image(model: YOLO, image_path: str, save: bool = True):
    """Run detection on one image. Returns the ultralytics Results object."""
    results = model.predict(source=image_path, imgsz=config.IMAGE_SIZE,
                             save=save, conf=config.CONF_THRESHOLD)
    return results[0]


def predict_video(model: YOLO, video_path: str, save: bool = True):
    """Run detection on a video file. Predictions are saved under runs/detect/."""
    return model.predict(source=video_path, imgsz=config.IMAGE_SIZE,
                          save=save, conf=config.CONF_THRESHOLD)


def visualize_prediction(result, ground_truth_label_path: str = None, ground_truth_image_path: str = None):
    """Show the predicted image, optionally next to the ground-truth boxes."""
    pred_save_path = os.path.join(result.save_dir, os.path.basename(result.path))

    if ground_truth_label_path and ground_truth_image_path:
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plot_bboxes(ground_truth_image_path, ground_truth_label_path)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(plt.imread(pred_save_path))
        plt.title("Predicted Image")
        plt.axis(False)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(plt.imread(pred_save_path))
        plt.title("Predicted Image")
        plt.axis(False)
    plt.show()
