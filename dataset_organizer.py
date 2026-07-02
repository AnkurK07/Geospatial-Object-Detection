"""
dataset_organizer.py
---------------------
Arranges images and YOLO labels into the folder layout ultralytics
expects (datasets/images/{train,val,test}, datasets/labels/{train,val,test})
and writes the data yaml files that point YOLO at them.
"""

import os
import shutil

import config


def _list_files(directory: str, extension: str) -> list:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)
    )


def _move_files(filepaths: list, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    for filepath in filepaths:
        if os.path.isfile(filepath):
            shutil.move(filepath, dest_dir)


def _move_labels_for(image_filepaths: list, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    for image_path in image_filepaths:
        label_name = os.path.basename(image_path).replace("jpg", "txt")
        label_path = os.path.join(config.YOLO_ANNOTATIONS_DIR, label_name)
        if os.path.isfile(label_path):
            shutil.move(label_path, dest_dir)


def organize_train_val():
    """Move trainval images/labels to train/, and a slice of test images/labels to val/."""
    if os.path.exists(config.IMG_TRAIN_DIR):
        print("[INFO] Dataset already organized, skipping.")
        return

    trainval_images = _list_files(config.TRAINVAL_IMAGES_DIR, ".jpg")
    test_images = _list_files(config.TEST_IMAGES_DIR, ".jpg")
    val_count = int(len(test_images) * config.VAL_SPLIT_FROM_TEST)
    val_images = test_images[:val_count]

    print(f"[INFO] Moving {len(trainval_images)} training images/labels...")
    _move_files(trainval_images, config.IMG_TRAIN_DIR)
    _move_labels_for(trainval_images, config.LABEL_TRAIN_DIR)

    print(f"[INFO] Moving {len(val_images)} validation images/labels...")
    _move_files(val_images, config.IMG_VAL_DIR)
    _move_labels_for(val_images, config.LABEL_VAL_DIR)


def organize_test():
    """Move the remaining held-out test images/labels into datasets/images/test."""
    if os.path.exists(config.IMG_TEST_DIR):
        print("[INFO] Test set already organized, skipping.")
        return

    test_images = _list_files(config.TEST_IMAGES_DIR, ".jpg")
    print(f"[INFO] Moving {len(test_images)} test images/labels...")
    _move_files(test_images, config.IMG_TEST_DIR)
    _move_labels_for(test_images, config.LABEL_TEST_DIR)


def _write_yaml(path: str, train_split: str, val_split: str):
    lines = [
        f"path: {os.path.abspath(config.DATASET_ROOT)}",
        f"train: {train_split}" if train_split else "train:",
        f"val: {val_split}",
        "",
        "names:",
    ]
    for idx, name in config.ID_TO_CLASS.items():
        lines.append(f"  {idx}: {name}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_data_yaml_files():
    """Write config_train.yaml (train+val) and config_test.yaml (test only)."""
    _write_yaml(config.TRAIN_YAML_PATH, "images/train", "images/val")
    _write_yaml(config.TEST_YAML_PATH, "", "images/test")
    print(f"[INFO] Wrote {config.TRAIN_YAML_PATH} and {config.TEST_YAML_PATH}.")
