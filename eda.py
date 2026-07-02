
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from annotations import extract_data_from_xml


def build_metadata_table() -> pd.DataFrame:
    """One row per image: filename, split, size, and objects present."""
    annot_files = sorted(
        os.path.join(config.ANNOTATIONS_DIR, f)
        for f in os.listdir(config.ANNOTATIONS_DIR) if f.endswith(".xml")
    )

    rows = []
    for annot_file in annot_files:
        data = extract_data_from_xml(annot_file)
        img_w, img_h, _ = data["image_size"]
        rows.append({
            "filename": data["filename"],
            "width": img_w,
            "height": img_h,
            "objects": ", ".join(sorted({b["class"] for b in data["bboxes"]})),
            "num_objects": len(data["bboxes"]),
        })
    return pd.DataFrame(rows)


def class_instance_counts(meta_df: pd.DataFrame = None) -> Counter:
    """Total count of each object class across all annotations."""
    meta_df = meta_df if meta_df is not None else build_metadata_table()
    all_classes = []
    for objects in meta_df["objects"]:
        all_classes.extend(objects.split(", "))
    return Counter(all_classes)


def plot_top_classes(meta_df: pd.DataFrame = None, top_n: int = 20):
    meta_df = meta_df if meta_df is not None else build_metadata_table()
    meta_df["objects"].value_counts()[:top_n].plot(kind="barh").invert_yaxis()
    plt.xlabel("Images (Count)")
    plt.title(f"Top {top_n} Object Class Combinations")
    plt.show()


def plot_class_instance_counts(meta_df: pd.DataFrame = None):
    counts = class_instance_counts(meta_df)
    plt.figure(figsize=(14, 6))
    plt.barh(list(counts.keys()), list(counts.values()))
    plt.xlabel("Objects (Count)")
    plt.title("Total Object Instances Per Class")
    plt.tight_layout()
    plt.show()
