# DIOR Object Detection (YOLOv8 / ultralytics)

A modular rewrite of the original notebook. Ultralytics YOLOv8 is
already PyTorch under the hood, so no framework swap was needed here -
this just splits the pipeline into small, single-purpose modules, same
as the satellite-segmentation project. No command-line arguments
anywhere: everything is controlled by editing `config.py`.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Dataset download ids, paths, class names, all hyperparameters. **Edit this first.** |
| `downloader.py` | Downloads and extracts the DIOR zips from Google Drive (same file ids as the original notebook). |
| `annotations.py` | Parses XML annotations, converts them to YOLO-format `.txt` labels, draws bounding boxes. |
| `dataset_organizer.py` | Arranges images/labels into `datasets/images|labels/{train,val,test}` and writes the data yaml files. |
| `eda.py` | Optional exploratory plots (class distribution, metadata table). Not required for training. |
| `trainer.py` | Trains YOLOv8 with ultralytics. |
| `evaluator.py` | Validates/tests a trained checkpoint. |
| `inference.py` | Predicts on an image or video and visualizes results. |
| `main.py` | Entry point — runs the full pipeline end to end. |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

The dataset download is the same Google Drive source as the original
notebook — the file ids live in `config.GDRIVE_FILES`, no changes
needed unless you're pointing at a different copy. Everything downloads
into `raw_data/` and extracts into `dior_data/` in your working
directory (local or Colab — no path to edit, since it isn't a Drive
folder like the segmentation project's dataset).

## Run the full pipeline

```bash
python main.py
```

This will, in order:
1. Download + extract the DIOR zips (skipped if already present)
2. Convert every XML annotation to YOLO format
3. Move images/labels into `datasets/images|labels/{train,val,test}`
4. Write `config_train.yaml` and `config_test.yaml`
5. Train YOLOv8 for `config.EPOCHS` epochs

Weights land in `runs/detect/<config.RUN_NAME>/weights/best.pt`.

## Evaluate on the held-out test set

```python
from evaluator import evaluate_model
evaluate_model()  # uses config.BEST_WEIGHTS_PATH by default
```

## Predict on a new image or video

```python
from inference import load_model, predict_image, predict_video, visualize_prediction

model = load_model()  # loads config.BEST_WEIGHTS_PATH by default

# image, with ground-truth comparison
result = predict_image(model, "datasets/images/test/12345.jpg")
visualize_prediction(result,
                      ground_truth_label_path="datasets/labels/test/12345.txt",
                      ground_truth_image_path="datasets/images/test/12345.jpg")

# video
predict_video(model, "test_data/ships_test.mp4")
```

## Notes on changes from the original notebook

- **Already PyTorch**: ultralytics YOLO is PyTorch-based, so this is a
  modularization pass, not a framework swap.
- **`%%writefile` yaml cells → `dataset_organizer.write_data_yaml_files()`**:
  the train/test yaml files are now generated from `config.CLASS_NAMES`
  instead of being hand-typed twice.
- **Hardcoded class order**: `config.CLASS_NAMES` is the same 20 DIOR
  classes in the same order as the notebook's `config.yaml` cell, so
  existing label `.txt` files (if you've already converted some) stay
  compatible.
- **EDA split out**: the notebook's exploratory plots now live in
  `eda.py` as optional functions, since they're not part of the
  training path.
