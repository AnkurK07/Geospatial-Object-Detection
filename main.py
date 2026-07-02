"""
main.py
-------
Entry point - just run:

    python main.py

Downloads the dataset (if needed), converts annotations to YOLO format,
organizes files into ultralytics' expected layout, writes the data
yaml files, and trains. Everything is configured through config.py -
no command-line flags.
"""

from downloader import prepare_raw_dataset
from annotations import convert_all_annotations
from dataset_organizer import organize_train_val, organize_test, write_data_yaml_files
from trainer import train_model


def run_pipeline():
    prepare_raw_dataset()
    convert_all_annotations()
    organize_train_val()
    organize_test()
    write_data_yaml_files()
    train_model()


if __name__ == "__main__":
    run_pipeline()
