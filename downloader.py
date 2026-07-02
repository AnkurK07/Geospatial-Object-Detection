"""
downloader.py
-------------
Downloads the DIOR dataset zips from Google Drive and extracts them.
Same source files as the original notebook (gdown + shutil).
"""

import os
import shutil
import gdown

import config


def download_dataset():
    """Download the raw zip files if they aren't already there."""
    if os.path.exists(config.RAW_DATA_DIR):
        print("[INFO] Raw data directory exists, skipping download.")
        return

    os.makedirs(config.RAW_DATA_DIR)
    print("[INFO] Downloading data...")
    for filename, file_id in config.GDRIVE_FILES.items():
        out_path = os.path.join(config.RAW_DATA_DIR, filename)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output=out_path)
    print("[INFO] Data has been downloaded.")


def extract_dataset():
    """Unzip everything in raw_data/ into the DIOR data directory."""
    if os.path.exists(config.DIOR_DATA_DIR):
        print("[INFO] DIOR data directory exists, skipping extraction.")
        return

    os.makedirs(config.DIOR_DATA_DIR)
    for filename in os.listdir(config.RAW_DATA_DIR):
        filepath = os.path.join(config.RAW_DATA_DIR, filename)
        shutil.unpack_archive(filename=filepath, extract_dir=config.DIOR_DATA_DIR)
        print(f'[INFO] File "{filepath}" extracted to "{config.DIOR_DATA_DIR}".')


def prepare_raw_dataset():
    download_dataset()
    extract_dataset()
