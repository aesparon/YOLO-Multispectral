# utils_downloader.py

import os
import requests
from tqdm import tqdm
import zipfile

def download_and_extract_zip(url, output_zip_path, extract_dir):
    """
    Downloads a ZIP file from the given URL with resume support and extracts it.

    Args:
        url (str): Direct download URL.
        output_zip_path (str): Local path to save ZIP file.
        extract_dir (str): Directory to extract contents into.
    """

    # Check if partial file exists
    existing_size = os.path.getsize(output_zip_path) if os.path.exists(output_zip_path) else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    # Get total size from HEAD request
    print("🔍 Checking file info...")
    total_size = int(requests.head(url).headers.get("content-length", 0))

    if existing_size >= total_size:
        print("✅ File already fully downloaded.")
    else:
        print("📥 Downloading with resume support...")
        response = requests.get(url, headers=headers, stream=True)
        mode = "ab" if existing_size else "wb"

        with open(output_zip_path, mode) as f, tqdm(
            total=total_size,
            initial=existing_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"✅ Download complete: {output_zip_path}")

    # Extract
    print("📦 Extracting...")
    with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting", unit="file"):
            zip_ref.extract(file, extract_dir)

    print(f"✅ Extracted to: {extract_dir}")
