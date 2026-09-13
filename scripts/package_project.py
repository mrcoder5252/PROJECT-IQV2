"""
Project-IQ: Complete Project Packager & Zip Generator
Creates clean, production-grade distribution zip archives in:
1. Local project root: project_iq_complete.zip
2. User Downloads directory: C:/Users/mrcod/Downloads/project_iq_complete.zip
Excludes temporary files, caches, and test artifacts.
"""

import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = Path(r"C:\Users\mrcod\Downloads")

EXCLUDE_DIRS = {
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv", "env",
    "chroma_db_test", "chunks_test", "cleaned_test", "extracted_test", "reports_test"
}

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd"}
EXCLUDE_FILES = {"project_iq_complete.zip"}


def create_distribution_zip():
    local_zip = PROJECT_ROOT / "project_iq_complete.zip"
    downloads_zip = DOWNLOADS_DIR / "project_iq_complete.zip"

    print(f"Packaging Project-IQ from: {PROJECT_ROOT}")
    print(f"Output targets:")
    print(f"  1. {local_zip}")
    print(f"  2. {downloads_zip}")

    total_files = 0
    total_bytes = 0

    with zipfile.ZipFile(local_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in sorted(files):
                if file in EXCLUDE_FILES or any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue

                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)

                # Skip files inside any test temp folders
                if any(part in EXCLUDE_DIRS for part in rel_path.parts):
                    continue

                zf.write(full_path, arcname=str(rel_path).replace("\\", "/"))
                total_files += 1
                total_bytes += full_path.stat().st_size

    zip_size = local_zip.stat().st_size
    print(f"Successfully packaged {total_files} files ({total_bytes:,} uncompressed bytes) into {zip_size:,} bytes.")

    # Copy to Downloads
    if DOWNLOADS_DIR.exists():
        shutil.copy2(local_zip, downloads_zip)
        print(f"Copied updated archive to: {downloads_zip}")
    else:
        print(f"Downloads directory not found at: {DOWNLOADS_DIR}")


if __name__ == "__main__":
    create_distribution_zip()
