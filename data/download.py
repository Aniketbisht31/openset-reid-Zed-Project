"""Download Market-1501 dataset from Google Drive."""
import os
import sys
import zipfile

try:
    import gdown
except ImportError:
    gdown = None


MARKET1501_GDRIVE_ID = "0B8-rUzbwVRk0c054eNlhNHpLRkE"
MARKET1501_URL = f"https://drive.google.com/uc?id={MARKET1501_GDRIVE_ID}"


def download_market1501(root: str = "./datasets/market1501") -> None:
    """Download and extract Market-1501 dataset.

    After extraction the directory layout is::

        root/
        ├── bounding_box_train/
        ├── bounding_box_test/
        ├── query/
        └── ...

    Args:
        root: Destination directory.
    """
    os.makedirs(root, exist_ok=True)

    # Skip if already present
    if os.path.isdir(os.path.join(root, "bounding_box_train")):
        print(f"[download] Market-1501 already exists at {root}")
        return

    if gdown is None:
        raise RuntimeError(
            "gdown is required for downloading.  pip install gdown"
        )

    zip_path = os.path.join(root, "Market-1501-v15.09.15.zip")

    print("[download] Downloading Market-1501 …")
    gdown.download(MARKET1501_URL, zip_path, quiet=False)

    print("[download] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    # The zip typically nests inside a subdirectory — flatten it.
    nested = os.path.join(root, "Market-1501-v15.09.15")
    if os.path.isdir(nested):
        import shutil
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(root, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(nested)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"[download] Market-1501 ready at {root}")


# ── CLI entry-point ──────────────────────────────────────────────────
if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else "./datasets/market1501"
    download_market1501(dst)
