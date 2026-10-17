"""Package what the Colab notebook needs into colab/colab_bundle.zip.

Usage:
    python src/make_colab_bundle.py
"""
import glob
import zipfile
from pathlib import Path

FILES = ["requirements.txt", "data/processed/chunks.parquet", "data/processed/normaliser_candidates.parquet",
         "results/normaliser/threshold.json", "labels/pairs_labelled.csv"]

if __name__ == "__main__":
    Path("colab").mkdir(exist_ok=True)
    with zipfile.ZipFile("colab/colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob("src/*.py") + [f for f in FILES if Path(f).exists()]:
            z.write(f)
            print("added", f)
    print(f"colab/colab_bundle.zip: {Path('colab/colab_bundle.zip').stat().st_size / 1e6:.1f} MB")
