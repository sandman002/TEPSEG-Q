"""TEPSEG: tumour epithelium regions in H&E whole-slide images (UNI2-h + MLP head)."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # many worker processes; keep each small

__version__ = "1.0.0"
