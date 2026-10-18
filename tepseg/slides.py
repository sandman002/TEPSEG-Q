"""Slide access for TEPSEG: tiffslide (SVS, NDPI, generic TIFF) and Philips iSyntax (pyisyntax).

Coordinates are level-0 pixels unless stated otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

THUMB_UM = 16.0          # the tissue mask needs a thumbnail at least this fine (um/px)


def is_isyntax(path: str | Path) -> bool:
    return str(path).lower().endswith(".isyntax")


class Slide:
    """The few things TEPSEG needs from a whole-slide image, for either backend."""

    def __init__(self, path: str | Path, mpp_fallback: Optional[float] = None):
        self.path = str(path)
        self.isyntax = is_isyntax(path)
        if self.isyntax:
            from isyntax import ISyntax

            self._s = ISyntax.open(self.path)
            mpp = self._s.mpp_x
            mag = None
        else:
            import tiffslide

            self._s = tiffslide.TiffSlide(self.path)
            p = self._s.properties
            mpp = p.get("tiffslide.mpp-x")
            mag = p.get("tiffslide.objective-power") or p.get("aperio.AppMag")
        self.dimensions = tuple(int(v) for v in self._s.dimensions)
        self.level_downsamples = [float(d) for d in self._s.level_downsamples]
        self.level_dimensions = [tuple(int(v) for v in d) for d in self._s.level_dimensions]
        if not mpp and mag:
            try:
                mpp = 10.0 / float(mag)          # scanner convention: magnification x um/px ~ 10
            except ValueError:
                mpp = None
        mpp = mpp or mpp_fallback
        if not mpp:
            raise ValueError(f"{Path(self.path).name}: no pixel size in the file; pass --mpp")
        self.mpp = float(mpp)

    def best_level(self, um_per_px: float) -> int:
        """Coarsest level that is still at least as fine as ``um_per_px``."""
        ok = [i for i, d in enumerate(self.level_downsamples) if self.mpp * d <= um_per_px * 1.05]
        return max(ok) if ok else 0

    def read(self, level: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """RGB uint8 (h, w, 3) of level ``level``; x0, y0 in level-0 px, w, h in level px.
        Areas without image data are white."""
        if self.isyntax:
            ds = self.level_downsamples[level]
            a = np.asarray(self._s.read_region(int(x0 / ds), int(y0 / ds), int(w), int(h), level=level))
            if a.shape[-1] == 4:
                rgb = a[..., :3].copy()
                rgb[a[..., 3] == 0] = 255
                return rgb
            return np.ascontiguousarray(a[..., :3])
        return np.asarray(self._s.read_region((int(x0), int(y0)), level, (int(w), int(h))).convert("RGB"))

    def thumbnail(self) -> np.ndarray:
        """RGB of the coarsest level that is at most THUMB_UM um/px (for the tissue mask)."""
        level = self.best_level(THUMB_UM)
        if not self.isyntax:
            try:  # tifffile decodes whole levels in C, much faster than tiffslide (NDPI especially)
                import tifffile

                lv = tifffile.TiffFile(self.path).series[0].levels
                if level < len(lv):
                    a = lv[level].asarray()
                    if a.ndim == 3 and a.shape[2] >= 3 and a.dtype == np.uint8 and \
                            abs(a.shape[1] - self.level_dimensions[level][0]) <= 2:
                        return np.ascontiguousarray(a[..., :3])
            except Exception:
                pass
        w, h = self.level_dimensions[level]
        return self.read(level, 0, 0, w, h)

    def close(self) -> None:
        try:
            self._s.close()
        except Exception:
            pass
