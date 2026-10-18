"""Band reader for TEPSEG's worker processes (no torch: workers start fast and stay small).

A task reads one rectangle of a slide at (or finer than) the model resolution
and returns it already downsampled to the model resolution.

Fast path for tiled-JPEG pyramids (e.g. Aperio SVS): the raw JPEG tiles are
decoded directly at 1/2, 1/4 or 1/8 size (libjpeg DCT scaling via Pillow's
draft mode), the coarsest that is still at least as fine as the target.
Other formats (NDPI strips, JPEG 2000, iSyntax ...) go through the slide reader.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .slides import Slide

_slides: dict = {}
_fast: dict = {}


def init_worker() -> None:
    cv2.setNumThreads(1)


def _fast_page(path: str, level: int):
    """(TiffFile, page) for ``level`` if it is a tiled JPEG page the fast path can decode, else None."""
    key = (path, level)
    if key in _fast:
        return _fast[key]
    page = None
    if not path.lower().endswith(".isyntax"):
        try:
            import tifffile

            tf = tifffile.TiffFile(path)
            if not tf.is_ndpi and tf.series:
                lv = tf.series[0].levels
                if level < len(lv):
                    pg = lv[level].keyframe if hasattr(lv[level], "keyframe") else lv[level].pages[0]
                    if pg.is_tiled and pg.compression == 7 and pg.photometric in (2, 6) and pg.samplesperpixel == 3:
                        page = (tf, pg)
        except Exception:
            page = None
    _fast[key] = page
    return page


def _decode_tile(data: bytes, tables, photometric: int, scale: int) -> np.ndarray:
    if tables is not None:
        data = tables[:-2] + data[2:]                     # tables' EOI and the tile's SOI dropped
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    # Photometric RGB: the JPEG holds R,G,B without colour transform; ask for the raw channels.
    im.draft("YCbCr" if photometric == 2 else "RGB", (max(1, w // scale), max(1, h // scale)))
    return np.asarray(im if photometric == 2 else im.convert("RGB"))


def _read_fast(tf, pg, x, y, w, h, scale_hint: float) -> np.ndarray:
    """Region (x, y, w, h) in this page's pixels, decoded at a power-of-two reduction <= scale_hint."""
    scale = 1
    while scale * 2 <= scale_hint and scale < 8:
        scale *= 2
    tw, th = pg.tilewidth, pg.tilelength
    nx = (pg.imagewidth + tw - 1) // tw
    tx0, ty0 = x // tw, y // th
    tx1, ty1 = (x + w - 1) // tw, (y + h - 1) // th
    sw, sh = tw // scale, th // scale
    canvas = np.full(((ty1 - ty0 + 1) * sh, (tx1 - tx0 + 1) * sw, 3), 255, np.uint8)
    fh = tf.filehandle
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            i = ty * nx + tx
            if i >= len(pg.dataoffsets) or pg.databytecounts[i] == 0:
                continue
            fh.seek(pg.dataoffsets[i])
            a = _decode_tile(fh.read(pg.databytecounts[i]), pg.jpegtables, pg.photometric, scale)
            canvas[(ty - ty0) * sh:(ty - ty0) * sh + a.shape[0], (tx - tx0) * sw:(tx - tx0) * sw + a.shape[1]] = a[:sh, :sw]
    ox, oy = (x - tx0 * tw) // scale, (y - ty0 * th) // scale
    return canvas[oy:oy + max(1, h // scale), ox:ox + max(1, w // scale)]


def read_band(task):
    """task = (key, path, mpp, level, x0, y0, w0, h0, out_w, out_h) with x0..h0 in level-0 pixels.

    Returns (key, uint8 array (out_h, out_w, 3)), white where the rectangle leaves the slide.
    """
    key, path, mpp, level, x0, y0, w0, h0, out_w, out_h = task
    s = _slides.get(path)
    if s is None:
        s = _slides[path] = Slide(path, mpp_fallback=mpp)
    ds = s.level_downsamples[level]
    W0, H0 = s.dimensions
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W0, x0 + w0), min(H0, y0 + h0)
    if ax1 - ax0 < ds or ay1 - ay0 < ds:
        return key, out
    rw, rh = int((ax1 - ax0) / ds), int((ay1 - ay0) / ds)
    sx, sy = out_w / w0, out_h / h0                       # model px per level-0 px
    tw, th = max(1, int(round((ax1 - ax0) * sx))), max(1, int(round((ay1 - ay0) * sy)))
    fast = _fast_page(path, level)
    if fast is not None:
        reg = _read_fast(fast[0], fast[1], int(ax0 / ds), int(ay0 / ds), rw, rh, scale_hint=rw / tw)
    else:
        reg = s.read(level, ax0, ay0, rw, rh)
    if reg.shape[1] != tw or reg.shape[0] != th:
        reg = cv2.resize(reg, (tw, th), interpolation=cv2.INTER_AREA)
    ox, oy = int(round((ax0 - x0) * sx)), int(round((ay0 - y0) * sy))
    out[oy:oy + th, ox:ox + tw] = reg[: out_h - oy, : out_w - ox]
    return key, out
