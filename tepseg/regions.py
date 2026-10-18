"""TEPSEG: tumour epithelium regions in H&E whole-slide images.

Every 128 um field of tissue (256 px at 20x, on a ``stride_um`` grid) is scored
by UNI2-h features and an MLP head. The smoothed probability map is
thresholded inside the tissue and traced into smooth polygons.

Speed: worker processes read one wide band of tissue per row and downsample it
to the model resolution; the GPU cuts the tiles out of each band and scores
them in fp16 batches while the workers read ahead.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import shapely
from scipy import ndimage as ndi

from .bandreader import init_worker, read_band
from .slides import Slide

log = logging.getLogger(__name__)

FIELD_UM, INPUT = 128.0, 224
UM = FIELD_UM / INPUT                   # model resolution, um/px
FINE_UM = 16.0                          # resolution of the region mask
TISSUE_FRACTION = 0.1                   # fields with less tissue are not scored
BAND_CELLS, MAX_SPAN = 8, 48
CLASS_NAME, CLASS_COLOR = "Tumor", [200, 0, 0]


def slide_id(path: str | Path) -> str:
    p = Path(path)
    return p.name[: -len(p.suffix)].strip() if p.suffix else p.name.strip()


def affordable_workers(requested: int, per_worker: int = 768 << 20) -> int:
    """Cap worker processes at the CPU count and, on Windows, at what free commit memory can hold
    (leaving 10% of the commit limit, at least 4 GB, for QuPath)."""
    n = min(requested, os.cpu_count() or 1)
    if sys.platform != "win32":
        return max(1, n)
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MemoryStatusEx()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return max(1, n)
    reserve = max(4 << 30, st.ullTotalPageFile // 10)
    return int(max(1, min(n, (st.ullAvailPageFile - reserve) // per_worker)))


def tissue_mask(rgb: np.ndarray, bright: int = 235) -> np.ndarray:
    """Tissue = saturated (Otsu on HSV saturation) and not bright (fat, glass): uint8 0/1 at thumbnail size."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, s = cv2.threshold(hsv[..., 1], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (s.astype(bool) & (hsv[..., 2] < bright)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _fraction(mask: np.ndarray, shape: tuple[int, int], extent: tuple[float, float]) -> np.ndarray:
    """Tissue fraction of each cell of a (gh, gw) grid that spans ``extent`` thumbnail px (x, y)."""
    gh, gw = shape
    ex, ey = extent
    pad = np.zeros((max(mask.shape[0], int(np.ceil(ey))), max(mask.shape[1], int(np.ceil(ex)))), np.float32)
    pad[: mask.shape[0], : mask.shape[1]] = mask
    return cv2.resize(pad[: int(np.ceil(ey)), : int(np.ceil(ex))], (gw, gh), interpolation=cv2.INTER_AREA)


def _tasks(path: str, mpp0: float, todo: np.ndarray, stride_um: float, level: int,
           band_cells: int = BAND_CELLS, max_span: int = MAX_SPAN) -> list:
    """Reads of bands of ``band_cells`` grid rows, split into runs of at most ``max_span`` cells."""
    gh, gw = todo.shape
    cell_px = stride_um / UM
    margin = (INPUT - cell_px) / 2
    tasks = []
    for r0 in range(0, gh, band_cells):
        cols = np.nonzero(todo[r0:r0 + band_cells].any(0))[0]
        if not len(cols):
            continue
        runs, start = [], cols[0]
        for a, b in zip(cols[:-1], cols[1:]):
            if b - a > 3 or b - start >= max_span:
                runs.append((start, a + 1))
                start = b
        runs.append((start, cols[-1] + 1))
        nr = min(band_cells, gh - r0)
        for c0, c1 in runs:
            out_w = int(round((c1 - c0) * cell_px + 2 * margin))
            out_h = int(round(nr * cell_px + 2 * margin))
            x0 = int(round((c0 * stride_um - margin * UM) / mpp0))
            y0 = int(round((r0 * stride_um - margin * UM) / mpp0))
            tasks.append(((r0, c0, nr, c1 - c0), path, mpp0, level, x0, y0, int(round(out_w * UM / mpp0)),
                          int(round(out_h * UM / mpp0)), out_w, out_h))
    return tasks


def _chaikin(ring: np.ndarray, iters: int) -> np.ndarray:
    """Corner cutting on a closed ring (n, 2) without the repeated end point."""
    for _ in range(iters):
        if len(ring) < 4:
            break
        nxt = np.roll(ring, -1, axis=0)
        ring = np.stack([0.75 * ring + 0.25 * nxt, 0.25 * ring + 0.75 * nxt], 1).reshape(-1, 2)
    return ring


def polygons(mask: np.ndarray, scale: float, smooth_iters: int = 2) -> list:
    """Polygons (with holes) of a binary mask, in level-0 px; ``scale`` = level-0 px per mask px."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    hier = hier[0]

    def ring(c):
        r = cv2.approxPolyDP(c, 0.7, True)[:, 0, :].astype(np.float64) + 0.5
        return _chaikin(r, smooth_iters) * scale if len(r) >= 3 else None

    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue
        shell = ring(c)
        if shell is None:
            continue
        holes, ch = [], hier[i][2]
        while ch != -1:
            h = ring(contours[ch])
            if h is not None:
                holes.append(h)
            ch = hier[ch][0]
        g = shapely.set_precision(shapely.make_valid(shapely.Polygon(shell, holes)), 0.1)
        out += [p for p in shapely.get_parts(shapely.get_parts(g)) if isinstance(p, shapely.Polygon) and not p.is_empty]
    return out


@dataclass
class SlideResult:
    """Regions and numbers for one slide. Polygons are in level-0 px."""
    regions: list
    region_prob: list
    mpp: float
    dimensions: tuple
    tissue_mm2: float
    tumour_mm2: float
    tiles: int
    mask: np.ndarray = field(repr=False)            # tumour mask on the FINE_UM grid (uint8 0/255)
    thumbnail: np.ndarray = field(repr=False)
    tile_xy: np.ndarray = field(repr=False)         # (n, 2) level-0 px of each scored field's top-left
    tile_prob: np.ndarray = field(repr=False)

    @property
    def fraction(self) -> float:
        return self.tumour_mm2 / self.tissue_mm2 if self.tissue_mm2 > 0 else 0.0


def segment_slide(path: str, classifier, pool, stride_um: float = 128.0, threshold: float = 0.5,
                  smooth_um: float = 100.0, min_area_mm2: float = 0.5, min_hole_mm2: float = 0.1,
                  batch: int = 128, mpp_fallback: Optional[float] = None,
                  progress: Optional[Callable[[int, int], None]] = None) -> SlideResult:
    import torch

    s = Slide(path, mpp_fallback)
    mpp0, (W0, H0) = s.mpp, s.dimensions
    gh, gw = int(np.ceil(H0 * mpp0 / stride_um)), int(np.ceil(W0 * mpp0 / stride_um))
    thumb = s.thumbnail()
    t_um = mpp0 * W0 / thumb.shape[1]                       # thumbnail um/px
    tissue = tissue_mask(thumb)
    todo = _fraction(tissue, (gh, gw), (gw * stride_um / t_um, gh * stride_um / t_um)) >= TISSUE_FRACTION
    level = s.best_level(UM)
    s.close()
    # fields spaced apart (fast mode): read just the fields, not the gaps between them
    sparse = stride_um >= 1.5 * FIELD_UM
    tasks = _tasks(path, mpp0, todo, stride_um, level, *((1, 1) if sparse else (BAND_CELLS, MAX_SPAN)))
    prob = np.zeros((gh, gw), np.float32)
    done = np.zeros((gh, gw), bool)
    cell_px = stride_um / UM
    n_total, n_done = int(todo.sum()), 0
    buf_t, buf_ij = [], []

    def flush():
        nonlocal buf_t, buf_ij
        if not buf_t:
            return
        x = torch.cat(buf_t)
        p = torch.cat([classifier(x[k:k + batch]) for k in range(0, len(x), batch)]).cpu().numpy()
        ij = np.concatenate(buf_ij)
        prob[ij[:, 0], ij[:, 1]] = p
        done[ij[:, 0], ij[:, 1]] = True
        buf_t, buf_ij = [], []

    for (r0, c0, nr, nc), band in pool.imap_unordered(read_band, tasks, chunksize=4 if sparse else 1):
        sel = np.argwhere(todo[r0:r0 + nr, c0:c0 + nc])
        if not len(sel):
            continue
        b = torch.from_numpy(band).to(classifier.device, non_blocking=True).permute(2, 0, 1)
        ys = np.round(sel[:, 0] * cell_px).astype(int)
        xs = np.round(sel[:, 1] * cell_px).astype(int)
        buf_t.append(torch.stack([b[:, y:y + INPUT, x:x + INPUT] for y, x in zip(ys, xs)]))
        buf_ij.append(sel + [r0, c0])
        n_done += len(sel)
        if sum(len(t) for t in buf_t) >= 2 * batch:
            flush()
        if progress:
            progress(n_done, n_total)
    flush()

    # probability map -> fine grid, inside the tissue
    sig = smooth_um / stride_um
    w = done.astype(np.float32)
    P = ndi.gaussian_filter(prob * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-6)
    fh, fw = int(np.ceil(H0 * mpp0 / FINE_UM)), int(np.ceil(W0 * mpp0 / FINE_UM))
    # cell centres sit at (i + 0.5) * stride; map the grid onto the fine grid with matching centres
    up = stride_um / FINE_UM
    M = np.float32([[up, 0, (up - 1) / 2], [0, up, (up - 1) / 2]])
    P_fine = cv2.warpAffine(P, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    covered = cv2.warpAffine(w, M, (fw, fh), flags=cv2.INTER_LINEAR) > 0
    tissue_fine = _fraction(tissue, (fh, fw), (fw * FINE_UM / t_um, fh * FINE_UM / t_um)) >= 0.5
    m = ((P_fine >= threshold) & tissue_fine & covered).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    px_mm2 = (FINE_UM / 1000) ** 2
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] * px_mm2 >= min_area_mm2
    m = keep[lab].astype(np.uint8)
    # fill small holes (and keep larger ones: stroma, necrosis, vessels inside the tumour)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    small = [i for i in range(1, n) if i not in border and st[i, cv2.CC_STAT_AREA] * px_mm2 < min_hole_mm2]
    if small:
        m[np.isin(lab, small)] = 1

    regions = polygons(m, FINE_UM / mpp0)
    regions.sort(key=lambda p: -p.area)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    means = ndi.mean(P_fine, lab, np.arange(1, n)) if n > 1 else []
    region_prob = []
    for p in regions:
        q = p.representative_point()
        i, j = int(q.y * mpp0 / FINE_UM), int(q.x * mpp0 / FINE_UM)
        l = lab[min(i, fh - 1), min(j, fw - 1)]
        region_prob.append(float(means[l - 1]) if l > 0 else float("nan"))
    sel = np.argwhere(done)
    return SlideResult(
        regions=regions, region_prob=region_prob, mpp=mpp0, dimensions=(W0, H0),
        tissue_mm2=float(tissue_fine.sum() * px_mm2),
        tumour_mm2=float(sum(p.area for p in regions) * mpp0 ** 2 / 1e6),
        tiles=n_total, mask=m * 255, thumbnail=thumb,
        tile_xy=np.round((sel[:, ::-1] * stride_um + (stride_um - FIELD_UM) / 2) / mpp0).astype(np.int64),
        tile_prob=prob[done])


def make_pool(workers: int = 8):
    """Spawned reader processes (spawn: safe after CUDA/torch threads have started)."""
    return mp.get_context("spawn").Pool(affordable_workers(workers), initializer=init_worker)


def _geometry(p: shapely.Polygon) -> dict:
    return {"type": "Polygon", "coordinates": [list(map(list, p.exterior.coords))] +
            [list(map(list, h.coords)) for h in p.interiors]}


def regions_geojson(r: SlideResult) -> dict:
    """Tumour regions as QuPath annotations (class 'Tumor', name 'TEPSEG')."""
    feats = []
    for p, pr in zip(r.regions, r.region_prob):
        meas = {"area_mm2": float(p.area * r.mpp ** 2 / 1e6)}
        if np.isfinite(pr):
            meas["mean_probability"] = round(pr, 4)
        feats.append({"type": "Feature", "geometry": _geometry(p), "properties": {
            "objectType": "annotation", "name": "TEPSEG",
            "classification": {"name": CLASS_NAME, "color": CLASS_COLOR}, "measurements": meas}})
    return {"type": "FeatureCollection", "features": feats}


def overlay(r: SlideResult, max_px: int = 2000) -> np.ndarray:
    """Thumbnail (BGR) with the tumour filled in green and outlined in white, for quick review."""
    img = r.thumbnail
    s = min(1.0, max_px / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    k = img.shape[1] / r.dimensions[0]
    fill = img.copy()
    for p in r.regions:
        rings = [np.round(np.asarray(p.exterior.coords) * k).astype(np.int32)] + \
                [np.round(np.asarray(h.coords) * k).astype(np.int32) for h in p.interiors]
        cv2.fillPoly(fill, rings, (50, 205, 50))
    img = cv2.addWeighted(fill, 0.4, img, 0.6, 0)
    for p in r.regions:
        rings = [np.round(np.asarray(p.exterior.coords) * k).astype(np.int32)] + \
                [np.round(np.asarray(h.coords) * k).astype(np.int32) for h in p.interiors]
        cv2.polylines(img, rings, True, (255, 255, 255), 2, cv2.LINE_AA)
    return img
