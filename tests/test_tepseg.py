"""Fast tests without slides or UNI2-h weights."""
import json
from importlib.resources import files

import cv2
import numpy as np
import shapely

from tepseg import cli
from tepseg import regions as R


def test_head_ships_with_package():
    h = np.load(files("tepseg") / "data" / "tepseg_head_v1.npz")
    assert h["W1"].shape == (512, 1536) and h["W3"].shape == (2, 256)
    assert str(h["backbone"]) == "MahmoodLab/UNI2-h"
    assert float(h["field_um"]) == R.FIELD_UM and int(h["input_px"]) == R.INPUT


def test_polygons_keep_large_holes_and_smooth():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 80, 1, -1)
    cv2.circle(m, (100, 100), 30, 0, -1)
    polys = R.polygons(m, scale=2.0)
    assert len(polys) == 1 and len(polys[0].interiors) == 1
    p = polys[0]
    expected = (np.pi * 80 ** 2 - np.pi * 30 ** 2) * 4
    assert abs(p.area - expected) / expected < 0.05
    assert p.is_valid


def test_chaikin_rounds_a_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    r = R._chaikin(sq, 2)
    assert len(r) == 16 and shapely.Polygon(r).area < 100


def test_tissue_mask_excludes_glass_and_fat():
    img = np.full((100, 100, 3), 245, np.uint8)                 # glass
    img[20:80, 20:80] = (200, 120, 180)                          # stained tissue
    img[40:60, 40:60] = (250, 250, 250)                          # fat
    m = R.tissue_mask(img)
    assert m[30, 30] == 1 and m[5, 5] == 0 and m[50, 50] == 0


def test_tasks_cover_every_field():
    todo = np.zeros((20, 30), bool)
    todo[2:15, 5:25] = True
    todo[18, 0] = True
    for stride, cells, span in ((128.0, R.BAND_CELLS, R.MAX_SPAN), (256.0, 1, 1)):
        tasks = R._tasks("x.svs", 0.25, todo, stride, 0, cells, span)
        covered = np.zeros_like(todo)
        for (r0, c0, nr, nc), *_ in tasks:
            covered[r0:r0 + nr, c0:c0 + nc] = True
        assert covered[todo].all()
        cell_px = stride / R.UM
        for (r0, c0, nr, nc), _, _, _, x0, y0, w0, h0, ow, oh in tasks:
            assert ow == round(nc * cell_px + 2 * (R.INPUT - cell_px) / 2)


def test_regions_geojson_is_qupath_annotations():
    p = shapely.box(0, 0, 4000, 4000)
    r = R.SlideResult(regions=[p], region_prob=[0.8], mpp=0.25, dimensions=(5000, 5000), tissue_mm2=2.0,
                      tumour_mm2=1.0, tiles=10, mask=np.zeros((2, 2), np.uint8), thumbnail=np.zeros((2, 2, 3), np.uint8),
                      tile_xy=np.zeros((0, 2)), tile_prob=np.zeros(0))
    fc = R.regions_geojson(r)
    props = fc["features"][0]["properties"]
    assert props["objectType"] == "annotation" and props["name"] == "TEPSEG"
    assert props["classification"]["name"] == "Tumor"
    assert abs(props["measurements"]["area_mm2"] - 1.0) < 1e-6
    assert r.fraction == 0.5


def test_expand_folders_and_summary(tmp_path):
    (tmp_path / "a.svs").write_bytes(b"")
    (tmp_path / "b.isyntax").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    found = cli.expand([str(tmp_path)])
    assert [p.split("/")[-1].split("\\")[-1] for p in found] == ["a.svs", "b.isyntax"]
    (tmp_path / "out" / "slides").mkdir(parents=True)
    (tmp_path / "out" / "slides" / "a.json").write_text(json.dumps({"slide": "a", "tumour_mm2": 1.5}))
    rows = cli.write_summary(tmp_path / "out").read_text().splitlines()
    assert rows[0].startswith("slide,tumour_mm2") and rows[1].startswith("a,1.5")
