"""Command line: ``tepseg run SLIDES_OR_FOLDERS -o OUT`` (or ``python -m tepseg``).

Per slide (id = file name without its last extension):
    OUT/regions/<id>.geojson   tumour regions as QuPath annotations (class "Tumor", name "TEPSEG")
    OUT/masks/<id>.png         tumour mask, 16 um per pixel
    OUT/overlays/<id>.jpg      thumbnail with the regions, for quick review
    OUT/slides/<id>.json       numbers for the slide
    OUT/tiles/<id>.csv         tumour probability of every scored field (--save-tiles)
and OUT/summary.csv with one row per slide.

``--json`` prints one JSON event per line on stdout (used by the QuPath extension).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__

SLIDE_EXT = (".svs", ".ndpi", ".tif", ".tiff", ".isyntax", ".scn", ".bif", ".qptiff")
SUMMARY_COLS = ["slide", "tumour_mm2", "tissue_mm2", "tumour_fraction", "regions", "tiles", "mpp",
                "stride_um", "threshold", "seconds", "path"]


def _emit(on: bool, **ev) -> None:
    if on:
        sys.stdout.write(json.dumps(ev) + "\n")
        sys.stdout.flush()


def expand(inputs: list[str]) -> list[str]:
    out = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in SLIDE_EXT)
        else:
            out.append(s)
    return out


def write_summary(out: Path) -> Path:
    rows = []
    for f in sorted((out / "slides").glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    path = out / "summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def cmd_run(args) -> int:
    import cv2
    import numpy as np

    from .model import TumourClassifier, WeightsError
    from .regions import make_pool, overlay, regions_geojson, segment_slide, slide_id

    slides = expand(args.slides)
    out = Path(args.output)
    subs = ["regions", "masks", "slides"] + ([] if args.no_overlay else ["overlays"]) + (["tiles"] if args.save_tiles else [])
    for sub in subs:
        (out / sub).mkdir(parents=True, exist_ok=True)
    todo = [p for p in slides if not (args.skip_existing and (out / "slides" / f"{slide_id(p)}.json").exists())]
    for p in slides:
        if p not in todo:
            _emit(args.json, event="slide_skipped", slide=p, id=slide_id(p))
            if not args.json:
                print(f"{slide_id(p)}: already done, skipped")
    failed, t_all = 0, time.time()
    if todo:
        _emit(args.json, event="status", message="Loading UNI2-h")
        try:
            clf = TumourClassifier(args.device, uni2=args.uni2)
        except WeightsError as e:
            _emit(args.json, event="error", error=str(e))
            print(f"error: {e}", file=sys.stderr)
            return 2
        with make_pool(args.workers) as pool:
            for k, path in enumerate(todo):
                sid = slide_id(path)
                t0 = time.time()
                _emit(args.json, event="slide_start", slide=path, id=sid, index=k, total=len(todo))
                last = [0.0]

                def progress(done, total, path=path):
                    if args.json and (time.time() - last[0] > 0.25 or done == total):
                        last[0] = time.time()
                        _emit(True, event="progress", slide=path, done=done, total=total)
                try:
                    r = segment_slide(path, clf, pool, args.stride_um, args.threshold, args.smooth_um,
                                      args.min_area_mm2, args.min_hole_mm2, args.batch, args.mpp, progress)
                    files = {"regions": out / "regions" / f"{sid}.geojson", "mask": out / "masks" / f"{sid}.png",
                             "slide": out / "slides" / f"{sid}.json"}
                    files["regions"].write_text(json.dumps(regions_geojson(r)))
                    cv2.imwrite(str(files["mask"]), r.mask)
                    if not args.no_overlay:
                        files["overlay"] = out / "overlays" / f"{sid}.jpg"
                        cv2.imwrite(str(files["overlay"]), overlay(r), [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if args.save_tiles:
                        files["tiles"] = out / "tiles" / f"{sid}.csv"
                        np.savetxt(files["tiles"], np.column_stack([r.tile_xy, np.round(r.tile_prob, 4)]),
                                   fmt=["%d", "%d", "%.4f"], delimiter=",", header="x,y,tumour_probability", comments="")
                    dt = time.time() - t0
                    row = {"slide": sid, "tumour_mm2": round(r.tumour_mm2, 3), "tissue_mm2": round(r.tissue_mm2, 3),
                           "tumour_fraction": round(r.fraction, 4), "regions": len(r.regions), "tiles": r.tiles,
                           "mpp": r.mpp, "stride_um": args.stride_um, "threshold": args.threshold,
                           "seconds": round(dt, 1), "path": path, "version": __version__}
                    files["slide"].write_text(json.dumps(row, indent=2))
                    _emit(args.json, event="slide_done", slide=path, id=sid, seconds=round(dt, 1),
                          tumour_mm2=row["tumour_mm2"], tissue_mm2=row["tissue_mm2"], fraction=row["tumour_fraction"],
                          regions=row["regions"], files={k: str(v) for k, v in files.items()})
                    if not args.json:
                        print(f"{sid}: {dt:.1f}s | tumour {r.tumour_mm2:.1f} mm2 of {r.tissue_mm2:.1f} mm2 tissue "
                              f"({100 * r.fraction:.0f}%), {len(r.regions)} regions", flush=True)
                except Exception as e:
                    failed += 1
                    logging.exception("%s failed", path)
                    _emit(args.json, event="slide_failed", slide=path, id=sid, error=str(e))
    summary = write_summary(out)
    _emit(args.json, event="done", slides=len(todo), skipped=len(slides) - len(todo), failed=failed,
          seconds=round(time.time() - t_all, 1), summary=str(summary))
    if not args.json:
        print(f"{len(todo)} slides, {failed} failed, {time.time() - t_all:.1f}s; summary: {summary}")
    return 1 if failed else 0


def _hf_user():
    try:
        from huggingface_hub import whoami
        return whoami().get("name")
    except Exception:
        return None


def _uni2_access() -> tuple[bool, str]:
    try:
        from huggingface_hub import auth_check

        from .model import UNI2_REPO
        auth_check(UNI2_REPO)
        return True, "access granted"
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name:
            return False, "access to UNI2-h not granted yet: request it on its Hugging Face page"
        if "NotFound" in name or "Repository" in name or "401" in str(e):
            return False, "not signed in, or the token cannot see UNI2-h"
        return False, f"{name}: {e}"


def cmd_info(args) -> int:
    from .model import UNI2_URL, WeightsError, find_uni2

    info = {"tepseg": __version__, "python": sys.version.split()[0], "platform": sys.platform}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2 ** 30, 1), cuda_version=torch.version.cuda)
    except ImportError:
        info.update(torch=None, cuda=False)
    try:
        info["uni2"] = str(find_uni2(download=False))
    except WeightsError:
        info["uni2"] = None
    info["hf_user"] = _hf_user()
    if not info["uni2"] and info["hf_user"]:
        info["uni2_access"], info["uni2_message"] = _uni2_access()
    info["uni2_page"] = UNI2_URL
    print(json.dumps(info, indent=2))
    return 0


def cmd_login(args) -> int:
    """Store a Hugging Face token (read from stdin, never echoed) and check access to UNI2-h."""
    from huggingface_hub import login

    token = args.token or (sys.stdin.readline().strip() if not sys.stdin.isatty() else __import__("getpass").getpass("Hugging Face token: "))
    if not token:
        print(json.dumps({"ok": False, "message": "no token given"}))
        return 2
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as e:
        print(json.dumps({"ok": False, "message": f"token rejected: {type(e).__name__}"}))
        return 2
    ok, msg = _uni2_access()
    print(json.dumps({"ok": ok, "user": _hf_user(), "message": msg}))
    return 0 if ok else 3


def cmd_fetch(args) -> int:
    from .model import WeightsError, find_uni2

    try:
        p = find_uni2(args.uni2)
    except WeightsError as e:
        print(json.dumps({"ok": False, "message": str(e)}))
        return 2
    print(json.dumps({"ok": True, "path": str(p)}))
    return 0


def cmd_import(args) -> int:
    """Check a local UNI2-h pytorch_model.bin (it must load into the architecture) and keep a copy."""
    import shutil

    from .model import imported_uni2, load_uni2

    src = Path(args.file)
    src = src / "pytorch_model.bin" if src.is_dir() else src
    try:
        load_uni2(src)
    except Exception as e:
        print(json.dumps({"ok": False, "message": f"not UNI2-h weights: {type(e).__name__}: {e}"}))
        return 2
    dst = imported_uni2()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(json.dumps({"ok": True, "path": str(dst)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tepseg", description="Tumour epithelium regions in H&E whole-slide images "
                                 "(UNI2-h + MLP head)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="segment tumour regions on slides or folders of slides")
    r.add_argument("slides", nargs="+")
    r.add_argument("-o", "--output", default="tepseg-output")
    r.add_argument("--stride-um", type=float, default=128.0, help="field spacing: 128 standard, 256 fast (~4x fewer fields)")
    r.add_argument("--threshold", type=float, default=0.5, help="tumour probability cut-off")
    r.add_argument("--smooth-um", type=float, default=100.0, help="smoothing of the probability map")
    r.add_argument("--min-area-mm2", type=float, default=0.5, help="smaller tumour regions are dropped")
    r.add_argument("--min-hole-mm2", type=float, default=0.1, help="smaller holes inside regions are filled")
    r.add_argument("--workers", type=int, default=8, help="slide-reading processes")
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--device", default="cuda")
    r.add_argument("--mpp", type=float, default=None, help="pixel size (um) for slides without metadata")
    r.add_argument("--uni2", default=None, help="local UNI2-h pytorch_model.bin (or its folder)")
    r.add_argument("--save-tiles", action="store_true", help="also write each field's probability (tiles/<id>.csv)")
    r.add_argument("--no-overlay", action="store_true", help="skip the review images")
    r.add_argument("--skip-existing", action="store_true", help="skip slides already in OUT/slides")
    r.add_argument("--json", action="store_true", help="JSON-lines progress on stdout")
    sub.add_parser("info", help="versions, GPU and UNI2-h status as JSON")
    lg = sub.add_parser("login", help="store a Hugging Face token (from stdin) and check UNI2-h access")
    lg.add_argument("--token", default=None, help=argparse.SUPPRESS)
    f = sub.add_parser("fetch-weights", help="download UNI2-h (needs access and a signed-in token)")
    f.add_argument("--uni2", default=None, help=argparse.SUPPRESS)
    im = sub.add_parser("import-weights", help="use a local UNI2-h pytorch_model.bin")
    im.add_argument("file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "timm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return {"run": cmd_run, "info": cmd_info, "login": cmd_login, "fetch-weights": cmd_fetch,
            "import-weights": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
