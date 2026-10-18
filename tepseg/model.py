"""UNI2-h backbone and the TEPSEG tumour head.

UNI2-h (MahmoodLab, ViT-giant) is gated on Hugging Face: each user requests
access at https://huggingface.co/MahmoodLab/UNI2-h and signs in once
(``tepseg login``). TEPSEG never redistributes it. A local copy of
``pytorch_model.bin`` can be used instead (``--uni2 PATH``, ``TEPSEG_UNI2`` or
``tepseg import-weights``).
"""
from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Optional

import numpy as np

UNI2_REPO = "MahmoodLab/UNI2-h"
UNI2_FILE = "pytorch_model.bin"
UNI2_URL = f"https://huggingface.co/{UNI2_REPO}"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class WeightsError(RuntimeError):
    """UNI2-h weights are not available (no access, not signed in, offline)."""


def home() -> Path:
    return Path(os.environ.get("TEPSEG_HOME", Path.home() / ".cache" / "tepseg"))


def imported_uni2() -> Path:
    return home() / "UNI2-h" / UNI2_FILE


def find_uni2(explicit: Optional[str] = None, download: bool = True) -> Path:
    """Path of UNI2-h's pytorch_model.bin: explicit path, TEPSEG_UNI2, imported copy, HF cache, or download."""
    for cand in (explicit, os.environ.get("TEPSEG_UNI2")):
        if cand:
            p = Path(cand)
            p = p / UNI2_FILE if p.is_dir() else p
            if not p.is_file():
                raise WeightsError(f"UNI2-h weights not found: {p}")
            return p
    if imported_uni2().is_file():
        return imported_uni2()
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(UNI2_REPO, UNI2_FILE)
    if isinstance(cached, str):
        return Path(cached)
    if not download:
        raise WeightsError("UNI2-h is not downloaded yet")
    try:
        return Path(hf_hub_download(UNI2_REPO, UNI2_FILE))
    except Exception as e:  # gated, not signed in, offline
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e) or "Unauthorized" in name:
            raise WeightsError(
                f"UNI2-h is gated. Request access at {UNI2_URL}, then sign in with `tepseg login` "
                "(QuPath: Set up backend > Hugging Face token), or point --uni2 at a local pytorch_model.bin.") from e
        raise WeightsError(f"Could not download UNI2-h ({name}: {e}). Check the connection, or use --uni2.") from e


def build_uni2():
    """UNI2-h architecture (timm), without weights."""
    import timm
    import torch

    return timm.create_model(
        "vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)


def load_uni2(path: Path):
    import torch

    net = build_uni2()
    state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    net.load_state_dict(state, strict=True)
    return net


class TumourClassifier:
    """UNI2-h features + MLP head: (n, 3, 224, 224) tiles in 0-255 -> tumour probability (n,)."""

    def __init__(self, device: str = "cuda", uni2: Optional[str] = None):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.half = device.startswith("cuda")
        self.uni2_path = find_uni2(uni2)
        net = load_uni2(self.uni2_path)
        self.net = (net.half() if self.half else net).to(device).eval()
        h = np.load(files("tepseg") / "data" / "tepseg_head_v1.npz")
        t = lambda k: torch.as_tensor(h[k], device=device).float()
        self.layers = [(t("W1"), t("b1")), (t("W2"), t("b2")), (t("W3"), t("b3"))]
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1) * 255

    def features(self, tiles):
        import torch

        with torch.inference_mode():
            x = (tiles.float() - self.mean) / self.std
            return self.net(x.half() if self.half else x).float()

    def head(self, f):
        import torch

        (W1, b1), (W2, b2), (W3, b3) = self.layers
        x = torch.relu(f @ W1.T + b1)
        x = torch.relu(x @ W2.T + b2)
        return torch.softmax(x @ W3.T + b3, 1)[:, 1]

    def __call__(self, tiles):
        import torch

        with torch.inference_mode():
            return self.head(self.features(tiles))
