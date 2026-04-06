from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from app.ml._compat import _patch_torchvision_functional_tensor


_patch_torchvision_functional_tensor()

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


ML_ROOT = Path(__file__).resolve().parent


class RealEsrGan:
    def __init__(self) -> None:
        self.model_path = ML_ROOT / "weights" / "realesrgan" / "RealESRGAN_x2plus.pth"
        self.device = torch.device("cuda")
        self.scale = 2
        self.tile = 0
        self.tile_pad = 10
        self.pre_pad = 0

    def _reverse_color_channels(self, image_np: np.ndarray) -> np.ndarray:
        return image_np[:, :, ::-1]

    def _load_model(self) -> RealESRGANer:
        if not self.model_path.exists():
            raise FileNotFoundError(f"重みファイルが見つかりません: {self.model_path}")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )

        return RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=torch.cuda.is_available(),
            device=self.device,
        )

    @torch.inference_mode()
    def processing(
        self,
        image_np: np.ndarray,
        outscale: float = 2,
    ) -> np.ndarray:
        model = self._load_model()
        bgr_image_np = self._reverse_color_channels(image_np=image_np)
        output_bgr_image_np, _ = model.enhance(bgr_image_np, outscale=outscale)
        output_pil = self._reverse_color_channels(image_np=output_bgr_image_np)
        return np.array(output_pil)
