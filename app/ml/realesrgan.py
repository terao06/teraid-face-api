from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from app.ml._compat import patch_torchvision_functional_tensor


patch_torchvision_functional_tensor()

from app.ml.models.realesrgan import load_realesrgan_model


ML_ROOT = Path(__file__).resolve().parent


class RealEsrGan:
    def __init__(self, weight_bytes: BytesIO) -> None:
        self.weight_bytes = weight_bytes
        self.device = torch.device("cuda")
        self.scale = 2

    def _reverse_color_channels(self, image_np: np.ndarray) -> np.ndarray:
        return image_np[:, :, ::-1]

    def _load_np_as_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        image_np = np.ascontiguousarray(image_np)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.float() / 255.0

    def _tensor_to_numpy(self, image_tensor: torch.Tensor) -> np.ndarray:
        image_tensor = image_tensor.detach().cpu().clamp(0, 1)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
        return (image_np * 255.0).round().astype(np.uint8)

    def _load_model(self) -> torch.nn.Module:
        return load_realesrgan_model(
            weight_bytes=self.weight_bytes,
            device=self.device,
            scale=self.scale,
        )

    @torch.inference_mode()
    def _enhance_image(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        *,
        outscale: float,
    ) -> torch.Tensor:
        _, _, h, w = input_tensor.shape
        pad_h = (self.scale - h % self.scale) % self.scale
        pad_w = (self.scale - w % self.scale) % self.scale

        if pad_h != 0 or pad_w != 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        output_tensor = model(input_tensor)
        output_tensor = output_tensor[
            :,
            :,
            : (h + pad_h) * self.scale,
            : (w + pad_w) * self.scale,
        ]
        if pad_h != 0 or pad_w != 0:
            output_tensor = output_tensor[:, :, : h * self.scale, : w * self.scale]

        if outscale != self.scale:
            output_tensor = F.interpolate(
                output_tensor,
                scale_factor=outscale / self.scale,
                mode="bicubic",
                align_corners=False,
            )
        return output_tensor

    @torch.inference_mode()
    def processing(
        self,
        image_np: np.ndarray,
        outscale: float = 2,
    ) -> np.ndarray:
        model = self._load_model()
        bgr_image_np = self._reverse_color_channels(image_np=image_np)
        input_tensor = self._load_np_as_tensor(image_np=bgr_image_np).to(device=self.device)
        output_tensor = self._enhance_image(model=model, input_tensor=input_tensor, outscale=outscale)
        output_bgr_image_np = self._tensor_to_numpy(image_tensor=output_tensor)
        return self._reverse_color_channels(image_np=output_bgr_image_np)
