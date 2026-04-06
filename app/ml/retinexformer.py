from __future__ import annotations

from pathlib import Path
import yaml

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.vendor.retinexformer.basicsr.models.archs.MST_Plus_Plus_arch import MST_Plus_Plus


ML_ROOT = Path(__file__).resolve().parent


class Retinexformer:
    def __init__(self) -> None:
        self.weight_path = ML_ROOT / "weights" / "retinexformer" / "MST_Plus_Plus_8x1150.pth"
        self.opt_path = ML_ROOT / "configs" / "retinexformer" / "MST_Plus_Plus_NTIRE_8x1150.yml"
        self.device = torch.device("cuda")

    def _load_np_as_tensor(self, image_np: np) -> torch.Tensor:
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor

    def _tensor_to_numpy(self, image_tensor: torch.Tensor) -> Image.Image:
        image_tensor = image_tensor.detach().cpu().clamp(0, 1)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
        image_np = (image_np * 255.0).round().astype(np.uint8)
        return image_np

    def _load_yaml_opt(self) -> dict:
        with open(self.opt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_network_from_yaml(self, opt: dict) -> torch.nn.Module:
        network_g = opt["network_g"]
        network_type = network_g["type"]

        if network_type != "MST_Plus_Plus":
            raise ValueError(f"Unexpected network type: {network_type}")

        model = MST_Plus_Plus(
            in_channels=network_g["in_channels"],
            out_channels=network_g["out_channels"],
            n_feat=network_g["n_feat"],
            stage=network_g["stage"],
        )
        return model

    def _load_model(self) -> torch.nn.Module:
        opt = self._load_yaml_opt()
        model = self._build_network_from_yaml(opt=opt)

        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=True)

        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(cleaned_state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def _enhance_image(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        factor: int = 4,
    ) -> torch.Tensor:
        _, _, h, w = input_tensor.shape

        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        if pad_h != 0 or pad_w != 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        output = model(input_tensor)
        output = output[:, :, :h, :w]
        return output
    
    def processing(self, image_np: np.ndarray) -> np.ndarray:
        model = self._load_model()
        image_tensor = self._load_np_as_tensor(image_np=image_np).to(device=self.device)
        output_tensor = self._enhance_image(model=model, input_tensor=image_tensor, factor=4)
        return self._tensor_to_numpy(image_tensor=output_tensor)
