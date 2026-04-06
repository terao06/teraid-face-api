from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from facexlib.detection.retinaface import RetinaFace
from facexlib.parsing.parsenet import ParseNet
from facexlib.utils import face_restoration_helper as face_helper_module
from app.ml._compat import _patch_torchvision_functional_tensor


_patch_torchvision_functional_tensor()
from gfpgan import GFPGANer


ML_ROOT = Path(__file__).resolve().parent


class Gfpgan:
    def __init__(self) -> None:
        self.weight_path = ML_ROOT / "weights" / "gfpgan" / "GFPGANv.pth"
        self.detection_model_path = ML_ROOT / "weights" / "gfpgan" / "detection_Resnet50_Final.pth"
        self.parsing_model_path = ML_ROOT / "weights" / "gfpgan" / "parsing_parsenet.pth"
        self.device = torch.device("cuda")
        self.upscale = 1
        self.arch = "clean"
        self.channel_multiplier = 2

    def _patch_facexlib_initializers(self) -> None:
        def custom_init_detection_model(
            model_name: str,
            half: bool = False,
            device: str | torch.device = "cuda",
            model_rootpath: str | Path | None = None,
        ):
            if model_name != "retinaface_resnet50":
                raise NotImplementedError(f"{model_name} is not implemented in this class.")

            model = RetinaFace(network_name="resnet50", half=half, device=device)
            load_net = torch.load(
                self.detection_model_path,
                map_location=lambda storage, loc: storage,
            )
            for key, value in deepcopy(load_net).items():
                if key.startswith("module."):
                    load_net[key[7:]] = value
                    load_net.pop(key)

            model.load_state_dict(load_net, strict=True)
            model.eval()
            return model.to(device)

        def custom_init_parsing_model(
            model_name: str = "parsenet",
            half: bool = False,
            device: str | torch.device = "cuda",
            model_rootpath: str | Path | None = None,
        ):
            if half:
                raise NotImplementedError("half=True is not implemented in this class.")
            if model_name != "parsenet":
                raise NotImplementedError(f"{model_name} is not implemented in this class.")

            model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
            load_net = torch.load(
                self.parsing_model_path,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(load_net, strict=True)
            model.eval()
            return model.to(device)

        face_helper_module.init_detection_model = custom_init_detection_model
        face_helper_module.init_parsing_model = custom_init_parsing_model

    def _load_model(self) -> GFPGANer:
        self._patch_facexlib_initializers()
        return GFPGANer(
            model_path=str(self.weight_path),
            upscale=self.upscale,
            arch=self.arch,
            channel_multiplier=self.channel_multiplier,
            bg_upsampler=None,
            device=self.device,
        )

    def _reverse_color_channels(self, image_np: np.ndarray) -> np.ndarray:
        return image_np[:, :, ::-1]

    @torch.inference_mode()
    def _enhance_image(
        self,
        model: GFPGANer,
        image_np: np.ndarray,
        *,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        _, restored_faces, restored_img = model.enhance(
            image_np,
            has_aligned=has_aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )

        if paste_back:
            if restored_img is None:
                raise RuntimeError("restored_img is None")
            return restored_img

        if not restored_faces:
            raise RuntimeError("No face detected")
        return restored_faces[0]

    def processing(
        self,
        image_np: np.ndarray,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        model = self._load_model()
        bgr_image_np = self._reverse_color_channels(image_np=image_np) # RGBからBGRに変換
        restored_bgr_image_np = self._enhance_image(
            model=model,
            image_np=bgr_image_np,
            has_aligned=has_aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return self._reverse_color_channels(image_np=restored_bgr_image_np) # BRGに戻す
