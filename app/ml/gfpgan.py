from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
from app.ml._compat import _patch_torchvision_functional_tensor

_patch_torchvision_functional_tensor()

from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import torch
from facexlib.detection.retinaface import RetinaFace
from facexlib.parsing.parsenet import ParseNet
from facexlib.utils import face_restoration_helper as face_helper_module
from torchvision.transforms.functional import normalize

from app.ml.models.gfpgan import load_gfpgan_model


ML_ROOT = Path(__file__).resolve().parent


class Gfpgan:
    def __init__(self, weight_bytes: BytesIO, resnet_weight_bytes: BytesIO, parsing_wight_bytes: BytesIO) -> None:
        self.weight_bytes = weight_bytes
        self.resnet_weight_bytes = resnet_weight_bytes
        self.parsing_wight_bytes = parsing_wight_bytes

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
                self.resnet_weight_bytes,
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
                self.parsing_wight_bytes,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(load_net, strict=True)
            model.eval()
            return model.to(device)

        face_helper_module.init_detection_model = custom_init_detection_model
        face_helper_module.init_parsing_model = custom_init_parsing_model

    def _load_model(self) -> torch.nn.Module:
        self._patch_facexlib_initializers()
        return load_gfpgan_model(
            weight_bytes=self.weight_bytes,
            device=self.device,
            channel_multiplier=self.channel_multiplier,
        )

    def _build_face_helper(self) -> FaceRestoreHelper:
        return FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=str(ML_ROOT / "weights"),
        )

    @torch.inference_mode()
    def _enhance_image(
        self,
        model: torch.nn.Module,
        image_np: np.ndarray,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        face_helper = self._build_face_helper()
        face_helper.clean_all()

        face_helper.read_image(image_np)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face,
            eye_dist_threshold=5,
        )
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = model(cropped_face_t, return_rgb=False)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f"\tFailed inference for GFPGAN: {error}.")
                restored_face = cropped_face

            face_helper.add_restored_face(restored_face.astype("uint8"))

        if paste_back:
            if not face_helper.restored_faces:
                raise RuntimeError("restored_img is None")

            face_helper.get_inverse_affine(None)
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=None)
            if restored_img is None:
                raise RuntimeError("restored_img is None")
            return restored_img

        if not face_helper.restored_faces:
            raise RuntimeError("No face detected")
        return face_helper.restored_faces[0]
    
    def _reverse_color_channels(self, image_np: np.ndarray) -> np.ndarray:
        return image_np[:, :, ::-1]

    def processing(
        self,
        image_np: np.ndarray,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        model = self._load_model()
        bgr_image_np = self._reverse_color_channels(image_np=image_np) # RGBからBGRに変換
        restored_bgr_image_np = self._enhance_image(
            model=model,
            image_np=bgr_image_np,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return self._reverse_color_channels(image_np=restored_bgr_image_np) # BRGに戻す
