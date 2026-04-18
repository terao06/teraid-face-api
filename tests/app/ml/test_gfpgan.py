from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import torch

from app.ml import gfpgan as gfpgan_module
from app.ml.gfpgan import Gfpgan


TEST_IMAGE_PATH = Path("tests/test_data/images/gfpgan/test_face.png")
RESULT_IMAGE_PATH = Path("tests/test_data/images/gfpgan/result_face.png")
WEIGHTS_DIR = Path("tests/test_data/s3/buckets/weights/gfpgan")
GFPGAN_WEIGHT_PATH = WEIGHTS_DIR / "GFPGANv.pth"
RESNET_WEIGHT_PATH = WEIGHTS_DIR / "detection_Resnet50_Final.pth"
PARSENET_WEIGHT_PATH = WEIGHTS_DIR / "parsing_parsenet.pth"


class TestGfpgan:
    @staticmethod
    def _dummy_weight_bytes() -> BytesIO:
        return BytesIO(b"dummy-weight-bytes")

    def _build_gfpgan(self) -> Gfpgan:
        return Gfpgan(
            weight_bytes=self._dummy_weight_bytes(),
            resnet_weight_bytes=self._dummy_weight_bytes(),
            parsing_wight_bytes=self._dummy_weight_bytes(),
        )

    @staticmethod
    def _load_weight_bytes(path: Path) -> BytesIO:
        return BytesIO(path.read_bytes())

    def test_patch_facexlib_initializers_replaces_helpers(self) -> None:
        gfpgan = self._build_gfpgan()
        original_detection = object()
        original_parsing = object()

        with (
            patch.object(
                gfpgan_module.face_helper_module,
                "init_detection_model",
                original_detection,
                create=True,
            ),
            patch.object(
                gfpgan_module.face_helper_module,
                "init_parsing_model",
                original_parsing,
                create=True,
            ),
        ):
            gfpgan._patch_facexlib_initializers()
            assert gfpgan_module.face_helper_module.init_detection_model is not original_detection
            assert gfpgan_module.face_helper_module.init_parsing_model is not original_parsing

    @patch.object(torch, "load")
    @patch.object(gfpgan_module, "RetinaFace")
    def test_custom_detection_initializer_loads_weights_and_strips_module_prefix(
        self,
        mock_retina_face: MagicMock,
        mock_torch_load: MagicMock,
    ) -> None:
        gfpgan = self._build_gfpgan()
        captured = {}

        class DummyRetinaFace:
            def __init__(self, network_name, half, device):
                captured["network_name"] = network_name
                captured["half"] = half
                captured["device"] = device

            def load_state_dict(self, state_dict, strict):
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def eval(self):
                captured["eval_called"] = True
                return self

            def to(self, device):
                captured["to_device"] = device
                return self

        mock_retina_face.side_effect = DummyRetinaFace
        mock_torch_load.return_value = {"module.weight": torch.tensor([1.0])}

        gfpgan._patch_facexlib_initializers()
        model = gfpgan_module.face_helper_module.init_detection_model(
            "retinaface_resnet50", device="cuda:0"
        )

        assert isinstance(model, DummyRetinaFace)
        assert captured["network_name"] == "resnet50"
        assert captured["half"] is False
        assert captured["device"] == "cuda:0"
        assert captured["strict"] is True
        assert captured["eval_called"] is True
        assert captured["to_device"] == "cuda:0"
        assert list(captured["state_dict"].keys()) == ["weight"]

    @pytest.mark.parametrize(
        ("call_initializer", "match"),
        [
            (
                lambda: gfpgan_module.face_helper_module.init_detection_model("unsupported"),
                "is not implemented in this class",
            ),
            (
                lambda: gfpgan_module.face_helper_module.init_parsing_model(half=True),
                "half=True",
            ),
            (
                lambda: gfpgan_module.face_helper_module.init_parsing_model("unsupported"),
                "is not implemented in this class",
            ),
        ],
    )
    def test_custom_initializers_reject_invalid_args(
        self, call_initializer, match: str
    ) -> None:
        gfpgan = self._build_gfpgan()
        gfpgan._patch_facexlib_initializers()

        with pytest.raises(NotImplementedError, match=match):
            call_initializer()

    @pytest.mark.parametrize(
        ("paste_back", "restored_faces", "restored_img", "match"),
        [
            (True, [], None, "restored_img is None"),
            (False, [], None, "No face detected"),
        ],
    )
    def test_enhance_image_raises_for_missing_outputs(
        self,
        paste_back: bool,
        restored_faces: list[np.ndarray],
        restored_img: np.ndarray | None,
        match: str,
    ) -> None:
        gfpgan = self._build_gfpgan()
        image_np = np.zeros((1, 1, 3), dtype=np.uint8)

        class DummyFaceHelper:
            def __init__(self) -> None:
                self.cropped_faces = [image_np]
                self.restored_faces = restored_faces

            def clean_all(self) -> None:
                return None

            def read_image(self, image) -> None:
                assert image is image_np

            def get_face_landmarks_5(self, **kwargs) -> None:
                return None

            def align_warp_face(self) -> None:
                return None

            def add_restored_face(self, face) -> None:
                return None

            def get_inverse_affine(self, matrix) -> None:
                return None

            def paste_faces_to_input_image(self, upsample_img=None):
                return restored_img

        class DummyModel:
            def __call__(self, image, return_rgb=False):
                raise RuntimeError("mock inference failure")

        gfpgan._build_face_helper = lambda: DummyFaceHelper()

        with pytest.raises(RuntimeError, match=match):
            gfpgan._enhance_image(
                model=DummyModel(),
                image_np=image_np,
                paste_back=paste_back,
            )

    @pytest.mark.parametrize(
        ("paste_back", "restored_faces", "restored_img", "expected"),
        [
            (
                True,
                [],
                np.full((2, 2, 3), 7, dtype=np.uint8),
                np.full((2, 2, 3), 7, dtype=np.uint8),
            ),
            (
                False,
                [np.full((4, 4, 3), 9, dtype=np.uint8)],
                None,
                np.full((4, 4, 3), 9, dtype=np.uint8),
            ),
        ],
    )
    def test_enhance_image_returns_expected_output(
        self,
        paste_back: bool,
        restored_faces: list[np.ndarray],
        restored_img: np.ndarray | None,
        expected: np.ndarray,
    ) -> None:
        gfpgan = self._build_gfpgan()
        image_np = np.zeros(expected.shape, dtype=np.uint8)
        captured = {}

        class DummyFaceHelper:
            def __init__(self) -> None:
                self.cropped_faces = [image_np]
                self.restored_faces = restored_faces

            def clean_all(self) -> None:
                return None

            def read_image(self, image) -> None:
                assert image is image_np

            def get_face_landmarks_5(self, **kwargs) -> None:
                captured["landmarks_kwargs"] = kwargs

            def align_warp_face(self) -> None:
                return None

            def add_restored_face(self, face) -> None:
                self.restored_faces.append(face)

            def get_inverse_affine(self, matrix) -> None:
                captured["inverse_affine"] = matrix

            def paste_faces_to_input_image(self, upsample_img=None):
                captured["upsample_img"] = upsample_img
                return restored_img

        class DummyModel:
            def __call__(self, image, return_rgb=False):
                captured["image_shape"] = tuple(image.shape)
                captured["return_rgb"] = return_rgb
                output = torch.full_like(image, -0.94509804)
                return (output,)

        gfpgan._build_face_helper = lambda: DummyFaceHelper()

        output = gfpgan._enhance_image(
            model=DummyModel(),
            image_np=image_np,
            only_center_face=True,
            paste_back=paste_back,
        )

        assert np.array_equal(output, expected)
        assert captured["image_shape"] == (1, 3, *expected.shape[:2])
        assert captured["return_rgb"] is False
        assert captured["landmarks_kwargs"] == {
            "only_center_face": True,
            "eye_dist_threshold": 5,
        }
        if paste_back:
            assert captured["inverse_affine"] is None
            assert captured["upsample_img"] is None

    @patch.object(torch, "load")
    @patch.object(gfpgan_module, "ParseNet")
    def test_custom_parsing_initializer_loads_weights(
        self,
        mock_parse_net: MagicMock,
        mock_torch_load: MagicMock,
    ) -> None:
        gfpgan = self._build_gfpgan()
        captured = {}

        class DummyParseNet:
            def __init__(self, in_size, out_size, parsing_ch):
                captured["in_size"] = in_size
                captured["out_size"] = out_size
                captured["parsing_ch"] = parsing_ch

            def load_state_dict(self, state_dict, strict):
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def eval(self):
                captured["eval_called"] = True
                return self

            def to(self, device):
                captured["to_device"] = device
                return self

        mock_parse_net.side_effect = DummyParseNet
        mock_torch_load.return_value = {"weight": torch.tensor([2.0])}

        gfpgan._patch_facexlib_initializers()
        model = gfpgan_module.face_helper_module.init_parsing_model("parsenet", device="cuda:1")

        assert isinstance(model, DummyParseNet)
        assert captured["in_size"] == 512
        assert captured["out_size"] == 512
        assert captured["parsing_ch"] == 19
        assert captured["strict"] is True
        assert captured["eval_called"] is True
        assert captured["to_device"] == "cuda:1"
        assert list(captured["state_dict"].keys()) == ["weight"]
        assert torch.equal(captured["state_dict"]["weight"], torch.tensor([2.0]))

    @patch.object(gfpgan_module, "load_gfpgan_model")
    @patch.object(Gfpgan, "_patch_facexlib_initializers")
    def test_load_model_builds_gfpganer_with_expected_args(
        self,
        mock_patch_facexlib_initializers: MagicMock,
        mock_load_gfpgan_model: MagicMock,
    ) -> None:
        gfpgan = self._build_gfpgan()
        model_instance = object()
        mock_load_gfpgan_model.return_value = model_instance

        model = gfpgan._load_model()

        assert model is model_instance
        mock_patch_facexlib_initializers.assert_called_once_with()
        mock_load_gfpgan_model.assert_called_once_with(
            weight_bytes=gfpgan.weight_bytes,
            device=gfpgan.device,
            channel_multiplier=gfpgan.channel_multiplier,
        )

    @patch.object(Gfpgan, "_enhance_image")
    @patch.object(Gfpgan, "_load_model")
    def test_processing_loads_model_and_enhances_image(
        self,
        mock_load_model: MagicMock,
        mock_enhance_image: MagicMock,
    ) -> None:
        gfpgan = self._build_gfpgan()
        image_np = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        expected_bgr = np.array(
            [
                [[5, 15, 25], [35, 45, 55]],
                [[65, 75, 85], [95, 105, 115]],
            ],
            dtype=np.uint8,
        )
        dummy_model = object()
        mock_load_model.return_value = dummy_model

        def fake_enhance_image(model, image_np, **kwargs):
            assert model is dummy_model
            assert np.array_equal(image_np, np.arange(12, dtype=np.uint8).reshape(2, 2, 3)[:, :, ::-1])
            assert kwargs == {
                "only_center_face": False,
                "paste_back": True,
            }
            return expected_bgr

        mock_enhance_image.side_effect = fake_enhance_image

        output = gfpgan.processing(
            image_np=image_np,
            only_center_face=False,
            paste_back=True,
        )

        assert np.array_equal(output, expected_bgr[:, :, ::-1])

    @patch.object(Gfpgan, "_enhance_image")
    @patch.object(Gfpgan, "_load_model")
    def test_processing_uses_default_flags(
        self,
        mock_load_model: MagicMock,
        mock_enhance_image: MagicMock,
    ) -> None:
        gfpgan = self._build_gfpgan()
        image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected_bgr = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))[:, :, ::-1]
        dummy_model = object()
        mock_load_model.return_value = dummy_model

        def fake_enhance_image(model, image_np, **kwargs):
            assert model is dummy_model
            assert kwargs == {
                "only_center_face": False,
                "paste_back": True,
            }
            return expected_bgr

        mock_enhance_image.side_effect = fake_enhance_image

        output = gfpgan.processing(image_np=image_np)

        assert isinstance(output, np.ndarray)
        assert output.shape == image_np.shape
        assert output.dtype == np.uint8
        assert np.array_equal(output, expected_bgr[:, :, ::-1])
    
    def test_processing(self) -> None:
        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        gfpgan = Gfpgan(
            weight_bytes=self._load_weight_bytes(GFPGAN_WEIGHT_PATH),
            resnet_weight_bytes=self._load_weight_bytes(RESNET_WEIGHT_PATH),
            parsing_wight_bytes=self._load_weight_bytes(PARSENET_WEIGHT_PATH),
        )
        image_np = gfpgan.processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
