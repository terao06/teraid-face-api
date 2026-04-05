from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from app.ml import gfpgan as gfpgan_module
from app.ml.gfpgan import Gfpgan


TEST_IMAGE_PATH = Path("tests/app/test_data/test_image/gfpgan/test_face.png")
RESULT_IMAGE_PATH = Path("tests/app/test_data/test_image/gfpgan/result_face.png")


class TestGfpgan:
    def test_patch_facexlib_initializers_replaces_helpers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gfpgan = Gfpgan()
        original_detection = object()
        original_parsing = object()

        monkeypatch.setattr(
            "app.ml.gfpgan.face_helper_module.init_detection_model",
            original_detection,
            raising=False,
        )
        monkeypatch.setattr(
            "app.ml.gfpgan.face_helper_module.init_parsing_model",
            original_parsing,
            raising=False,
        )

        gfpgan._patch_facexlib_initializers()

        assert gfpgan_module.face_helper_module.init_detection_model is not original_detection
        assert gfpgan_module.face_helper_module.init_parsing_model is not original_parsing

    def test_custom_detection_initializer_loads_weights_and_strips_module_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gfpgan = Gfpgan()
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

        monkeypatch.setattr("app.ml.gfpgan.RetinaFace", DummyRetinaFace)
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location: {"module.weight": torch.tensor([1.0])},
        )

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

    def test_custom_detection_initializer_rejects_unsupported_model_name(self) -> None:
        gfpgan = Gfpgan()
        gfpgan._patch_facexlib_initializers()

        with pytest.raises(NotImplementedError, match="is not implemented in this class"):
            gfpgan_module.face_helper_module.init_detection_model("unsupported")

    def test_custom_parsing_initializer_loads_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gfpgan = Gfpgan()
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

        monkeypatch.setattr("app.ml.gfpgan.ParseNet", DummyParseNet)
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location: {"weight": torch.tensor([2.0])},
        )

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

    def test_custom_parsing_initializer_rejects_invalid_args(self) -> None:
        gfpgan = Gfpgan()
        gfpgan._patch_facexlib_initializers()

        with pytest.raises(NotImplementedError, match="half=True"):
            gfpgan_module.face_helper_module.init_parsing_model(half=True)

        with pytest.raises(NotImplementedError, match="is not implemented in this class"):
            gfpgan_module.face_helper_module.init_parsing_model("unsupported")

    def test_load_model_builds_gfpganer_with_expected_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gfpgan = Gfpgan()
        captured = {}

        class DummyGFPGANer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            gfpgan,
            "_patch_facexlib_initializers",
            lambda: captured.setdefault("patched", True),
        )
        monkeypatch.setattr("app.ml.gfpgan.GFPGANer", DummyGFPGANer)

        model = gfpgan._load_model()

        assert isinstance(model, DummyGFPGANer)
        assert captured["patched"] is True
        assert captured["model_path"] == str(gfpgan.weight_path)
        assert captured["upscale"] == gfpgan.upscale
        assert captured["arch"] == gfpgan.arch
        assert captured["channel_multiplier"] == gfpgan.channel_multiplier
        assert captured["bg_upsampler"] is None
        assert captured["device"] == gfpgan.device

    def test_enhance_image_returns_restored_img_when_paste_back_true(self) -> None:
        gfpgan = Gfpgan()
        image_np = np.zeros((2, 2, 3), dtype=np.uint8)
        restored_img = np.full((2, 2, 3), 7, dtype=np.uint8)
        captured = {}

        class DummyModel:
            def enhance(self, image, **kwargs):
                captured["image"] = image
                captured["kwargs"] = kwargs
                return None, [], restored_img

        output = gfpgan._enhance_image(
            model=DummyModel(),
            image_np=image_np,
            has_aligned=True,
            only_center_face=True,
            paste_back=True,
        )

        assert np.array_equal(output, restored_img)
        assert captured["image"] is image_np
        assert captured["kwargs"] == {
            "has_aligned": True,
            "only_center_face": True,
            "paste_back": True,
        }

    def test_enhance_image_raises_when_restored_img_is_none(self) -> None:
        gfpgan = Gfpgan()

        class DummyModel:
            def enhance(self, image, **kwargs):
                return None, [], None

        with pytest.raises(RuntimeError, match="restored_img is None"):
            gfpgan._enhance_image(
                model=DummyModel(),
                image_np=np.zeros((1, 1, 3), dtype=np.uint8),
                paste_back=True,
            )

    def test_enhance_image_returns_first_face_when_paste_back_false(self) -> None:
        gfpgan = Gfpgan()
        restored_face = np.full((4, 4, 3), 9, dtype=np.uint8)

        class DummyModel:
            def enhance(self, image, **kwargs):
                return None, [restored_face], None

        output = gfpgan._enhance_image(
            model=DummyModel(),
            image_np=np.zeros((4, 4, 3), dtype=np.uint8),
            paste_back=False,
        )

        assert np.array_equal(output, restored_face)

    def test_enhance_image_raises_when_no_face_detected(self) -> None:
        gfpgan = Gfpgan()

        class DummyModel:
            def enhance(self, image, **kwargs):
                return None, [], None

        with pytest.raises(RuntimeError, match="No face detected"):
            gfpgan._enhance_image(
                model=DummyModel(),
                image_np=np.zeros((1, 1, 3), dtype=np.uint8),
                paste_back=False,
            )

    def test_processing_loads_model_and_enhances_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gfpgan = Gfpgan()
        image_np = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        expected_bgr = np.array(
            [
                [[5, 15, 25], [35, 45, 55]],
                [[65, 75, 85], [95, 105, 115]],
            ],
            dtype=np.uint8,
        )
        dummy_model = object()
        captured = {}

        monkeypatch.setattr(gfpgan, "_load_model", lambda: dummy_model)

        def fake_enhance_image(model, image_np, **kwargs):
            captured["model"] = model
            captured["image_np"] = image_np
            captured["kwargs"] = kwargs
            return expected_bgr

        monkeypatch.setattr(gfpgan, "_enhance_image", fake_enhance_image)

        output = gfpgan.processing(
            image_np=image_np,
            has_aligned=True,
            only_center_face=False,
            paste_back=True,
        )

        assert np.array_equal(output, expected_bgr[:, :, ::-1])
        assert captured["model"] is dummy_model
        assert np.array_equal(captured["image_np"], image_np[:, :, ::-1])
        assert captured["kwargs"] == {
            "has_aligned": True,
            "only_center_face": False,
            "paste_back": True,
        }

    def test_processing(self) -> None:

        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        image_np = Gfpgan().processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
