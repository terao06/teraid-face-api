from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import torch

from app.ml.retinexformer import Retinexformer
from app.ml.models.retinexformer import MST_Plus_Plus


TEST_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/test_face.png")
RESULT_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/result_face.png")


class TestRetinexformer:
    def test_load_np_as_tensor_converts_hwc_to_nchw(self) -> None:
        former = Retinexformer()
        image_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

        tensor = former._load_np_as_tensor(image_np)

        assert tensor.shape == (1, 3, 2, 2)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))

    def test_tensor_to_numpy_clamps_and_returns_ndarray(self) -> None:
        former = Retinexformer()
        tensor = torch.tensor(
            [[
                [[1.2, 0.5], [0.0, 0.1]],
                [[-0.1, 0.25], [0.5, 0.75]],
                [[0.2, 0.0], [1.0, 0.333]],
            ]],
            dtype=torch.float32,
        )

        image_np = former._tensor_to_numpy(tensor)

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == (2, 2, 3)
        assert image_np.dtype == np.uint8
        assert np.array_equal(image_np[0, 0], np.array([255, 0, 51], dtype=np.uint8))

    def test_build_network_from_yaml_rejects_unexpected_network_type(self) -> None:
        former = Retinexformer()

        with pytest.raises(ValueError, match="Unexpected network type"):
            former._build_network_from_yaml({"network_g": {"type": "unexpected"}})

    def test_build_network_from_yaml_builds_mst_plus_plus(self) -> None:
        former = Retinexformer()

        model = former._build_network_from_yaml(
            {
                "network_g": {
                    "type": "MST_Plus_Plus",
                    "in_channels": 3,
                    "out_channels": 3,
                    "n_feat": 31,
                    "stage": 2,
                }
            }
        )

        assert isinstance(model, MST_Plus_Plus)
        assert model.stage == 2
        assert model.conv_in.in_channels == 3
        assert model.conv_in.out_channels == 31
        assert model.conv_out.in_channels == 31
        assert model.conv_out.out_channels == 3

    @patch.object(torch, "load")
    @patch.object(Retinexformer, "_build_network_from_yaml")
    @patch.object(Retinexformer, "_load_yaml_opt", return_value={"network_g": {}})
    def test_load_model_uses_params_ema_and_strips_module_prefix(
        self,
        _mock_load_yaml_opt: MagicMock,
        mock_build_network_from_yaml: MagicMock,
        mock_torch_load: MagicMock,
    ) -> None:
        former = Retinexformer()
        captured = {}

        class DummyModel:
            def load_state_dict(self, state_dict, strict):
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def to(self, device):
                captured["device"] = device
                return self

            def eval(self):
                captured["eval_called"] = True
                return self

        dummy_model = DummyModel()
        mock_build_network_from_yaml.return_value = dummy_model
        mock_torch_load.return_value = {
            "params_ema": {"module.weight": torch.tensor([1.0])}
        }

        model = former._load_model()

        assert model is dummy_model
        assert captured["strict"] is True
        assert captured["device"] == former.device
        assert captured["eval_called"] is True
        assert list(captured["state_dict"].keys()) == ["weight"]
        _mock_load_yaml_opt.assert_called_once()

    def test_enhance_image_pads_to_factor_and_crops_back(self) -> None:
        former = Retinexformer()
        input_tensor = torch.arange(1 * 3 * 5 * 6, dtype=torch.float32).reshape(1, 3, 5, 6)
        captured = {}

        class DummyModel:
            def __call__(self, tensor):
                captured["shape"] = tensor.shape
                return tensor + 1

        output = former._enhance_image(DummyModel(), input_tensor, factor=4)

        assert captured["shape"] == (1, 3, 8, 8)
        assert output.shape == (1, 3, 5, 6)
        assert torch.equal(output, input_tensor + 1)

    def test_processing(self) -> None:
        test_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        test_image_np = np.asarray(test_image).astype(np.float32) / 255.0

        former = Retinexformer()
        image_np = former.processing(image_np=test_image_np)

        result_image = Image.open(RESULT_IMAGE_PATH).convert("RGB")
        expected = np.asarray(result_image)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
