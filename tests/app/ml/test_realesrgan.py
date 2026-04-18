from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
import numpy as np
import torch

from app.ml import realesrgan as realesrgan_module
from app.ml.realesrgan import RealEsrGan


TEST_IMAGE_PATH = Path("tests/test_data/images/realesrgan/test_face.png")
RESULT_IMAGE_PATH = Path("tests/test_data/images/realesrgan/result_face.png")
WEIGHT_PATH = Path("tests/test_data/s3/buckets/weights/realesrgan/RealESRGAN_x2plus.pth")


class TestRealEsrGan:
    @staticmethod
    def _dummy_weight_bytes() -> BytesIO:
        return BytesIO(b"dummy-weight-bytes")

    @staticmethod
    def _load_weight_bytes(path: Path) -> BytesIO:
        return BytesIO(path.read_bytes())

    def test_reverse_color_channels_reverses_rgb_and_bgr_order(self) -> None:
        realesrgan = RealEsrGan(weight_bytes=self._dummy_weight_bytes())
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        output = realesrgan._reverse_color_channels(image_np)

        assert np.array_equal(output, image_np[:, :, ::-1])

    @patch.object(realesrgan_module, "load_realesrgan_model")
    def test_load_model_builds_realesrganer_with_expected_args(
        self,
        mock_load_realesrgan_model: MagicMock,
    ) -> None:
        realesrgan = RealEsrGan(weight_bytes=self._dummy_weight_bytes())
        model_instance = object()
        mock_load_realesrgan_model.return_value = model_instance

        model = realesrgan._load_model()

        assert model is model_instance
        mock_load_realesrgan_model.assert_called_once_with(
            weight_bytes=realesrgan.weight_bytes,
            device=realesrgan.device,
            scale=realesrgan.scale,
        )

    def test_enhance_image_pads_to_scale_and_crops_back(self) -> None:
        realesrgan = RealEsrGan(weight_bytes=self._dummy_weight_bytes())
        input_tensor = torch.arange(1 * 3 * 3 * 5, dtype=torch.float32).reshape(1, 3, 3, 5)
        captured = {}

        class DummyModel:
            def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                captured["shape"] = tuple(tensor.shape)
                upscaled = tensor.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
                return upscaled + 1

        output = realesrgan._enhance_image(
            model=DummyModel(),
            input_tensor=input_tensor,
            outscale=2,
        )

        assert captured["shape"] == (1, 3, 4, 6)
        assert output.shape == (1, 3, 6, 10)
        expected = (input_tensor.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)) + 1
        assert torch.equal(output, expected)

    @patch.object(RealEsrGan, "_load_model")
    def test_processing_loads_model_and_enhances_image(self, mock_load_model: MagicMock) -> None:
        realesrgan = RealEsrGan(weight_bytes=self._dummy_weight_bytes())
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        class DummyModel:
            def __call__(self, image: torch.Tensor) -> torch.Tensor:
                assert tuple(image.shape) == (1, 3, 2, 2)
                expected_bgr = image_np[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
                assert np.allclose(image.detach().cpu().numpy(), expected_bgr.astype(np.float32))
                return torch.tensor(
                    [
                        [
                            [[23, 26], [29, 32]],
                            [[22, 25], [28, 31]],
                            [[21, 24], [27, 30]],
                        ]
                    ],
                    dtype=torch.float32,
                    device=image.device,
                ) / 255.0

        mock_load_model.return_value = DummyModel()

        output = realesrgan.processing(image_np=image_np, outscale=2)

        expected_rgb = np.array(
            [
                [[21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32]],
            ],
            dtype=np.uint8,
        )
        assert np.array_equal(output, expected_rgb)

    def test_processing(self) -> None:
        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        image_np = RealEsrGan(
            weight_bytes=self._load_weight_bytes(WEIGHT_PATH)
        ).processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
