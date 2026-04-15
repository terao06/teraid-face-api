from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
import numpy as np
import pytest
import torch

from app.ml import realesrgan as realesrgan_module
from app.ml.realesrgan import RealEsrGan


TEST_IMAGE_PATH = Path("tests/test_data/test_image/realesrgan/test_face.png")
RESULT_IMAGE_PATH = Path("tests/test_data/test_image/realesrgan/result_face.png")


class TestRealEsrGan:
    def test_reverse_color_channels_reverses_rgb_and_bgr_order(self) -> None:
        realesrgan = RealEsrGan()
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        output = realesrgan._reverse_color_channels(image_np)

        assert np.array_equal(output, image_np[:, :, ::-1])

    def test_load_model_raises_when_model_path_does_not_exist(self) -> None:
        realesrgan = RealEsrGan()
        realesrgan.model_path = realesrgan.model_path.with_name("missing.pth")

        with pytest.raises(FileNotFoundError, match="missing.pth"):
            realesrgan._load_model()

    @patch.object(torch.cuda, "is_available", return_value=True)
    @patch.object(realesrgan_module, "RealESRGANer")
    @patch.object(realesrgan_module, "RRDBNet")
    def test_load_model_builds_realesrganer_with_expected_args(
        self,
        mock_rrdbnet: MagicMock,
        mock_realesrganer: MagicMock,
        _mock_is_available: MagicMock,
    ) -> None:
        realesrgan = RealEsrGan()

        class DummyPath:
            def exists(self) -> bool:
                return True

            def __str__(self) -> str:
                return "dummy/realesrgan.pth"

        realesrgan.model_path = DummyPath()
        rrdbnet_instance = object()
        realesrganer_instance = object()
        mock_rrdbnet.return_value = rrdbnet_instance
        mock_realesrganer.return_value = realesrganer_instance

        model = realesrgan._load_model()

        assert model is realesrganer_instance
        mock_rrdbnet.assert_called_once_with(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=realesrgan.scale,
        )
        mock_realesrganer.assert_called_once_with(
            scale=realesrgan.scale,
            model_path=str(realesrgan.model_path),
            model=rrdbnet_instance,
            tile=realesrgan.tile,
            tile_pad=realesrgan.tile_pad,
            pre_pad=realesrgan.pre_pad,
            half=True,
            device=realesrgan.device,
        )
        _mock_is_available.assert_called_once()

    @patch.object(RealEsrGan, "_load_model")
    def test_processing_loads_model_and_enhances_image(self, mock_load_model: MagicMock) -> None:
        realesrgan = RealEsrGan()
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        enhanced_bgr = np.array(
            [
                [[21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32]],
            ],
            dtype=np.uint8,
        )

        class DummyModel:
            def enhance(self, image, outscale):
                assert np.array_equal(image, image_np[:, :, ::-1])
                assert outscale == 4
                return enhanced_bgr, "unused"

        mock_load_model.return_value = DummyModel()

        output = realesrgan.processing(image_np=image_np, outscale=4)

        assert np.array_equal(output, enhanced_bgr[:, :, ::-1])
    
    def test_processing(self) -> None:
        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        image_np = RealEsrGan().processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
