import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest

from app.helpers.validation_helper import ValidationHelper
from app.ml.face_alignment import FaceAlignment
from app.ml.gfpgan import Gfpgan
from app.ml.realesrgan import RealEsrGan
from app.ml.retinexformer import Retinexformer
from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.services.face_image_processing_service import FaceImageProcessingService


CAPTURED_VALIDATION_INPUTS: list[Image.Image] = []
CAPTURED_SCRFD_WEIGHT_INPUTS: list[bytes] = []
CAPTURED_FACE_ALIGNMENT_INPUTS: list[np.ndarray] = []
CAPTURED_RETINEX_INPUTS: list[np.ndarray] = []
CAPTURED_GFPGAN_INPUTS: list[np.ndarray] = []
CAPTURED_REALESRGAN_INPUTS: list[np.ndarray] = []
FACE_ALIGNMENT_OUTPUT = np.full((2, 2, 3), (120, 130, 140), dtype=np.uint8)
RETINEX_OUTPUT = np.full((2, 2, 3), (200, 150, 100), dtype=np.uint8)
GFPGAN_OUTPUT = np.full((2, 2, 3), (80, 90, 100), dtype=np.uint8)
REALESRGAN_OUTPUT = np.full((2, 2, 3), (40, 50, 60), dtype=np.uint8)
ORIGINAL_FROMARRAY = Image.fromarray


def reset_dummy_state() -> None:
    CAPTURED_VALIDATION_INPUTS.clear()
    CAPTURED_SCRFD_WEIGHT_INPUTS.clear()
    CAPTURED_FACE_ALIGNMENT_INPUTS.clear()
    CAPTURED_RETINEX_INPUTS.clear()
    CAPTURED_GFPGAN_INPUTS.clear()
    CAPTURED_REALESRGAN_INPUTS.clear()


def dummy_compatible_fromarray(obj: np.ndarray, mode: str | None = None) -> Image.Image:
    if np.issubdtype(obj.dtype, np.floating):
        obj = np.clip(obj, 0.0, 1.0)
        obj = (obj * 255.0).round().astype(np.uint8)
    return ORIGINAL_FROMARRAY(obj, mode=mode)


def dummy_validation_with_face(*, image: Image.Image, scrfd_weight_bytes: bytes) -> None:
    CAPTURED_VALIDATION_INPUTS.append(image)
    CAPTURED_SCRFD_WEIGHT_INPUTS.append(scrfd_weight_bytes)


def dummy_face_alignment_processing(*, image_np: np.ndarray) -> np.ndarray:
    CAPTURED_FACE_ALIGNMENT_INPUTS.append(image_np)
    return FACE_ALIGNMENT_OUTPUT


def dummy_retinexformer_processing(*, image_np: np.ndarray) -> np.ndarray:
    CAPTURED_RETINEX_INPUTS.append(image_np)
    return RETINEX_OUTPUT


def dummy_gfpgan_processing(*, image_np: np.ndarray) -> np.ndarray:
    CAPTURED_GFPGAN_INPUTS.append(image_np)
    return GFPGAN_OUTPUT


def dummy_realesrgan_processing(*, image_np: np.ndarray) -> np.ndarray:
    CAPTURED_REALESRGAN_INPUTS.append(image_np)
    return REALESRGAN_OUTPUT


def dummy_raise_on_save(self, fp, format=None, **params):
    raise OSError("save failed")


def dummy_get_object(*, bucket_name: str, key: str) -> bytes:
    assert bucket_name == "weights"
    assert key in {
        "scrfd/scrfd.onnx",
        "facealignment/face_landmarker.task",
        "retinexformer/MST_Plus_Plus_8x1150.pth",
        "gfpgan/GFPGANv.pth",
        "gfpgan/detection_Resnet50_Final.pth",
        "gfpgan/parsing_parsenet.pth",
        "realesrgan/RealESRGAN_x2plus.pth",
    }
    return b"dummy-weight-bytes"


class TestFaceImageProcessingService:
    """FaceImageProcessingService の処理を検証するテスト。"""

    def _create_base64_image(
        self,
        *,
        color: tuple[int, int, int],
        extension: ExtensionType,
        size: tuple[int, int] = (2, 2),
    ) -> str:
        """指定した画像を base64 文字列として返す。"""
        image = Image.new("RGB", size=size, color=color)
        buffer = BytesIO()
        image.save(buffer, format=extension.value)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded: str) -> Image.Image:
        """base64 文字列を RGB の PIL.Image に戻す。"""
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    @pytest.mark.parametrize(
        (
            "use_brightness_adjustment_lm",
            "use_correction_lm",
            "use_resolution_lm",
            "use_angle_correction",
            "input_extension",
            "expected_size_bytes",
            "expected_face_alignment_calls",
            "expected_retinex_calls",
            "expected_gfpgan_calls",
            "expected_realesrgan_calls",
        ),
        [
            (False, False, False, False, ExtensionType.PNG, 75, 0, 0, 0, 0),
            (True, False, False, False, ExtensionType.PNG, 75, 0, 1, 0, 0),
            (False, True, False, False, ExtensionType.PNG, 75, 0, 0, 1, 0),
            (False, False, True, False, ExtensionType.PNG, 75, 0, 0, 0, 1),
            (True, True, True, False, ExtensionType.PNG, 75, 0, 1, 1, 1),
            (False, False, False, True, ExtensionType.PNG, 75, 1, 0, 0, 0),
            (True, False, False, True, ExtensionType.PNG, 75, 1, 1, 0, 0),
            (False, False, False, False, ExtensionType.JPEG, 632, 0, 0, 0, 0),
            (True, False, False, False, ExtensionType.JPEG, 632, 0, 1, 0, 0),
            (False, True, False, False, ExtensionType.JPEG, 631, 0, 0, 1, 0),
            (False, False, True, False, ExtensionType.JPEG, 632, 0, 0, 0, 1),
            (True, True, True, False, ExtensionType.JPEG, 632, 0, 1, 1, 1),
            (False, False, False, True, ExtensionType.JPEG, 630, 1, 0, 0, 0),
        ],
        ids=[
            "png_without_lm",
            "png_brightness_only",
            "png_correction_only",
            "png_resolution_only",
            "png_all_lm",
            "png_angle_only",
            "png_angle_and_brightness",
            "jpeg_without_lm",
            "jpeg_brightness_only",
            "jpeg_correction_only",
            "jpeg_resolution_only",
            "jpeg_all_lm",
            "jpeg_angle_only",
        ],
    )
    @patch("app.services.face_image_processing_service.S3Client.get_object", side_effect=dummy_get_object)
    @patch.object(Image, "fromarray", side_effect=dummy_compatible_fromarray)
    @patch.object(ValidationHelper, "validation_with_face", side_effect=dummy_validation_with_face)
    @patch.object(RealEsrGan, "processing", side_effect=dummy_realesrgan_processing)
    @patch.object(Gfpgan, "processing", side_effect=dummy_gfpgan_processing)
    @patch.object(Retinexformer, "processing", side_effect=dummy_retinexformer_processing)
    @patch.object(FaceAlignment, "processing", side_effect=dummy_face_alignment_processing)
    def test_processing_returns_expected_image_and_extension(
        self,
        _mock_face_alignment_processing: MagicMock,
        _mock_retinexformer_processing: MagicMock,
        _mock_gfpgan_processing: MagicMock,
        _mock_realesrgan_processing: MagicMock,
        _mock_validation_with_face: MagicMock,
        _mock_fromarray: MagicMock,
        mock_s3_get_object: MagicMock,
        mock_ssm: MagicMock,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
        use_angle_correction: bool,
        input_extension: ExtensionType,
        expected_size_bytes: int,
        expected_face_alignment_calls: int,
        expected_retinex_calls: int,
        expected_gfpgan_calls: int,
        expected_realesrgan_calls: int,
    ) -> None:
        """指定フラグに応じて処理結果と呼び出し回数が変わることを検証する。"""
        reset_dummy_state()
        service = FaceImageProcessingService()
        encoded_image = self._create_base64_image(
            color=(16, 32, 48),
            extension=input_extension,
        )

        response = service.processing(
            content=encoded_image,
            extension=input_extension,
            use_angle_correction=use_angle_correction,
            use_brightness_adjustment_lm=use_brightness_adjustment_lm,
            use_correction_lm=use_correction_lm,
            use_resolution_lm=use_resolution_lm,
        )

        assert isinstance(response, FaceImageProcessingResponse)
        assert response.extension == input_extension
        assert response.size_bytes == expected_size_bytes
        assert len(CAPTURED_VALIDATION_INPUTS) == 1
        assert CAPTURED_SCRFD_WEIGHT_INPUTS == [b"dummy-weight-bytes"]
        assert isinstance(CAPTURED_VALIDATION_INPUTS[0], Image.Image)
        assert np.array_equal(
            np.asarray(CAPTURED_VALIDATION_INPUTS[0]),
            np.full((2, 2, 3), (16, 32, 48), dtype=np.uint8),
        )
        assert len(CAPTURED_FACE_ALIGNMENT_INPUTS) == expected_face_alignment_calls
        assert len(CAPTURED_RETINEX_INPUTS) == expected_retinex_calls
        assert len(CAPTURED_GFPGAN_INPUTS) == expected_gfpgan_calls
        assert len(CAPTURED_REALESRGAN_INPUTS) == expected_realesrgan_calls
        if expected_face_alignment_calls:
            assert CAPTURED_FACE_ALIGNMENT_INPUTS[0].dtype == np.uint8
            assert CAPTURED_FACE_ALIGNMENT_INPUTS[0].shape == (2, 2, 3)
        if expected_retinex_calls:
            assert CAPTURED_RETINEX_INPUTS[0].dtype == np.float32
            assert CAPTURED_RETINEX_INPUTS[0].shape == (2, 2, 3)
            expected_retinex_input = (
                FACE_ALIGNMENT_OUTPUT.astype(np.float32) / 255.0
                if use_angle_correction
                else np.full((2, 2, 3), (16, 32, 48), dtype=np.float32) / 255.0
            )
            assert np.array_equal(CAPTURED_RETINEX_INPUTS[0], expected_retinex_input)
        if expected_gfpgan_calls:
            expected_gfpgan_input = (
                RETINEX_OUTPUT
                if use_brightness_adjustment_lm
                else FACE_ALIGNMENT_OUTPUT
                if use_angle_correction
                else np.full((2, 2, 3), (16, 32, 48), dtype=np.uint8)
            )
            assert np.array_equal(CAPTURED_GFPGAN_INPUTS[0], expected_gfpgan_input)
        if expected_realesrgan_calls:
            if use_correction_lm:
                expected_realesrgan_input = GFPGAN_OUTPUT
            elif use_brightness_adjustment_lm:
                expected_realesrgan_input = RETINEX_OUTPUT
            elif use_angle_correction:
                expected_realesrgan_input = FACE_ALIGNMENT_OUTPUT
            else:
                expected_realesrgan_input = np.full((2, 2, 3), (16, 32, 48), dtype=np.uint8)
            assert np.array_equal(CAPTURED_REALESRGAN_INPUTS[0], expected_realesrgan_input)

        if use_angle_correction:
            _mock_face_alignment_processing.assert_called_once()
            mock_s3_get_object.assert_any_call(
                bucket_name="weights",
                key="facealignment/face_landmarker.task",
            )
        else:
            _mock_face_alignment_processing.assert_not_called()
        if use_brightness_adjustment_lm:
            _mock_retinexformer_processing.assert_called_once()
            mock_ssm.assert_called_once_with()
            mock_s3_get_object.assert_any_call(
                bucket_name="weights",
                key="scrfd/scrfd.onnx",
            )
            mock_s3_get_object.assert_any_call(
                bucket_name="weights",
                key="retinexformer/MST_Plus_Plus_8x1150.pth",
            )
        else:
            _mock_retinexformer_processing.assert_not_called()
            mock_ssm.assert_called_once_with()
            mock_s3_get_object.assert_any_call(
                bucket_name="weights",
                key="scrfd/scrfd.onnx",
            )
        if use_correction_lm:
            _mock_gfpgan_processing.assert_called_once()
        else:
            _mock_gfpgan_processing.assert_not_called()
        if use_resolution_lm:
            _mock_realesrgan_processing.assert_called_once()
        else:
            _mock_realesrgan_processing.assert_not_called()

        _mock_validation_with_face.assert_called_once()
        _mock_fromarray.assert_called_once()

    @pytest.mark.parametrize(
        ("extension", "expected_format"),
        [
            (ExtensionType.PNG, "PNG"),
            (ExtensionType.JPEG, "JPEG"),
        ],
        ids=[
            "png_encoding",
            "jpeg_encoding",
        ],
    )
    def test_pil_image_to_base64_encodes_with_requested_format(
        self,
        extension: ExtensionType,
        expected_format: str,
    ) -> None:
        """指定形式で base64 エンコードできることを検証する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGBA", size=(2, 2), color=(10, 20, 30, 255))

        encoded = service._pil_image_to_base64(image=image, extension=extension)
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))

        assert decoded.format == expected_format

    @pytest.mark.parametrize(
        "extension",
        [ExtensionType.PNG, ExtensionType.JPEG],
        ids=[
            "png_save_failure",
            "jpeg_save_failure",
        ],
    )
    @patch.object(Image.Image, "save", autospec=True, side_effect=dummy_raise_on_save)
    def test_pil_image_to_base64_raises_value_error_when_save_fails(
        self,
        _mock_save: MagicMock,
        extension: ExtensionType,
    ) -> None:
        """画像保存失敗時に ValueError を送出することを検証する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(1, 1), color=(0, 0, 0))

        with pytest.raises(ValueError, match="Failed to encode image: save failed"):
            service._pil_image_to_base64(image=image, extension=extension)
        
        _mock_save.assert_called_once()

    @pytest.mark.parametrize(
        ("extension", "expected_format"),
        [
            (ExtensionType.PNG, "PNG"),
            (ExtensionType.JPEG, "JPEG"),
        ],
        ids=[
            "png_size_bytes",
            "jpeg_size_bytes",
        ],
    )
    def test_get_image_size_bytes_returns_encoded_image_size(
        self,
        extension: ExtensionType,
        expected_format: str,
    ) -> None:
        """指定形式で保存した画像バイト数を返すことを検証する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(3, 3), color=(10, 20, 30))

        size_bytes = service._get_image_size_bytes(img=image, format=extension)

        buffer = BytesIO()
        image.save(buffer, format=expected_format)
        assert size_bytes == len(buffer.getvalue())
