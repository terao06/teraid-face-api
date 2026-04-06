import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.services.face_image_processing_service import FaceImageProcessingService


class TestFaceImageProcessingService:
    """FaceImageProcessingService の処理内容を検証するテスト。"""

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
            "input_extension",
            "expected_color",
            "expected_size_bytes",
            "expected_retinex_calls",
            "expected_gfpgan_calls",
            "expected_realesrgan_calls",
        ),
        [
            (False, False, False, ExtensionType.PNG, (16, 32, 48), 75, 0, 0, 0),
            (True, False, False, ExtensionType.PNG, (200, 150, 100), 75, 1, 0, 0),
            (False, True, False, ExtensionType.PNG, (80, 90, 100), 75, 0, 1, 0),
            (False, False, True, ExtensionType.PNG, (40, 50, 60), 75, 0, 0, 1),
            (True, True, True, ExtensionType.PNG, (40, 50, 60), 75, 1, 1, 1),
            (False, False, False, ExtensionType.JPEG, (16, 32, 48), 632, 0, 0, 0),
            (True, False, False, ExtensionType.JPEG, (200, 150, 100), 632, 1, 0, 0),
            (False, True, False, ExtensionType.JPEG, (80, 90, 100), 631, 0, 1, 0),
            (False, False, True, ExtensionType.JPEG, (40, 50, 60), 632, 0, 0, 1),
            (True, True, True, ExtensionType.JPEG, (40, 50, 60), 632, 1, 1, 1),
        ],
        ids=[
            "png_without_lm",
            "png_brightness_only",
            "png_correction_only",
            "png_resolution_only",
            "png_all_lm",
            "jpeg_without_lm",
            "jpeg_brightness_only",
            "jpeg_correction_only",
            "jpeg_resolution_only",
            "jpeg_all_lm",
        ],
    )
    def test_processing_returns_expected_image_and_extension(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
        input_extension: ExtensionType,
        expected_color: tuple[int, int, int],
        expected_size_bytes: int,
        expected_retinex_calls: int,
        expected_gfpgan_calls: int,
        expected_realesrgan_calls: int,
    ) -> None:
        """指定したフラグに応じて処理結果と呼び出し回数が変わることを検証する。"""
        service = FaceImageProcessingService()
        encoded_image = self._create_base64_image(
            color=(16, 32, 48),
            extension=input_extension,
        )

        retinex_output = np.full((2, 2, 3), (200, 150, 100), dtype=np.uint8)
        gfpgan_output = np.full((2, 2, 3), (80, 90, 100), dtype=np.uint8)
        realesrgan_output = np.full((2, 2, 3), (40, 50, 60), dtype=np.uint8)
        captured_retinex_inputs: list[np.ndarray] = []
        captured_gfpgan_inputs: list[np.ndarray] = []
        captured_realesrgan_inputs: list[np.ndarray] = []
        original_fromarray = Image.fromarray

        def compatible_fromarray(obj: np.ndarray, mode: str | None = None) -> Image.Image:
            if np.issubdtype(obj.dtype, np.floating):
                obj = np.clip(obj, 0.0, 1.0)
                obj = (obj * 255.0).round().astype(np.uint8)
            return original_fromarray(obj, mode=mode)

        class DummyRetinexformer:
            def processing(self, *, image_np: np.ndarray) -> np.ndarray:
                captured_retinex_inputs.append(image_np)
                return retinex_output

        class DummyGfpgan:
            def processing(self, *, image_np: np.ndarray) -> np.ndarray:
                captured_gfpgan_inputs.append(image_np)
                return gfpgan_output

        class DummyRealEsrGan:
            def processing(self, *, image_np: np.ndarray) -> np.ndarray:
                captured_realesrgan_inputs.append(image_np)
                return realesrgan_output

        monkeypatch.setattr(
            "app.services.face_image_processing_service.Retinexformer",
            DummyRetinexformer,
        )
        monkeypatch.setattr(
            "app.services.face_image_processing_service.Gfpgan",
            DummyGfpgan,
        )
        monkeypatch.setattr(
            "app.services.face_image_processing_service.RealEsrGan",
            DummyRealEsrGan,
        )
        monkeypatch.setattr(Image, "fromarray", compatible_fromarray)

        response = service.processing(
            content=encoded_image,
            extension=input_extension,
            use_brightness_adjustment_lm=use_brightness_adjustment_lm,
            use_correction_lm=use_correction_lm,
            use_resolution_lm=use_resolution_lm,
        )
        result_image = Image.open(BytesIO(base64.b64decode(response.content))).convert("RGB")

        assert response == FaceImageProcessingResponse(
            content=response.content,
            extension=input_extension,
            size_bytes=expected_size_bytes,
        )
        assert len(captured_retinex_inputs) == expected_retinex_calls
        assert len(captured_gfpgan_inputs) == expected_gfpgan_calls
        assert len(captured_realesrgan_inputs) == expected_realesrgan_calls
        if expected_retinex_calls:
            assert captured_retinex_inputs[0].dtype == np.float32
            assert captured_retinex_inputs[0].shape == (2, 2, 3)
        if expected_gfpgan_calls:
            expected_gfpgan_input = (
                retinex_output
                if use_brightness_adjustment_lm
                else np.full((2, 2, 3), (16, 32, 48), dtype=np.uint8)
            )
            assert np.array_equal(captured_gfpgan_inputs[0], expected_gfpgan_input)
        if expected_realesrgan_calls:
            if use_correction_lm:
                expected_realesrgan_input = gfpgan_output
            elif use_brightness_adjustment_lm:
                expected_realesrgan_input = retinex_output
            else:
                expected_realesrgan_input = np.full((2, 2, 3), (16, 32, 48), dtype=np.uint8)
            assert np.array_equal(captured_realesrgan_inputs[0], expected_realesrgan_input)
        assert result_image.size == (2, 2)
        assert np.allclose(
            np.asarray(result_image, dtype=np.float32),
            np.full((2, 2, 3), expected_color, dtype=np.float32),
            atol=5.0,
        )

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
    def test_pil_image_to_base64_raises_value_error_when_save_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extension: ExtensionType,
    ) -> None:
        """画像保存失敗時に ValueError を送出することを検証する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(1, 1), color=(0, 0, 0))

        def raise_on_save(self, fp, format=None, **params):
            raise OSError("save failed")

        monkeypatch.setattr(Image.Image, "save", raise_on_save)

        with pytest.raises(ValueError, match="Failed to encode image: save failed"):
            service._pil_image_to_base64(image=image, extension=extension)

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
        """指定形式で保存した画像バイト数を返すことを確認する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(3, 3), color=(10, 20, 30))

        size_bytes = service._get_image_size_bytes(img=image, format=extension)

        buffer = BytesIO()
        image.save(buffer, format=expected_format)
        assert size_bytes == len(buffer.getvalue())
