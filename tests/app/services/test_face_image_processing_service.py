import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.services.face_image_processing_service import FaceImageProcessingService


class TestFaceImageProcessingService:
    def _create_base64_image(
        self,
        *,
        color: tuple[int, int, int],
        extension: ExtensionType,
        size: tuple[int, int] = (2, 2),
    ) -> str:
        image = Image.new("RGB", size=size, color=color)
        buffer = BytesIO()
        image.save(buffer, format=extension.value)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    @pytest.mark.parametrize(
        (
            "use_brightness_adjustment",
            "input_extension",
            "expected_color",
            "expected_processing_calls",
        ),
        [
            (False, ExtensionType.PNG, (16, 32, 48), 0),
            (True, ExtensionType.PNG, (200, 150, 100), 1),
            (False, ExtensionType.JPEG, (16, 32, 48), 0),
            (True, ExtensionType.JPEG, (200, 150, 100), 1),
        ],
    )
    def test_processing_returns_expected_image_and_extension(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_brightness_adjustment: bool,
        input_extension: ExtensionType,
        expected_color: tuple[int, int, int],
        expected_processing_calls: int,
    ) -> None:
        service = FaceImageProcessingService()
        encoded_image = self._create_base64_image(
            color=(16, 32, 48),
            extension=input_extension,
        )
        enhanced_image_np = np.full((2, 2, 3), (200, 150, 100), dtype=np.uint8)
        captured_image_np: list[np.ndarray] = []
        original_fromarray = Image.fromarray

        def compatible_fromarray(obj: np.ndarray, mode: str | None = None) -> Image.Image:
            if np.issubdtype(obj.dtype, np.floating):
                obj = np.clip(obj, 0.0, 1.0)
                obj = (obj * 255.0).round().astype(np.uint8)
            return original_fromarray(obj, mode=mode)

        class DummyRetinexformer:
            def processing(self, *, image_np: np.ndarray) -> np.ndarray:
                captured_image_np.append(image_np)
                return enhanced_image_np

        monkeypatch.setattr(
            "app.services.face_image_processing_service.Retinexformer",
            DummyRetinexformer,
        )
        monkeypatch.setattr(Image, "fromarray", compatible_fromarray)

        response = service.processing(
            content=encoded_image,
            extension=input_extension,
            use_brightness_adjustment=use_brightness_adjustment,
            use_correction=False,
        )
        result_image = self._decode_image(response.content)

        assert response == FaceImageProcessingResponse(
            content=response.content,
            extension=input_extension,
        )
        assert len(captured_image_np) == expected_processing_calls
        if expected_processing_calls:
            assert captured_image_np[0].dtype == np.float32
            assert captured_image_np[0].shape == (2, 2, 3)
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
    )
    def test_pil_image_to_base64_encodes_with_requested_format(
        self,
        extension: ExtensionType,
        expected_format: str,
    ) -> None:
        service = FaceImageProcessingService()
        image = Image.new("RGBA", size=(2, 2), color=(10, 20, 30, 255))

        encoded = service._pil_image_to_base64(image=image, extension=extension)
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))

        assert decoded.format == expected_format

    def test_pil_image_to_base64_raises_value_error_when_save_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(1, 1), color=(0, 0, 0))

        def raise_on_save(self, fp, format=None, **params):
            raise OSError("save failed")

        monkeypatch.setattr(Image.Image, "save", raise_on_save)

        with pytest.raises(ValueError, match="Failed to encode image: save failed"):
            service._pil_image_to_base64(image=image, extension=ExtensionType.PNG)
