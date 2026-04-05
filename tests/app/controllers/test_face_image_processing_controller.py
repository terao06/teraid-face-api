import pytest

from app.controllers.face_image_processing_controller import FaceImageProcessingController
from app.models.requests.face_image_processing_request import (
    ExtensionType,
    FaceImageProcessingRequest,
)
from app.models.responses.face_image_processing_response import (
    FaceImageProcessingResponse,
)


class TestFaceImageProcessingController:
    def test_processing_delegates_to_service_with_request_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = FaceImageProcessingRequest(
            content="encoded-image",
            extension=ExtensionType.PNG,
            use_brightness_adjustment=True,
            use_correction=False,
        )
        expected_response = FaceImageProcessingResponse(
            content="processed-image",
            extension=ExtensionType.PNG,
        )
        captured: dict[str, object] = {}

        class DummyFaceImageProcessingService:
            def processing(
                self,
                *,
                content: str,
                extension: ExtensionType,
                use_brightness_adjustment: bool,
                use_correction: bool,
            ) -> FaceImageProcessingResponse:
                captured["content"] = content
                captured["extension"] = extension
                captured["use_brightness_adjustment"] = use_brightness_adjustment
                captured["use_correction"] = use_correction
                return expected_response

        monkeypatch.setattr(
            "app.controllers.face_image_processing_controller.FaceImageProcessingService",
            DummyFaceImageProcessingService,
        )

        response = FaceImageProcessingController().processing(request=request)

        assert response == expected_response
        assert captured == {
            "content": "encoded-image",
            "extension": ExtensionType.PNG,
            "use_brightness_adjustment": True,
            "use_correction": False,
        }
