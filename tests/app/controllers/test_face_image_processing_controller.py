from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.exceptions import FaceNotFoundException, MultipleFacesDetectionException
from app.core.messages import ValidationMessages
from app.controllers.face_image_processing_controller import FaceImageProcessingController
from app.models.requests.face_image_processing_request import (
    ExtensionType,
    FaceImageProcessingRequest,
)
from app.models.responses.face_image_processing_response import (
    FaceImageProcessingResponse,
)
from app.services.face_image_processing_service import FaceImageProcessingService


class TestFaceImageProcessingController:
    @pytest.mark.parametrize(
        (
            "extension",
            "use_brightness_adjustment_lm",
            "use_correction_lm",
            "use_resolution_lm",
        ),
        [
            (ExtensionType.PNG, True, False, True),
            (ExtensionType.JPEG, False, True, False),
        ],
    )
    @patch.object(FaceImageProcessingService, "processing")
    def test_processing_delegates_to_service_with_request_values(
        self,
        mock_processing,
        extension: ExtensionType,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
    ) -> None:
        request = FaceImageProcessingRequest(
            content="encoded-image",
            extension=extension,
            use_brightness_adjustment_lm=use_brightness_adjustment_lm,
            use_correction_lm=use_correction_lm,
            use_resolution_lm=use_resolution_lm,
        )
        expected_response = FaceImageProcessingResponse(
            content="processed-image",
            extension=extension,
            size_bytes=123,
        )
        mock_processing.return_value = expected_response

        response = FaceImageProcessingController().processing(request=request)

        assert response == expected_response
        mock_processing.assert_called_once_with(
            content="encoded-image",
            extension=extension,
            use_brightness_adjustment_lm=use_brightness_adjustment_lm,
            use_correction_lm=use_correction_lm,
            use_resolution_lm=use_resolution_lm,
        )

    @pytest.mark.parametrize(
        ("raised_exception", "expected_status_code", "expected_detail"),
        [
            (FaceNotFoundException(), 404, ValidationMessages.FACE_NOT_FOUND),
            (
                MultipleFacesDetectionException(),
                409,
                ValidationMessages.MULTIPLE_FACES_DETECTED,
            ),
        ],
        ids=[
            "face_not_found_maps_to_404",
            "multiple_faces_maps_to_409",
        ],
    )
    @patch.object(FaceImageProcessingService, "processing")
    def test_processing_converts_domain_exceptions_to_http_exception(
        self,
        mock_processing,
        raised_exception: Exception,
        expected_status_code: int,
        expected_detail: str,
    ) -> None:
        request = FaceImageProcessingRequest(
            content="encoded-image",
            extension=ExtensionType.PNG,
            use_brightness_adjustment_lm=False,
            use_correction_lm=False,
            use_resolution_lm=False,
        )
        mock_processing.side_effect = raised_exception

        with pytest.raises(HTTPException) as exc_info:
            FaceImageProcessingController().processing(request=request)

        assert exc_info.value.status_code == expected_status_code
        assert exc_info.value.detail == expected_detail
