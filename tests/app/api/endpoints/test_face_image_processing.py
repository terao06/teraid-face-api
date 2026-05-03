import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
from fastapi import HTTPException

from app.apis.endpoints.face_image_processing import face_image_process
from app.controllers.face_image_processing_controller import FaceImageProcessingController
from app.core.messages import ValidationMessages
from app.main import app
from app.models.requests.face_image_processing_request import (
    ExtensionType,
    FaceImageProcessingRequest,
)
from app.models.responses.face_image_processing_response import (
    FaceImageProcessingResponse,
)


class TestFaceImageProcessing:
    TEST_FACE_IMAGE_PATH = Path("tests/test_data/images/retinexformer/test_face.png")
    RETINEXFORMER_RESULT_IMAGE_PATH = Path(
        "tests/test_data/images/retinexformer/result_face.png"
    )
    GFPGAN_RESULT_IMAGE_PATH = Path(
        "tests/test_data/images/gfpgan/result_face.png"
    )
    REALESRGAN_RESULT_IMAGE_PATH = Path(
        "tests/test_data/images/realesrgan/result_face.png"
    )
    TEST_ANGLE_COLLECT_IMAGE_PATH = Path("tests/test_data/images/facealignment/test_face.png")
    ANGLE_COLLECT_RESULT_IMAGE_PATH = Path("tests/test_data/images/facealignment/result_face.png")

    def _encode_image(self, image: Image.Image, extension: ExtensionType) -> str:
        buffer = BytesIO()
        image.save(buffer, format=extension.value)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _post_face_image_process(self, payload: dict[str, object]) -> tuple[int, dict[str, object] | str]:
        request_body = json.dumps(payload).encode("utf-8")
        response_status: int | None = None
        response_body = bytearray()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/face-image-process/",
            "raw_path": b"/face-image-process/",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(request_body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 123),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": request_body, "more_body": False}

        async def send(message: dict[str, object]) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        try:
            asyncio.run(app(scope, receive, send))
        except Exception:
            if response_status is None:
                raise

        assert response_status is not None
        response_text = response_body.decode("utf-8")
        try:
            return response_status, json.loads(response_text)
        except json.JSONDecodeError:
            return response_status, response_text

    @pytest.mark.parametrize(
        ("request_extension", "response_extension"),
        [
            (ExtensionType.PNG, ExtensionType.JPEG),
            (ExtensionType.JPEG, ExtensionType.PNG),
        ],
    )
    @patch.object(FaceImageProcessingController, "processing")
    def test_face_image_process_wraps_controller_response(
        self,
        mock_processing: MagicMock,
        request_extension: ExtensionType,
        response_extension: ExtensionType,
    ) -> None:
        request = FaceImageProcessingRequest(
            content="encoded-image",
            extension=request_extension,
            use_angle_correction=True,
            use_brightness_adjustment_lm=True,
            use_correction_lm=False,
            use_resolution_lm=False,
        )
        expected_response = FaceImageProcessingResponse(
            content="processed-image",
            extension=response_extension,
            size_bytes=123,
        )
        mock_processing.return_value = expected_response

        response = face_image_process(request=request)

        mock_processing.assert_called_once_with(request=request)
        assert response == {
            "status": "success",
            "data": {
                "content": "processed-image",
                "extension": response_extension,
                "size_bytes": 123,
            },
        }

    @pytest.mark.parametrize(
        ("raised_exception", "expected_status_code", "expected_detail"),
        [
            (
                HTTPException(status_code=404, detail=ValidationMessages.FACE_NOT_FOUND),
                404,
                ValidationMessages.FACE_NOT_FOUND,
            ),
            (
                HTTPException(
                    status_code=404,
                    detail=ValidationMessages.ANGLE_CORRECTION_ERROR,
                ),
                404,
                ValidationMessages.ANGLE_CORRECTION_ERROR,
            ),
            (
                HTTPException(
                    status_code=409,
                    detail=ValidationMessages.MULTIPLE_FACES_DETECTED,
                ),
                409,
                ValidationMessages.MULTIPLE_FACES_DETECTED,
            ),
        ],
        ids=[
            "face_not_found_returns_404",
            "angle_correction_error_returns_404",
            "multiple_faces_returns_409",
        ],
    )
    @patch.object(FaceImageProcessingController, "processing")
    def test_face_image_process_returns_http_exception_response(
        self,
        mock_processing: MagicMock,
        raised_exception: HTTPException,
        expected_status_code: int,
        expected_detail: str,
    ) -> None:
        mock_processing.side_effect = raised_exception

        status_code, payload = self._post_face_image_process(
            {
                "content": "encoded-image",
                "extension": ExtensionType.PNG.value,
                "use_angle_correction": False,
                "use_brightness_adjustment_lm": False,
                "use_correction_lm": False,
                "use_resolution_lm": False,
            }
        )

        assert status_code == expected_status_code
        assert payload == {"detail": expected_detail}

    @pytest.mark.parametrize(
        (
            "use_angle_correction",
            "use_brightness_adjustment_lm",
            "use_correction_lm",
            "use_resolution_lm",
            "expected_status_code",
            "test_image_path",
            "expected_image_path",
        ),
        [
            (False, False, False, False, 200, TEST_FACE_IMAGE_PATH, TEST_FACE_IMAGE_PATH),
            (False, True, False, False, 200, TEST_FACE_IMAGE_PATH, RETINEXFORMER_RESULT_IMAGE_PATH),
            (False, True, True, False, 200, TEST_FACE_IMAGE_PATH, GFPGAN_RESULT_IMAGE_PATH),
            (False, True, True, True, 200, TEST_FACE_IMAGE_PATH, REALESRGAN_RESULT_IMAGE_PATH),
            (True, False, False, False, 200, TEST_ANGLE_COLLECT_IMAGE_PATH, ANGLE_COLLECT_RESULT_IMAGE_PATH),
        ],
    )
    @pytest.mark.usefixtures("initialize_s3")
    def test_face_image_process_with_real_image(
        self,
        mock_ssm: MagicMock,
        use_angle_correction: bool,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
        expected_status_code: int,
        test_image_path: Path,
        expected_image_path: Path | None,
    ) -> None:
        source_image = self._load_image(test_image_path)
        encoded_image = self._encode_image(
            image=source_image,
            extension=ExtensionType.PNG,
        )
        status_code, payload = self._post_face_image_process(
            {
                "content": encoded_image,
                "extension": ExtensionType.PNG.value,
                "use_angle_correction": use_angle_correction,
                "use_brightness_adjustment_lm": use_brightness_adjustment_lm,
                "use_correction_lm": use_correction_lm,
                "use_resolution_lm": use_resolution_lm,
            }
        )

        mock_ssm.assert_called_once_with()
        assert status_code == expected_status_code
        assert expected_image_path is not None
        expected_image = self._load_image(expected_image_path)
        result_image = self._decode_image(payload["data"]["content"])
        assert payload["status"] == "success"
        assert payload["data"]["extension"] == ExtensionType.PNG.value
        assert isinstance(payload["data"]["size_bytes"], int)
        assert payload["data"]["size_bytes"] > 0
        assert result_image.size == expected_image.size
        diff = np.abs(
            np.asarray(result_image, dtype=np.int16) - np.asarray(expected_image, dtype=np.int16)
        )
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
