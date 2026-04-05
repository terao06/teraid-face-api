import base64
import json
from io import BytesIO
from pathlib import Path
import asyncio

import numpy as np
from PIL import Image
import pytest

from app.apis.endpoints.face_image_processing import (
    face_image_process,
)
from app.main import app
from app.models.requests.face_image_processing_request import (
    ExtensionType,
    FaceImageProcessingRequest,
)
from app.models.responses.face_image_processing_response import (
    FaceImageProcessingResponse,
)


class TestFaceImageProcessing:
    TEST_FACE_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/test_face.png")
    RETINEXFORMER_RESULT_IMAGE_PATH = Path(
        "tests/app/test_data/test_image/retinexformer/result_face.png"
    )
    GFPGAN_RESULT_IMAGE_PATH = Path(
        "tests/app/test_data/test_image/gfpgan/result_face.png"
    )

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
        response_headers: list[tuple[bytes, bytes]] = []
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
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }

        async def send(message: dict[str, object]) -> None:
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                response_headers = list(message.get("headers", []))
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

    def test_face_image_process_wraps_controller_response(
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
            extension=ExtensionType.JPEG,
        )
        captured: dict[str, object] = {}

        class DummyFaceImageProcessingController:
            def processing(
                self, *, request: FaceImageProcessingRequest
            ) -> FaceImageProcessingResponse:
                captured["request"] = request
                return expected_response

        monkeypatch.setattr(
            "app.apis.endpoints.face_image_processing.FaceImageProcessingController",
            DummyFaceImageProcessingController,
        )

        response = face_image_process(request=request)

        assert captured["request"] == request
        assert response == {
            "status": "success",
            "data": {
                "content": "processed-image",
                "extension": ExtensionType.JPEG,
            },
        }

    @pytest.mark.parametrize(
        ("use_brightness_adjustment", "use_correction", "expected_status_code", 
         "test_image_path", "expected_image_path"),
        [
            (False, False, 200, TEST_FACE_IMAGE_PATH, TEST_FACE_IMAGE_PATH),
            (True, False, 200, TEST_FACE_IMAGE_PATH, RETINEXFORMER_RESULT_IMAGE_PATH),
            (True, True, 200, TEST_FACE_IMAGE_PATH, GFPGAN_RESULT_IMAGE_PATH),
        ],
    )
    def test_face_image_process_with_real_image(
        self,
        use_brightness_adjustment: bool,
        use_correction: bool,
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
                "use_brightness_adjustment": use_brightness_adjustment,
                "use_correction": use_correction,
            }
        )

        assert status_code == expected_status_code
        assert expected_image_path is not None
        expected_image = self._load_image(expected_image_path)
        result_image = self._decode_image(payload["data"]["content"])
        assert payload["status"] == "success"
        assert payload["data"]["extension"] == ExtensionType.PNG.value
        assert result_image.size == expected_image.size
        diff = np.abs(
            np.asarray(result_image, dtype=np.int16) - np.asarray(expected_image, dtype=np.int16)
        )
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
