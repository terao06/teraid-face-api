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
    """顔画像処理エンドポイントのレスポンス整形と実画像処理を検証するテストクラス。"""

    TEST_FACE_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/test_face.png")
    RETINEXFORMER_RESULT_IMAGE_PATH = Path(
        "tests/app/test_data/test_image/retinexformer/result_face.png"
    )
    GFPGAN_RESULT_IMAGE_PATH = Path(
        "tests/app/test_data/test_image/gfpgan/result_face.png"
    )

    REALESRGAN_RESULT_IMAGE_PATH = Path(
        "tests/app/test_data/test_image/realesrgan/result_face.png"
    )

    def _encode_image(self, image: Image.Image, extension: ExtensionType) -> str:
        """PIL.Image を指定拡張子で base64 文字列へ変換する。"""
        # 実画像テスト用の入力を API リクエストと同じ形式へ変換する。
        buffer = BytesIO()
        image.save(buffer, format=extension.value)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded: str) -> Image.Image:
        """base64 文字列を RGB の PIL.Image として復元する。"""
        # API レスポンスの画像を画素比較できるように PIL.Image へ戻す。
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

    def _load_image(self, path: Path) -> Image.Image:
        """テスト画像を RGB の PIL.Image として読み込む。"""
        return Image.open(path).convert("RGB")

    def _post_face_image_process(self, payload: dict[str, object]) -> tuple[int, dict[str, object] | str]:
        """ASGI アプリへ直接 POST し、ステータスコードとレスポンス本文を返す。"""
        # ルーティングからレスポンス整形までを通すため、ASGI レベルで直接リクエストを組み立てる。
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
            # 単一チャンクの JSON リクエストボディを返す。
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }

        async def send(message: dict[str, object]) -> None:
            # 返却されたステータスコードとレスポンス本文を組み立てる。
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        try:
            asyncio.run(app(scope, receive, send))
        except Exception:
            # 例外送出前にレスポンスが開始されていなければ、そのまま失敗として扱う。
            if response_status is None:
                raise

        assert response_status is not None
        response_text = response_body.decode("utf-8")
        try:
            # JSON レスポンスであれば辞書へ変換し、それ以外は文字列のまま返す。
            return response_status, json.loads(response_text)
        except json.JSONDecodeError:
            return response_status, response_text

    @pytest.mark.parametrize(
        ("request_extension", "response_extension"),
        [
            (ExtensionType.PNG, ExtensionType.JPEG),
            (ExtensionType.JPEG, ExtensionType.PNG),
        ],
        ids=[
            "PNGリクエストでもJPEGレスポンスをsuccess形式で返す",
            "JPEGリクエストでもPNGレスポンスをsuccess形式で返す",
        ],
    )
    def test_face_image_process_wraps_controller_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_extension: ExtensionType,
        response_extension: ExtensionType,
    ) -> None:
        """コントローラの戻り値が success 形式のレスポンスへ包まれることを確認する。"""
        # エンドポイントへ渡すリクエストと、コントローラが返す想定レスポンスを用意する。
        request = FaceImageProcessingRequest(
            content="encoded-image",
            extension=request_extension,
            use_brightness_adjustment_lm=True,
            use_correction_lm=False,
            use_resolution_lm=False,
        )
        expected_response = FaceImageProcessingResponse(
            content="processed-image",
            extension=response_extension,
            size_bytes=123,
        )
        captured: dict[str, object] = {}

        class DummyFaceImageProcessingController:
            """エンドポイントから渡されたリクエストを記録するダミーコントローラ。"""

            def processing(
                self, *, request: FaceImageProcessingRequest
            ) -> FaceImageProcessingResponse:
                # エンドポイントが委譲したリクエスト内容を保持する。
                captured["request"] = request
                return expected_response

        # 実コントローラを差し替え、レスポンス整形の責務だけを検証する。
        monkeypatch.setattr(
            "app.apis.endpoints.face_image_processing.FaceImageProcessingController",
            DummyFaceImageProcessingController,
        )

        # エンドポイント関数を直接呼び出し、返却形式を確認する。
        response = face_image_process(request=request)

        # コントローラへの委譲内容と、success/data ラップ後のレスポンスを確認する。
        assert captured["request"] == request
        assert response == {
            "status": "success",
            "data": {
                "content": "processed-image",
                "extension": response_extension,
                "size_bytes": 123,
            },
        }

    @pytest.mark.parametrize(
        (
            "use_brightness_adjustment_lm",
            "use_correction_lm",
            "use_resolution_lm",
            "expected_status_code",
            "test_image_path",
            "expected_image_path",
        ),
        [
            (False, False, False, 200, TEST_FACE_IMAGE_PATH, TEST_FACE_IMAGE_PATH),
            (True, False, False, 200, TEST_FACE_IMAGE_PATH, RETINEXFORMER_RESULT_IMAGE_PATH),
            (True, True, False, 200, TEST_FACE_IMAGE_PATH, GFPGAN_RESULT_IMAGE_PATH),
            (True, True, True, 200, TEST_FACE_IMAGE_PATH, REALESRGAN_RESULT_IMAGE_PATH),
        ],
        ids=[
            "補正なしなら入力画像に近い結果を返す",
            "明るさ補正のみならRetinexformer結果に近い画像を返す",
            "明るさ補正と顔補正の両方ならGFPGAN結果に近い画像を返す",
            "すべて使用した場合、real esr gan結果に近い画像を返す",
        ],
    )
    def test_face_image_process_with_real_image(
        self,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
        expected_status_code: int,
        test_image_path: Path,
        expected_image_path: Path | None,
    ) -> None:
        """実画像入力で補正条件ごとの期待画像に十分近い結果を返すことを確認する。"""
        # 入力画像を base64 化し、エンドポイントへ直接 POST する。
        source_image = self._load_image(test_image_path)
        encoded_image = self._encode_image(
            image=source_image,
            extension=ExtensionType.PNG,
        )
        status_code, payload = self._post_face_image_process(
            {
                "content": encoded_image,
                "extension": ExtensionType.PNG.value,
                "use_brightness_adjustment_lm": use_brightness_adjustment_lm,
                "use_correction_lm": use_correction_lm,
                "use_resolution_lm": use_resolution_lm,
            }
        )

        # ステータスコードと返却画像が期待どおりであることを確認する。
        assert status_code == expected_status_code
        assert expected_image_path is not None
        expected_image = self._load_image(expected_image_path)
        result_image = self._decode_image(payload["data"]["content"])
        assert payload["status"] == "success"
        assert payload["data"]["extension"] == ExtensionType.PNG.value
        assert isinstance(payload["data"]["size_bytes"], int)
        assert payload["data"]["size_bytes"] > 0
        assert result_image.size == expected_image.size
        # 実画像比較では平均差分と高パーセンタイル差分の両方を確認する。
        diff = np.abs(
            np.asarray(result_image, dtype=np.int16) - np.asarray(expected_image, dtype=np.int16)
        )
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
