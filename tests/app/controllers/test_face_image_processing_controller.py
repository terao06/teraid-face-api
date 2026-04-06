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
    """FaceImageProcessingController のサービス委譲を検証するテストクラス。"""

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
        ids=[
            "PNG入力で明るさ補正のみをサービスへ委譲する",
            "JPEG入力で顔補正のみをサービスへ委譲する",
        ],
    )
    def test_processing_delegates_to_service_with_request_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extension: ExtensionType,
        use_brightness_adjustment_lm: bool,
        use_correction_lm: bool,
        use_resolution_lm: bool,
    ) -> None:
        """リクエストの各値がそのままサービスへ渡され、結果が返されることを確認する。"""
        # コントローラが受け取るリクエストと、サービスから返る想定レスポンスを用意する。
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
        captured: dict[str, object] = {}

        class DummyFaceImageProcessingService:
            """コントローラから委譲された引数を記録するダミーサービス。"""

            def processing(
                self,
                *,
                content: str,
                extension: ExtensionType,
                use_brightness_adjustment_lm: bool,
                use_correction_lm: bool,
                use_resolution_lm: bool,
            ) -> FaceImageProcessingResponse:
                # コントローラが渡した値を保持し、後で委譲内容を検証する。
                captured["content"] = content
                captured["extension"] = extension
                captured["use_brightness_adjustment_lm"] = use_brightness_adjustment_lm
                captured["use_correction_lm"] = use_correction_lm
                captured["use_resolution_lm"] = use_resolution_lm
                return expected_response

        # 実サービスを差し替え、コントローラの引数受け渡しだけを検証する。
        monkeypatch.setattr(
            "app.controllers.face_image_processing_controller.FaceImageProcessingService",
            DummyFaceImageProcessingService,
        )

        # 公開メソッドを実行し、サービス戻り値がそのまま返ることを確認する。
        response = FaceImageProcessingController().processing(request=request)

        # レスポンスの透過と、リクエスト値の完全な委譲を確認する。
        assert response == expected_response
        assert captured == {
            "content": "encoded-image",
            "extension": extension,
            "use_brightness_adjustment_lm": use_brightness_adjustment_lm,
            "use_correction_lm": use_correction_lm,
            "use_resolution_lm": use_resolution_lm,
        }
