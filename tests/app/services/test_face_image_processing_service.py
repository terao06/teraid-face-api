import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.services.face_image_processing_service import FaceImageProcessingService


class TestFaceImageProcessingService:
    """FaceImageProcessingService の画像変換と処理分岐を検証するテストクラス。"""

    def _create_base64_image(
        self,
        *,
        color: tuple[int, int, int],
        extension: ExtensionType,
        size: tuple[int, int] = (2, 2),
    ) -> str:
        """指定色・拡張子の画像を base64 文字列へ変換して返す。"""
        # テスト入力用の単色画像を生成し、サービス入力と同じ base64 形式へ変換する。
        image = Image.new("RGB", size=size, color=color)
        buffer = BytesIO()
        image.save(buffer, format=extension.value)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded: str) -> Image.Image:
        """base64 文字列を RGB の PIL.Image として復元する。"""
        # サービス出力を画素値で比較できるように PIL.Image へ戻す。
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
        ids=[
            "PNG入力で明るさ補正なしなら元画像をそのまま返す",
            "PNG入力で明るさ補正ありなら補正後画像を返す",
            "JPEG入力で明るさ補正なしなら元画像をそのまま返す",
            "JPEG入力で明るさ補正ありなら補正後画像を返す",
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
        """明るさ補正の有無と拡張子に応じて期待どおりの画像と拡張子を返すことを確認する。"""
        service = FaceImageProcessingService()
        # 入力画像と補正後画像を用意し、明るさ補正の分岐を同じテストで検証する。
        encoded_image = self._create_base64_image(
            color=(16, 32, 48),
            extension=input_extension,
        )
        enhanced_image_np = np.full((2, 2, 3), (200, 150, 100), dtype=np.uint8)
        captured_image_np: list[np.ndarray] = []
        original_fromarray = Image.fromarray

        def compatible_fromarray(obj: np.ndarray, mode: str | None = None) -> Image.Image:
            # float 配列が渡された場合も PIL へ安全に変換できるようにする。
            if np.issubdtype(obj.dtype, np.floating):
                obj = np.clip(obj, 0.0, 1.0)
                obj = (obj * 255.0).round().astype(np.uint8)
            return original_fromarray(obj, mode=mode)

        class DummyRetinexformer:
            """明るさ補正への入力画像を記録し、固定の補正結果を返すダミークラス。"""

            def processing(self, *, image_np: np.ndarray) -> np.ndarray:
                # サービスから渡された画像配列を保持し、後で型と形状を確認する。
                captured_image_np.append(image_np)
                return enhanced_image_np

        # 外部補正処理を差し替え、サービス内部の分岐と画像変換だけを検証する。
        monkeypatch.setattr(
            "app.services.face_image_processing_service.Retinexformer",
            DummyRetinexformer,
        )
        monkeypatch.setattr(Image, "fromarray", compatible_fromarray)

        # 公開メソッドを実行し、返却画像をデコードして比較する。
        response = service.processing(
            content=encoded_image,
            extension=input_extension,
            use_brightness_adjustment=use_brightness_adjustment,
            use_correction=False,
        )
        result_image = self._decode_image(response.content)

        # レスポンスの拡張子維持、補正呼び出し回数、最終的な画素値を確認する。
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
        ids=[
            "PNG指定ならPNG形式でエンコードする",
            "JPEG指定ならJPEG形式でエンコードする",
        ],
    )
    def test_pil_image_to_base64_encodes_with_requested_format(
        self,
        extension: ExtensionType,
        expected_format: str,
    ) -> None:
        """指定した拡張子に対応する画像形式で base64 エンコードされることを確認する。"""
        service = FaceImageProcessingService()
        # RGBA 画像を用意し、各拡張子でどの形式に保存されるかを確認する。
        image = Image.new("RGBA", size=(2, 2), color=(10, 20, 30, 255))

        # private メソッドでエンコード後、画像として復元して形式を確認する。
        encoded = service._pil_image_to_base64(image=image, extension=extension)
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))

        assert decoded.format == expected_format

    @pytest.mark.parametrize(
        "extension",
        [ExtensionType.PNG, ExtensionType.JPEG],
        ids=[
            "PNG保存失敗時はValueErrorへ変換する",
            "JPEG保存失敗時はValueErrorへ変換する",
        ],
    )
    def test_pil_image_to_base64_raises_value_error_when_save_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extension: ExtensionType,
    ) -> None:
        """画像保存に失敗した場合に ValueError へ変換して送出することを確認する。"""
        service = FaceImageProcessingService()
        image = Image.new("RGB", size=(1, 1), color=(0, 0, 0))

        def raise_on_save(self, fp, format=None, **params):
            # Pillow 保存時の失敗を再現し、例外ラップを確認する。
            raise OSError("save failed")

        # 画像保存処理を失敗させ、サービスが ValueError へ変換することを確認する。
        monkeypatch.setattr(Image.Image, "save", raise_on_save)

        with pytest.raises(ValueError, match="Failed to encode image: save failed"):
            service._pil_image_to_base64(image=image, extension=extension)
