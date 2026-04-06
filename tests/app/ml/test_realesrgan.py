import numpy as np
import pytest
import torch

from app.ml.realesrgan import RealEsrGan


class TestRealEsrGan:
    """RealEsrGan の補助メソッドと処理フローを検証するテストクラス。"""

    def test_reverse_color_channels_reverses_rgb_and_bgr_order(self) -> None:
        """色チャンネル順が末尾軸で反転されることを確認する。"""
        realesrgan = RealEsrGan()
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        output = realesrgan._reverse_color_channels(image_np)

        assert np.array_equal(output, image_np[:, :, ::-1])

    def test_load_model_raises_when_model_path_does_not_exist(self) -> None:
        """モデル重みが存在しない場合に例外を送出することを確認する。"""
        realesrgan = RealEsrGan()
        realesrgan.model_path = realesrgan.model_path.with_name("missing.pth")

        with pytest.raises(FileNotFoundError, match="missing.pth"):
            realesrgan._load_model()

    def test_load_model_builds_realesrganer_with_expected_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RealESRGANer が期待どおりの引数で構築されることを確認する。"""
        realesrgan = RealEsrGan()
        captured = {}

        class DummyPath:
            """存在確認と文字列表現だけを持つダミーパス。"""

            def exists(self) -> bool:
                return True

            def __str__(self) -> str:
                return "dummy/realesrgan.pth"

        class DummyRRDBNet:
            """RRDBNet の初期化引数を記録するダミークラス。"""

            def __init__(self, **kwargs):
                captured["rrdbnet_kwargs"] = kwargs
                captured["rrdbnet_instance"] = self

        class DummyRealESRGANer:
            """RealESRGANer の初期化引数を記録するダミークラス。"""

            def __init__(self, **kwargs):
                captured["realesrganer_kwargs"] = kwargs

        realesrgan.model_path = DummyPath()
        monkeypatch.setattr("app.ml.realesrgan.RRDBNet", DummyRRDBNet)
        monkeypatch.setattr("app.ml.realesrgan.RealESRGANer", DummyRealESRGANer)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        model = realesrgan._load_model()

        assert isinstance(model, DummyRealESRGANer)
        assert captured["rrdbnet_kwargs"] == {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": realesrgan.scale,
        }
        assert captured["realesrganer_kwargs"] == {
            "scale": realesrgan.scale,
            "model_path": str(realesrgan.model_path),
            "model": captured["rrdbnet_instance"],
            "tile": realesrgan.tile,
            "tile_pad": realesrgan.tile_pad,
            "pre_pad": realesrgan.pre_pad,
            "half": True,
            "device": realesrgan.device,
        }

    def test_processing_loads_model_and_enhances_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """processing が BGR 変換後に enhance を呼び、最後に RGB へ戻すことを確認する。"""
        realesrgan = RealEsrGan()
        image_np = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        enhanced_bgr = np.array(
            [
                [[21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32]],
            ],
            dtype=np.uint8,
        )
        captured = {}

        class DummyModel:
            """enhance 呼び出し内容を記録して固定値を返すダミーモデル。"""

            def enhance(self, image, outscale):
                captured["image"] = image
                captured["outscale"] = outscale
                return enhanced_bgr, "unused"

        dummy_model = DummyModel()
        monkeypatch.setattr(realesrgan, "_load_model", lambda: dummy_model)

        output = realesrgan.processing(image_np=image_np, outscale=4)

        assert captured["outscale"] == 4
        assert np.array_equal(captured["image"], image_np[:, :, ::-1])
        assert np.array_equal(output, enhanced_bgr[:, :, ::-1])
