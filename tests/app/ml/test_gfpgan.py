from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from app.ml import gfpgan as gfpgan_module
from app.ml.gfpgan import Gfpgan


TEST_IMAGE_PATH = Path("tests/app/test_data/test_image/gfpgan/test_face.png")
RESULT_IMAGE_PATH = Path("tests/app/test_data/test_image/gfpgan/result_face.png")


class TestGfpgan:
    """Gfpgan の補助メソッドと処理フローを検証するテストクラス。"""

    def test_patch_facexlib_initializers_replaces_helpers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """facexlib の初期化関数がクラス独自実装へ差し替えられることを確認する。"""
        gfpgan = Gfpgan()
        original_detection = object()
        original_parsing = object()

        # 元のヘルパー関数をダミーオブジェクトへ差し替え、後で置き換わったか判定できるようにする。
        monkeypatch.setattr(
            "app.ml.gfpgan.face_helper_module.init_detection_model",
            original_detection,
            raising=False,
        )
        monkeypatch.setattr(
            "app.ml.gfpgan.face_helper_module.init_parsing_model",
            original_parsing,
            raising=False,
        )

        # 初期化関数の差し替え処理を実行する。
        gfpgan._patch_facexlib_initializers()

        # 検出モデル用・パースモデル用の両方が新しい関数へ更新されていることを確認する。
        assert gfpgan_module.face_helper_module.init_detection_model is not original_detection
        assert gfpgan_module.face_helper_module.init_parsing_model is not original_parsing

    def test_custom_detection_initializer_loads_weights_and_strips_module_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """検出モデル初期化時に重みを読み込み、`module.` 接頭辞を除去することを確認する。"""
        gfpgan = Gfpgan()
        captured = {}

        class DummyRetinaFace:
            """RetinaFace の生成と重みロード内容を記録するダミーモデル。"""

            def __init__(self, network_name, half, device):
                # 初期化引数を保持し、期待した設定で生成されたか後続で確認できるようにする。
                captured["network_name"] = network_name
                captured["half"] = half
                captured["device"] = device

            def load_state_dict(self, state_dict, strict):
                # 読み込まれた重みと strict 設定を保存する。
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def eval(self):
                # 推論モードへ切り替えられたことを記録する。
                captured["eval_called"] = True
                return self

            def to(self, device):
                # 指定デバイスへ移動されたことを記録する。
                captured["to_device"] = device
                return self

        # モデル生成と重み読み込みを差し替え、初期化関数の責務だけを単体で検証する。
        monkeypatch.setattr("app.ml.gfpgan.RetinaFace", DummyRetinaFace)
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location: {"module.weight": torch.tensor([1.0])},
        )

        # facexlib 側の初期化関数を差し替えた上で、検出モデルを生成する。
        gfpgan._patch_facexlib_initializers()

        model = gfpgan_module.face_helper_module.init_detection_model(
            "retinaface_resnet50", device="cuda:0"
        )

        # 重みキーから `module.` が外され、モデルが推論用設定で初期化されることを確認する。
        assert isinstance(model, DummyRetinaFace)
        assert captured["network_name"] == "resnet50"
        assert captured["half"] is False
        assert captured["device"] == "cuda:0"
        assert captured["strict"] is True
        assert captured["eval_called"] is True
        assert captured["to_device"] == "cuda:0"
        assert list(captured["state_dict"].keys()) == ["weight"]

    @pytest.mark.parametrize(
        ("call_initializer", "match"),
        [
            (
                lambda: gfpgan_module.face_helper_module.init_detection_model("unsupported"),
                "is not implemented in this class",
            ),
            (
                lambda: gfpgan_module.face_helper_module.init_parsing_model(half=True),
                "half=True",
            ),
            (
                lambda: gfpgan_module.face_helper_module.init_parsing_model("unsupported"),
                "is not implemented in this class",
            ),
        ],
        ids=[
            "未対応の検出モデル名なら例外を送出する",
            "parsing_modelにhalf=Trueを渡すと例外を送出する",
            "未対応のパースモデル名なら例外を送出する",
        ],
    )
    def test_custom_initializers_reject_invalid_args(
        self, call_initializer, match: str
    ) -> None:
        """未対応の初期化引数やモデル名が指定された場合に例外を送出することを確認する。"""
        gfpgan = Gfpgan()
        gfpgan._patch_facexlib_initializers()

        # サポート外のモデル名や引数を渡し、想定どおり早期に失敗することを確認する。
        with pytest.raises(NotImplementedError, match=match):
            call_initializer()

    @pytest.mark.parametrize(
        ("paste_back", "restored_faces", "restored_img", "match"),
        [
            (True, [], None, "restored_img is None"),
            (False, [], None, "No face detected"),
        ],
        ids=[
            "paste_back有効で復元画像がなければ例外を送出する",
            "paste_back無効で顔画像がなければ例外を送出する",
        ],
    )
    def test_enhance_image_raises_for_missing_outputs(
        self,
        paste_back: bool,
        restored_faces: list[np.ndarray],
        restored_img: np.ndarray | None,
        match: str,
    ) -> None:
        """必要な復元結果が得られない場合に適切な例外を送出することを確認する。"""
        gfpgan = Gfpgan()

        class DummyModel:
            """復元結果が欠けている異常系を再現するダミーモデル。"""

            def enhance(self, image, **kwargs):
                return None, restored_faces, restored_img

        # `paste_back` の設定に応じて必要な結果が欠けている場合は失敗することを確認する。
        with pytest.raises(RuntimeError, match=match):
            gfpgan._enhance_image(
                model=DummyModel(),
                image_np=np.zeros((1, 1, 3), dtype=np.uint8),
                paste_back=paste_back,
            )

    @pytest.mark.parametrize(
        ("paste_back", "restored_faces", "restored_img", "expected"),
        [
            (
                True,
                [],
                np.full((2, 2, 3), 7, dtype=np.uint8),
                np.full((2, 2, 3), 7, dtype=np.uint8),
            ),
            (
                False,
                [np.full((4, 4, 3), 9, dtype=np.uint8)],
                None,
                np.full((4, 4, 3), 9, dtype=np.uint8),
            ),
        ],
        ids=[
            "paste_back有効なら復元済み全体画像を返す",
            "paste_back無効なら復元済み顔画像を返す",
        ],
    )
    def test_enhance_image_returns_expected_output(
        self,
        paste_back: bool,
        restored_faces: list[np.ndarray],
        restored_img: np.ndarray | None,
        expected: np.ndarray,
    ) -> None:
        """`paste_back` 設定に応じて適切な復元画像を返すことを確認する。"""
        gfpgan = Gfpgan()
        image_np = np.zeros(expected.shape, dtype=np.uint8)
        captured = {}

        class DummyModel:
            """enhance 呼び出し内容を記録し、指定された復元結果を返すダミーモデル。"""

            def enhance(self, image, **kwargs):
                # 呼び出し時の入力画像とオプションを保存する。
                captured["image"] = image
                captured["kwargs"] = kwargs
                return None, restored_faces, restored_img

        # `paste_back` の分岐ごとに返却される画像を確認する。
        output = gfpgan._enhance_image(
            model=DummyModel(),
            image_np=image_np,
            has_aligned=True,
            only_center_face=True,
            paste_back=paste_back,
        )

        # 返却値とモデルへの委譲内容が期待どおりであることを確認する。
        assert np.array_equal(output, expected)
        assert captured["image"] is image_np
        assert captured["kwargs"] == {
            "has_aligned": True,
            "only_center_face": True,
            "paste_back": paste_back,
        }

    def test_custom_parsing_initializer_loads_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """パースモデル初期化時に重みを読み込み、推論用設定で返すことを確認する。"""
        gfpgan = Gfpgan()
        captured = {}

        class DummyParseNet:
            """ParseNet の初期化内容とロード状態を記録するダミーモデル。"""

            def __init__(self, in_size, out_size, parsing_ch):
                # 生成時のネットワーク設定を保持する。
                captured["in_size"] = in_size
                captured["out_size"] = out_size
                captured["parsing_ch"] = parsing_ch

            def load_state_dict(self, state_dict, strict):
                # 読み込まれた重みを保持する。
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def eval(self):
                # 推論モードへの切り替えを記録する。
                captured["eval_called"] = True
                return self

            def to(self, device):
                # デバイス移動を記録する。
                captured["to_device"] = device
                return self

        # 実モデル依存を排除し、重みロードと初期化パラメータに絞って検証する。
        monkeypatch.setattr("app.ml.gfpgan.ParseNet", DummyParseNet)
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location: {"weight": torch.tensor([2.0])},
        )

        # 差し替え済み初期化関数からパースモデルを生成する。
        gfpgan._patch_facexlib_initializers()

        model = gfpgan_module.face_helper_module.init_parsing_model("parsenet", device="cuda:1")

        # 既定の入出力サイズと重みがそのまま使われることを確認する。
        assert isinstance(model, DummyParseNet)
        assert captured["in_size"] == 512
        assert captured["out_size"] == 512
        assert captured["parsing_ch"] == 19
        assert captured["strict"] is True
        assert captured["eval_called"] is True
        assert captured["to_device"] == "cuda:1"
        assert list(captured["state_dict"].keys()) == ["weight"]
        assert torch.equal(captured["state_dict"]["weight"], torch.tensor([2.0]))

    def test_load_model_builds_gfpganer_with_expected_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """モデル読み込み時に GFPGANer が期待どおりの引数で構築されることを確認する。"""
        gfpgan = Gfpgan()
        captured = {}

        class DummyGFPGANer:
            """GFPGANer のコンストラクタ引数を記録するダミークラス。"""

            def __init__(self, **kwargs):
                captured.update(kwargs)

        # 前処理パッチ適用と GFPGANer 生成を差し替え、渡される引数だけを検証する。
        monkeypatch.setattr(
            gfpgan,
            "_patch_facexlib_initializers",
            lambda: captured.setdefault("patched", True),
        )
        monkeypatch.setattr("app.ml.gfpgan.GFPGANer", DummyGFPGANer)

        # モデル構築処理を実行する。
        model = gfpgan._load_model()

        # ラッパー生成時に必要な設定値がすべて渡されることを確認する。
        assert isinstance(model, DummyGFPGANer)
        assert captured["patched"] is True
        assert captured["model_path"] == str(gfpgan.weight_path)
        assert captured["upscale"] == gfpgan.upscale
        assert captured["arch"] == gfpgan.arch
        assert captured["channel_multiplier"] == gfpgan.channel_multiplier
        assert captured["bg_upsampler"] is None
        assert captured["device"] == gfpgan.device


    def test_processing_loads_model_and_enhances_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """processing がモデル読み込み後に RGB/BGR を変換して補正処理を行うことを確認する。"""
        gfpgan = Gfpgan()
        image_np = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        expected_bgr = np.array(
            [
                [[5, 15, 25], [35, 45, 55]],
                [[65, 75, 85], [95, 105, 115]],
            ],
            dtype=np.uint8,
        )
        dummy_model = object()
        captured = {}

        # モデル読み込みと画像補正を差し替え、processing の配線だけを検証する。
        monkeypatch.setattr(gfpgan, "_load_model", lambda: dummy_model)

        def fake_enhance_image(model, image_np, **kwargs):
            # 補正処理へ渡された入力とオプションを記録する。
            captured["model"] = model
            captured["image_np"] = image_np
            captured["kwargs"] = kwargs
            return expected_bgr

        monkeypatch.setattr(gfpgan, "_enhance_image", fake_enhance_image)

        # 公開メソッドを実行し、内部での色順変換と委譲内容を確認する。
        output = gfpgan.processing(
            image_np=image_np,
            has_aligned=True,
            only_center_face=False,
            paste_back=True,
        )

        # 内部では BGR で処理し、最終出力は RGB に戻されることを確認する。
        assert np.array_equal(output, expected_bgr[:, :, ::-1])
        assert captured["model"] is dummy_model
        assert np.array_equal(captured["image_np"], image_np[:, :, ::-1])
        assert captured["kwargs"] == {
            "has_aligned": True,
            "only_center_face": False,
            "paste_back": True,
        }

    def test_processing(self) -> None:
        """実行結果の画像が期待する出力画像と十分近いことを確認する。"""
        # テスト用入力画像と期待画像を読み込み、実際の推論結果と比較できるようにする。
        test_image_np = np.asarray(Image.open(TEST_IMAGE_PATH).convert("RGB"))
        expected = np.asarray(Image.open(RESULT_IMAGE_PATH).convert("RGB"))

        # 実際の processing を実行し、出力画像との差分を計算する。
        image_np = Gfpgan().processing(image_np=test_image_np)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        # 型とサイズに加え、画素差分が十分小さいことを確認する。
        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
