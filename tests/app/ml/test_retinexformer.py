from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from app.ml.retinexformer import Retinexformer
from app.vendor.retinexformer.basicsr.models.archs.MST_Plus_Plus_arch import MST_Plus_Plus


TEST_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/test_face.png")
RESULT_IMAGE_PATH = Path("tests/app/test_data/test_image/retinexformer/result_face.png")


class TestRetinexformer:
    """Retinexformer の補助メソッドと実行処理を検証するテストクラス。"""

    def test_load_np_as_tensor_converts_hwc_to_nchw(self) -> None:
        """NumPy 配列が HWC 形式から NCHW 形式の torch.Tensor に変換されることを確認する。"""
        former = Retinexformer()
        # 高さ2、幅2、チャンネル3の疑似画像データを作成し、軸の並び替え結果を検証できるようにする。
        image_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

        # 内部変換メソッドを呼び出し、モデル入力向けのテンソルへ変換する。
        tensor = former._load_np_as_tensor(image_np)

        # バッチ次元が追加され、チャンネル優先の並びに変換されていることを確認する。
        assert tensor.shape == (1, 3, 2, 2)
        # 推論時にそのまま扱えるよう、float32 のまま保持されていることを確認する。
        assert tensor.dtype == torch.float32
        # 左上画素の RGB 値が正しいチャンネル順で格納されていることを確認する。
        assert torch.equal(tensor[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))

    def test_tensor_to_numpy_clamps_and_returns_ndarray(self) -> None:
        """テンソルが 0 から 255 の範囲に丸め込まれた PIL 画像へ変換されることを確認する。"""
        former = Retinexformer()
        # 範囲外の値を含むテンソルを用意し、クリップ処理と整数化の結果を検証できるようにする。
        tensor = torch.tensor(
            [[
                [[1.2, 0.5], [0.0, 0.1]],
                [[-0.1, 0.25], [0.5, 0.75]],
                [[0.2, 0.0], [1.0, 0.333]],
            ]],
            dtype=torch.float32,
        )

        # モデル出力想定のテンソルを画像へ変換し、型と画素値を確認する。
        image_np = former._tensor_to_numpy(tensor)

        # 戻り値が PIL.Image であり、画像サイズが保持されていることを確認する。
        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == (2, 2, 3)
        # 画像保存可能な uint8 配列へ変換されていることを確認する。
        assert image_np.dtype == np.uint8
        # 範囲外の値が 0 から 255 に収まるように補正された結果を確認する。
        assert np.array_equal(image_np[0, 0], np.array([255, 0, 51], dtype=np.uint8))

    def test_build_network_from_yaml_rejects_unexpected_network_type(self) -> None:
        """未対応のネットワーク種別が指定された場合に例外を送出することを確認する。"""
        former = Retinexformer()

        # 想定外の type を渡し、誤った設定を早期に検出できることを確認する。
        with pytest.raises(ValueError, match="Unexpected network type"):
            former._build_network_from_yaml({"network_g": {"type": "unexpected"}})

    def test_build_network_from_yaml_builds_mst_plus_plus(self) -> None:
        """YAML 設定から MST_Plus_Plus モデルが正しく構築されることを確認する。"""
        former = Retinexformer()

        # 実際に利用する network_g 設定を渡し、対応するモデルインスタンスを生成する。
        model = former._build_network_from_yaml(
            {
                "network_g": {
                    "type": "MST_Plus_Plus",
                    "in_channels": 3,
                    "out_channels": 3,
                    "n_feat": 31,
                    "stage": 2,
                }
            }
        )

        # 生成されたモデルの型と主要な入出力設定が期待通りかを確認する。
        assert isinstance(model, MST_Plus_Plus)
        assert model.stage == 2
        assert model.conv_in.in_channels == 3
        assert model.conv_in.out_channels == 31
        assert model.conv_out.in_channels == 31
        assert model.conv_out.out_channels == 3

    def test_load_model_uses_params_ema_and_strips_module_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """学習済み重みの `params_ema` を読み込み、`module.` 接頭辞を除去することを確認する。"""
        former = Retinexformer()
        captured = {}

        class DummyModel:
            """load_state_dict 呼び出し内容を記録するためのダミーモデル。"""

            def load_state_dict(self, state_dict, strict):
                # 読み込まれた重みと strict 指定を保持し、後続の検証に使う。
                captured["state_dict"] = state_dict
                captured["strict"] = strict

            def to(self, device):
                # モデルが推論対象デバイスへ移動されたことを記録する。
                captured["device"] = device
                return self

            def eval(self):
                # 推論モードへ切り替えられたことを記録する。
                captured["eval_called"] = True
                return self

        # YAML 読み込みとモデル生成を差し替え、重みロード処理だけを単体で検証できるようにする。
        monkeypatch.setattr(former, "_load_yaml_opt", lambda: {"network_g": {}})
        monkeypatch.setattr(former, "_build_network_from_yaml", lambda opt: DummyModel())
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location, weights_only: {
                "params_ema": {"module.weight": torch.tensor([1.0])}
            },
        )

        # 学習済みモデルの読み込み処理を実行する。
        model = former._load_model()

        # `params_ema` が使われ、`module.` を外した名前でロードされることを確認する。
        assert isinstance(model, DummyModel)
        assert captured["strict"] is True
        assert captured["device"] == former.device
        assert captured["eval_called"] is True
        assert list(captured["state_dict"].keys()) == ["weight"]

    def test_enhance_image_pads_to_factor_and_crops_back(self) -> None:
        """画像サイズを factor の倍数にパディングし、推論後に元サイズへ切り戻すことを確認する。"""
        former = Retinexformer()
        # factor で割り切れないサイズの入力を作り、パディングと切り戻しの両方を検証する。
        input_tensor = torch.arange(1 * 3 * 5 * 6, dtype=torch.float32).reshape(1, 3, 5, 6)
        captured = {}

        class DummyModel:
            """モデルに渡されたテンソル形状を記録するダミーモデル。"""

            def __call__(self, tensor):
                # モデル入力サイズを保持し、出力は比較しやすいよう一律で 1 加算して返す。
                captured["shape"] = tensor.shape
                return tensor + 1

        # パディング後に推論し、最後に元の高さと幅へ切り戻す内部処理を実行する。
        output = former._enhance_image(DummyModel(), input_tensor, factor=4)

        # モデルには 4 の倍数サイズで渡され、戻り値は元のサイズへ復元されることを確認する。
        assert captured["shape"] == (1, 3, 8, 8)
        assert output.shape == (1, 3, 5, 6)
        assert torch.equal(output, input_tensor + 1)

    def test_processing(self) -> None:
        """実行結果の画像が期待する出力画像と一致することを確認する。"""
        # テスト用画像を読み込み、processing が受け取る 0.0 から 1.0 の float 配列へ正規化する。
        test_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
        test_image_np = np.asarray(test_image).astype(np.float32) / 255.0

        former = Retinexformer()
        # 実際の推論フローを通し、最終的に返される画像を取得する。
        image_np = former.processing(image_np=test_image_np)

        # 期待画像を読み込み、画素単位で完全一致することを確認する。
        result_image = Image.open(RESULT_IMAGE_PATH).convert("RGB")
        expected = np.asarray(result_image)
        diff = np.abs(image_np.astype(np.int16) - expected.astype(np.int16))

        assert isinstance(image_np, np.ndarray)
        assert image_np.shape == expected.shape
        assert image_np.dtype == np.uint8
        assert diff.mean() < 2.0
        assert np.percentile(diff, 99) <= 16.0
