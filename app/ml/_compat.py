from __future__ import annotations
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale


def patch_torchvision_functional_tensor() -> None:
    """
    古い依存ライブラリ向けに ``torchvision.transforms.functional_tensor`` を補う。

    ``basicsr`` や、それに依存する GFPGAN / RealESRGAN は
    ``torchvision.transforms.functional_tensor`` から
    ``rgb_to_grayscale`` を import する実装を持つ。
    一方で新しい torchvision ではそのモジュールが存在しないため、
    そのまま import するとライブラリ読み込み時に失敗する。
    この関数は関連ライブラリを import する前に最小限の互換モジュールを
    ``sys.modules`` に登録し、既存実装を動かせるようにする。
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules[functional_tensor.__name__] = functional_tensor
