from __future__ import annotations

from collections import OrderedDict
from io import BytesIO

import torch
from torch import nn
from torch.nn import functional as F


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"Input spatial size must be divisible by scale={scale}.")

    out_h = h // scale
    out_w = w // scale
    x = x.view(b, c, out_h, scale, out_w, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * scale * scale, out_h, out_w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat=num_feat, num_grow_ch=num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        *,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale

        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_hr(feat))
        return self.conv_last(feat)


def load_realesrgan_model(
    weight_bytes: BytesIO,
    device: torch.device,
    scale: int,
    num_feat: int = 64,
    num_block: int = 23,
    num_grow_ch: int = 32,
) -> RRDBNet:
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        scale=scale,
    )

    checkpoint = torch.load(weight_bytes, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "")] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model
