from __future__ import annotations

from io import BytesIO
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


ML_ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def default_init_weights(module_list, scale: float = 1.0, bias_fill: float = 0.0, **kwargs) -> None:
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for submodule in module.modules():
            if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(submodule.weight, **kwargs)
                submodule.weight.data *= scale
                if submodule.bias is not None:
                    submodule.bias.data.fill_(bias_fill)
            elif isinstance(submodule, _BatchNorm):
                init.constant_(submodule.weight, 1)
                if submodule.bias is not None:
                    submodule.bias.data.fill_(bias_fill)


class NormStyleCode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        default_init_weights(
            self.modulation,
            scale=1,
            bias_fill=1,
            a=0,
            mode="fan_in",
            nonlinearity="linear",
        )

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        style = self.modulation(style).view(batch_size, 1, channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch_size * self.out_channels,
            channels,
            self.kernel_size,
            self.kernel_size,
        )

        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

        _, channels, height, width = x.shape
        x = x.view(1, batch_size * channels, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        return out.view(batch_size, self.out_channels, *out.shape[2:4])


class StyleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
        )
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.modulated_conv(x, style) * 2**0.5
        if noise is None:
            batch_size, _, height, width = out.shape
            noise = out.new_empty(batch_size, 1, height, width).normal_()
        out = out + self.weight * noise
        out = out + self.bias
        return self.activate(out)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, num_style_feat: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            3,
            kernel_size=1,
            num_style_feat=num_style_feat,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.modulated_conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel: int, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.repeat(batch_size, 1, 1, 1)


class StyleGAN2GeneratorClean(nn.Module):
    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_style_feat = num_style_feat

        style_mlp_layers: list[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            style_mlp_layers.extend(
                [
                    nn.Linear(num_style_feat, num_style_feat, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        default_init_weights(
            self.style_mlp,
            scale=1,
            bias_fill=0,
            a=0.2,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(
            channels["4"],
            channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
        )
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels["4"]
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(1, 1, resolution, resolution))

        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    sample_mode="upsample",
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                )
            )
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def forward(
        self,
        styles: list[torch.Tensor],
        input_is_latent: bool = False,
        noise: list[torch.Tensor | None] | None = None,
        randomize_noise: bool = True,
        truncation: float = 1,
        truncation_latent: torch.Tensor | None = None,
        inject_index: int | None = None,
        return_latents: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not input_is_latent:
            styles = [self.style_mlp(style) for style in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]

        if truncation < 1:
            truncated = []
            for style in styles:
                truncated.append(truncation_latent + truncation * (style - truncation_latent))
            styles = truncated

        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], dim=1)
        else:
            raise ValueError("styles must contain one or two tensors")

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        index = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, index], noise=noise1)
            out = conv2(out, latent[:, index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, index + 2], skip)
            index += 2

        if return_latents:
            return skip, latent
        return skip, None


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1.0,
        sft_half: bool = False,
    ) -> None:
        super().__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
        )
        self.sft_half = sft_half

    def forward(
        self,
        styles: list[torch.Tensor],
        conditions: list[torch.Tensor],
        input_is_latent: bool = False,
        noise: list[torch.Tensor | None] | None = None,
        randomize_noise: bool = True,
        truncation: float = 1,
        truncation_latent: torch.Tensor | None = None,
        inject_index: int | None = None,
        return_latents: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not input_is_latent:
            styles = [self.style_mlp(style) for style in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]

        if truncation < 1:
            truncated = []
            for style in styles:
                truncated.append(truncation_latent + truncation * (style - truncation_latent))
            styles = truncated

        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], dim=1)
        else:
            raise ValueError("styles must contain one or two tensors")

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        index = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, index], noise=noise1)

            if index < len(conditions):
                if self.sft_half:
                    out_same, out_sft = torch.split(out, out.size(1) // 2, dim=1)
                    out_sft = out_sft * conditions[index - 1] + conditions[index]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * conditions[index - 1] + conditions[index]

            out = conv2(out, latent[:, index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, index + 2], skip)
            index += 2

        if return_latents:
            return skip, latent
        return skip, None


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "down") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == "down":
            self.scale_factor = 0.5
        elif mode == "up":
            self.scale_factor = 2
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)

        skip = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        skip = self.skip(skip)
        return out + skip


class GFPGANv1Clean(nn.Module):
    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        channel_multiplier: int = 1,
        decoder_load_path: str | Path | None = None,
        fix_decoder: bool = True,
        num_mlp: int = 8,
        input_is_latent: bool = False,
        different_w: bool = False,
        narrow: float = 1.0,
        sft_half: bool = False,
    ) -> None:
        super().__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            "4": int(512 * unet_narrow),
            "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow),
            "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2 ** self.log_size

        self.conv_body_first = nn.Conv2d(3, channels[f"{first_out_size}"], 1)

        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2**(i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels["4"], 3, 1, 1)

        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode="up"))
            in_channels = out_channels

        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f"{2**i}"], 3, 1))

        if different_w:
            linear_out_channel = (self.log_size * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out_channel)

        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half,
        )

        if decoder_load_path:
            decoder_state = torch.load(decoder_load_path, map_location="cpu")["params_ema"]
            self.stylegan_decoder.load_state_dict(decoder_state)

        if fix_decoder:
            for parameter in self.stylegan_decoder.parameters():
                parameter.requires_grad = False

        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            sft_out_channels = out_channels if sft_half else out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False,
        return_rgb: bool = True,
        randomize_noise: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        conditions: list[torch.Tensor] = []
        unet_skips: list[torch.Tensor] = []
        out_rgbs: list[torch.Tensor] = []

        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for down_block in self.conv_body_down:
            feat = down_block(feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        for i, up_block in enumerate(self.conv_body_up):
            feat = feat + unet_skips[i]
            feat = up_block(feat)
            scale = self.condition_scale[i](feat)
            shift = self.condition_shift[i](feat)
            conditions.append(scale.clone())
            conditions.append(shift.clone())
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        image, _ = self.stylegan_decoder(
            [style_code],
            conditions,
            return_latents=return_latents,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise,
        )
        return image, out_rgbs


def load_gfpgan_model(
    weight_bytes: BytesIO,
    device: torch.device | str,
    channel_multiplier: int = 2,
) -> GFPGANv1Clean:
    runtime_device = torch.device(device)
    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=channel_multiplier,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    ).to(runtime_device)

    loadnet = torch.load(weight_bytes, map_location=runtime_device)
    keyname = "params_ema" if "params_ema" in loadnet else "params"
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    return model
