
import math
from typing import Any, Callable, List, Optional, Tuple, Union

from mmcv.ops import fused_bias_leakyrelu, upfirdn2d
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import timm


class Print(nn.Module):
    def __init__(self, prompt: str = ""):
        super().__init__()
        self.prompt = prompt

    def forward(self, input):
        print(self.prompt, ">>>", input.shape)
        return input

class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)


class Activation(nn.Module):
    activation: Union[nn.Module, Callable[[Any], torch.Tensor]]

    def __init__(self, activation: str, use_scale: bool, *args, **kwargs):
        super(Activation, self).__init__()
        self.use_scale = use_scale
        if activation == "relu":
            self.activation = nn.ReLU(*args, **kwargs)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(*args, **kwargs)
        elif activation == "prelu":
            self.activation = nn.PReLU(*args, **kwargs)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = Swish()
        else:
            assert False, f"unsupported activation: {activation}"

    def forward(self, input):
        if self.use_scale:
            return self.activation(input) * math.sqrt(2)
        else:
            return self.activation(input)


class NoiseBlock(nn.Module):
    def __init__(self):
        super(NoiseBlock, self).__init__()
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        return x + self.scale * torch.randn(B, 1, H, W, device=x.device)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
                bias=False),
            nn.BatchNorm2d(num_features=out_channels * 2),
            nn.GLU(dim=1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class UpsampleBlock2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock2, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
                bias=False),
            # NoiseBlock(),
            # nn.BatchNorm2d(num_features=out_channels * 2),
            # nn.GLU(dim=1),
            # nn.Conv2d(
            #     in_channels=out_channels,
            #     out_channels=out_channels * 2,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     padding_mode="replicate",
            #     bias=False),
            NoiseBlock(),
            nn.BatchNorm2d(num_features=out_channels * 2),
            nn.GLU(dim=1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class SkipLayerExcitation(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int):
        super(SkipLayerExcitation, self).__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Conv2d(
                in_channels=feature_channels,
                out_channels=in_channels,
                kernel_size=4,
                bias=False),
            Swish(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False),
            nn.Sigmoid())

    def forward(self, input: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return input * self.channel_wise(feature)


class Generator(nn.Module):
    def __init__(self, channels_info: List[Tuple[int, int]], latent_dim: int = 256):
        super(Generator, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=channels_info[0][0] * 2,
                kernel_size=4,
                bias=False),
            nn.BatchNorm2d(num_features=channels_info[0][0] * 2),
            nn.GLU(dim=1))
        self.feature_layers = (len(channels_info) - 2) // 2
        self.pre_upsample_layers = len(channels_info) % 2 + 1
        self.upsample_layers = nn.ModuleList([
            UpsampleBlock2(in_channels=in_channels, out_channels=out_channels)
            if i % 2 == 0 else
            UpsampleBlock(in_channels=in_channels, out_channels=out_channels)
            for i, (in_channels, out_channels) in enumerate(channels_info)])
        self.skip_layer_excitation_layers = nn.ModuleList([
            SkipLayerExcitation(in_channels=in_channels, feature_channels=feature_channels)
            for (_, in_channels), (_, feature_channels) in zip(
                channels_info[self.pre_upsample_layers + self.feature_layers: -1],
                channels_info[self.pre_upsample_layers - 1:
                             self.pre_upsample_layers + self.feature_layers - 1])])
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_info[-1][1],
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
                bias=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.normalize(input, dim=1)
        x = self.conv_trans(x)
        buf: List[torch.Tensor] = []
        skip_layer_excitation_start_idx = self.feature_layers + self.pre_upsample_layers
        for idx, upsample in enumerate(self.upsample_layers):
            x = upsample(x)
            if idx < skip_layer_excitation_start_idx - 1:
                buf.append(x)
            if skip_layer_excitation_start_idx <= idx < len(self.upsample_layers) - 1:
                x = self.skip_layer_excitation_layers[
                        idx - skip_layer_excitation_start_idx](x, buf.pop(0))
        return self.out(x)


class Projector(nn.Module):
    def __init__(self,
                 feature_channels: List[int],
                 projected_channels: List[int],
                 model_name: str = "tf_efficientnet_lite1"):
        super(Projector, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.ccm = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False)
            for in_channels, out_channels in zip(feature_channels, projected_channels)])
        for module in self.ccm:
            nn.init.kaiming_uniform_(module.weight.data, a=0.2)
        self.csm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            if i != 0 else
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False)
            for i, (in_channels, out_channels) in
                enumerate(zip(
                    [projected_channels[-1], *projected_channels[::-1]],
                    projected_channels[::-1],
                    ))])
        for module in self.csm:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight.data, a=0.2)
            else:
                nn.init.kaiming_uniform_(module[0].weight.data, a=0.2)

    def forward(self, input: torch.Tensor
            ) -> Tuple[torch.Tensor,
                       torch.Tensor,
                       torch.Tensor,
                       torch.Tensor]:
        x = self.model.conv_stem(input)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x64 = self.model.blocks[0:2](x)
        x32 = self.model.blocks[2](x64)
        x16 = self.model.blocks[3:5](x32)
        x8 = self.model.blocks[5:](x16)
        x8 = self.model.conv_head(x8)
        x8 = self.model.bn2(x8)

        x64 = self.ccm[0](x64)
        x32 = self.ccm[1](x32)
        x16 = self.ccm[2](x16)
        x8 = self.ccm[3](x8)

        x8 = self.csm[0](x8)
        x16 = self.csm[1](x8) + x16
        x32 = self.csm[2](x16) + x32
        x64 = self.csm[3](x32) + x64
        return x64, x32, x16, x8

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, slope: float = 0.2):
        super(DownBlock, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="replicate",
                bias=False)),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.LeakyReLU(negative_slope=slope, inplace=True),
            Swish())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class Discriminator(nn.Module):
    def __init__(self,
                 projected_channels: List[int],
                 channel_info: List[Tuple[int, int]]):
        super(Discriminator, self).__init__()
        in_channels64, in_channels32, in_channels16, in_channels8 = projected_channels
        self.model64 = nn.Sequential(
            DownBlock(
                in_channels=in_channels64,
                out_channels=channel_info[0][0])
            ,*[
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels)
                for in_channels, out_channels in channel_info
            ],
            spectral_norm(
                nn.Conv2d(
                    in_channels=channel_info[-1][1],
                    out_channels=1,
                    kernel_size=4)))
        self.model32 = nn.Sequential(
            DownBlock(
                in_channels=in_channels32,
                out_channels=channel_info[1][0])
            ,*[
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels)
                for in_channels, out_channels in channel_info[1:]
            ],
            spectral_norm(
                nn.Conv2d(
                    in_channels=channel_info[-1][1],
                    out_channels=1,
                    kernel_size=4)))
        self.model16 = nn.Sequential(
            DownBlock(
                in_channels=in_channels16,
                out_channels=channel_info[2][0])
            ,*[
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels)
                for in_channels, out_channels in channel_info[2:]
            ],
            spectral_norm(
                nn.Conv2d(
                    in_channels=channel_info[-1][1],
                    out_channels=1,
                    kernel_size=4)))
        self.model8 = nn.Sequential(
            DownBlock(
                in_channels=in_channels8,
                # channel_info[3][0] == channel_info[2][1]
                out_channels=channel_info[2][1])
            ,*[
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels)
                for in_channels, out_channels in channel_info[3:]
            ],
            spectral_norm(
                nn.Conv2d(
                    in_channels=channel_info[-1][1],
                    out_channels=1,
                    kernel_size=4)))

    def forward(self,
                input64: torch.Tensor,
                input32: torch.Tensor,
                input16: torch.Tensor,
                input8: torch.Tensor
            ) -> Tuple[torch.Tensor,
                       torch.Tensor,
                       torch.Tensor,
                       torch.Tensor]:
        return self.model64(input64), \
               self.model32(input32), \
               self.model16(input16), \
               self.model8(input8)

