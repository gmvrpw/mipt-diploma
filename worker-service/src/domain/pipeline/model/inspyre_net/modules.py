from operator import xor
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding="same",
        bias: bool = False,
        bn: bool = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        if "__iter__" not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if "__iter__" not in dir(stride):
            stride = (stride, stride)
        if "__iter__" not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == "same":
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == "valid":
            width_pad_size = 0
            height_pad_size = 0
        else:
            if "__iter__" in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, pad_size,
            dilation, groups, bias=bias,
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, mode: str = "hw", stage_size=None) -> None:
        super().__init__()
        self.mode = mode

        self.query_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.stage_size = stage_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = x.size()

        axis = 1
        if "h" in self.mode:
            axis *= height
        if "w" in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        return self.gamma * out + x


class PAA_kernel(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        receptive_size: int,
        stage_size=None,
    ) -> None:
        super().__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 1)
        self.conv1 = Conv2d(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = Conv2d(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = Conv2d(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = SelfAttention(
            out_channel, "h", stage_size[0] if stage_size is not None else None,
        )
        self.Wattn = SelfAttention(
            out_channel, "w", stage_size[1] if stage_size is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        Hx = self.Hattn(x)
        Wx = self.Wattn(x)
        return self.conv3(Hx + Wx)


class PAA_e(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        base_size=None,
        stage=None,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        self.branch0 = Conv2d(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3, self.stage_size)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5, self.stage_size)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7, self.stage_size)

        self.conv_cat = Conv2d(4 * out_channel, out_channel, 3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))


class PAA_d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int = 1,
        depth: int = 64,
        base_size=None,
        stage=None,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channel, depth, 3)
        self.conv2 = Conv2d(depth, depth, 3)
        self.conv3 = Conv2d(depth, depth, 3)
        self.conv4 = Conv2d(depth, depth, 3)
        self.conv5 = Conv2d(depth, out_channel, 3, bn=False)

        self.base_size = base_size
        self.stage = stage

        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = [None, None]

        self.Hattn = SelfAttention(depth, "h", self.stage_size[0])
        self.Wattn = SelfAttention(depth, "w", self.stage_size[1])

    @staticmethod
    def _upsample(img: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(img, size=size, mode="bilinear", align_corners=True)

    def forward(self, fs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        fx = fs[0]
        for i in range(1, len(fs)):
            fs[i] = self._upsample(fs[i], fx.shape[-2:])
        fx = torch.cat(fs[::-1], dim=1)

        fx = self.conv1(fx)
        Hfx = self.Hattn(fx)
        Wfx = self.Wattn(fx)

        fx = self.conv2(Hfx + Wfx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        out = self.conv5(fx)

        return fx, out


class SICA(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int = 1,
        depth: int = 64,
        base_size=None,
        stage=None,
        lmap_in: bool = False,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        self.conv_query = nn.Sequential(
            Conv2d(in_channel, depth, 3, relu=True),
            Conv2d(depth, depth, 3, relu=True),
        )
        self.conv_key = nn.Sequential(
            Conv2d(in_channel, depth, 1, relu=True),
            Conv2d(depth, depth, 1, relu=True),
        )
        self.conv_value = nn.Sequential(
            Conv2d(in_channel, depth, 1, relu=True),
            Conv2d(depth, depth, 1, relu=True),
        )

        self.ctx = 5 if lmap_in else 3

        self.conv_out1 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out2 = Conv2d(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))
        if lmap_in:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(
        self,
        x: torch.Tensor,
        smap: torch.Tensor,
        lmap: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not xor(self.lmap_in is True, lmap is not None)
        b, _, h, w = x.shape

        smap = F.interpolate(smap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1)
        bg = torch.clip(-p, 0, 1)
        cg = self.threshold - torch.abs(p)

        if self.lmap_in and lmap is not None:
            lmap = F.interpolate(
                lmap, size=x.shape[-2:], mode="bilinear", align_corners=False,
            )
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1)
            bp = torch.clip(-lp, 0, 1)
            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w

        f = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode="bilinear", align_corners=False).view(b, self.ctx, shape_mul)

        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)

        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        sim = torch.bmm(query, key)
        sim = (self.depth ** -0.5) * sim
        sim = F.softmax(sim, dim=-1)

        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out


def _ellipse_kernel(k: int) -> torch.Tensor:
    r = (k - 1) / 2.0
    inv_r = 1.0 / r if r > 0 else 1.0
    coords = torch.arange(k, dtype=torch.float32) - r
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    mask = (xx * inv_r) ** 2 + (yy * inv_r) ** 2 <= 1.0
    return mask.float()


class Transition:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self._kernel = _ellipse_kernel(k)

    def to(self, device, dtype=None):
        if dtype is not None:
            self._kernel = self._kernel.to(device=device, dtype=dtype)
        else:
            self._kernel = self._kernel.to(device=device)
        return self

    def _morph(self, x: torch.Tensor, op: str) -> torch.Tensor:
        # binary morphology with arbitrary structuring element via unfold:
        # dilation = max over neighborhood positions where kernel != 0
        # erosion  = min over the same positions
        b, c, h, w = x.shape
        k = self.k
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        patches = F.unfold(x_pad, kernel_size=(k, k))
        patches = patches.view(b, c, k * k, h * w)
        mask = self._kernel.flatten().to(device=x.device).bool()
        patches = patches[:, :, mask, :]
        if op == "dilation":
            out = patches.max(dim=2).values
        else:
            out = patches.min(dim=2).values
        return out.view(b, c, h, w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        dx = self._morph(x, "dilation")
        ex = self._morph(x, "erosion")
        return ((dx - ex) > 0.5).float()