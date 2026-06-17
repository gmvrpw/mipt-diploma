import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(ksize: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


class ImagePyramid:
    def __init__(self, ksize: int = 7, sigma: float = 1.0, channels: int = 1) -> None:
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        kernel_2d = _gaussian_kernel_2d(ksize, sigma)
        self.kernel = kernel_2d.repeat(channels, 1, 1, 1)

    def to(self, device, dtype=None):
        if dtype is not None:
            self.kernel = self.kernel.to(device=device, dtype=dtype)
        else:
            self.kernel = self.kernel.to(device=device)
        return self

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2,) * 4, mode="reflect")
        x = F.conv2d(x, kernel * 4, groups=self.channels)
        return x

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        x = F.pad(x, (self.ksize // 2,) * 4, mode="reflect")
        x = F.conv2d(x, kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x

    def deconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.reduce(x)
        expanded = self.expand(reduced)
        if x.shape != expanded.shape:
            expanded = F.interpolate(expanded, x.shape[-2:])
        laplacian = x - expanded
        return reduced, laplacian

    def reconstruct(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        if laplacian.shape[-2:] != expanded.shape[-2:]:
            laplacian = F.interpolate(
                laplacian, expanded.shape[-2:], mode="bilinear", align_corners=True,
            )
        return expanded + laplacian