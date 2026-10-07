"""On-GPU data augmentation for a batch of (N, 1, 48, 48) face images.

Every time the model sees a face it gets a slightly different version:
mirrored, rotated, shifted, zoomed, with a small patch erased. The model can no
longer memorize exact pixels and has to learn what actually signals an emotion.
"""
import math

import torch
import torch.nn.functional as F


def augment(x, max_rotate=10, max_shift=0.1, scale_range=(0.9, 1.1), erase_prob=0.5, erase_size=12):
    n, device = x.shape[0], x.device

    # Random horizontal flip — a mirrored smile is still a smile.
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)

    # Random rotation / shift / zoom combined into one affine transform.
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_rotate)
    scale = torch.empty(n, device=device).uniform_(*scale_range)
    tx, ty = ((torch.rand(2, n, device=device) * 2 - 1) * max_shift * 2)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], dim=1),
        torch.stack([sin * flip, cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # Random erasing — hide a small square so the model can't rely on one spot.
    h, w = x.shape[-2:]
    cy = torch.randint(0, h, (n, 1, 1), device=device)
    cx = torch.randint(0, w, (n, 1, 1), device=device)
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    box = ((ys - cy).abs() < erase_size // 2) & ((xs - cx).abs() < erase_size // 2)
    box &= (torch.rand(n, 1, 1, device=device) < erase_prob)
    return x.masked_fill(box.unsqueeze(1), 0.0)  # 0 = the dataset's mean pixel


def _blur_kernels(n, k, device, blur_prob):
    """One k x k kernel per image: identity, Gaussian (out of focus) or a line (motion blur)."""
    r = k // 2
    ys, xs = torch.meshgrid(torch.arange(-r, r + 1, device=device, dtype=torch.float32),
                            torch.arange(-r, r + 1, device=device, dtype=torch.float32), indexing="ij")
    sigma = torch.empty(n, 1, 1, device=device).uniform_(0.6, 1.6)
    gauss = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    angle = torch.rand(n, 1, 1, device=device) * math.pi
    length = torch.empty(n, 1, 1, device=device).uniform_(2.0, float(k))
    along = xs * torch.cos(angle) + ys * torch.sin(angle)
    across = -xs * torch.sin(angle) + ys * torch.cos(angle)
    motion = ((across.abs() <= 0.5) & (along.abs() <= length / 2)).float()
    identity = ((xs == 0) & (ys == 0)).float().expand(n, k, k)
    kind = torch.rand(n, 1, 1, device=device)
    kernels = torch.where(kind < blur_prob / 2, gauss, torch.where(kind < blur_prob, motion, identity))
    return (kernels / kernels.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)


def augment_video(x, max_rotate=10, max_shift=0.06, scale_range=(0.9, 1.1), blur_prob=0.3,
                  lowres_prob=0.3, noise_prob=0.5, erase_prob=0.4):
    """Augmentation for eDv1.0: simulates what a webcam does to a face.

    x: (N, 1, H, W) floats in [0, 1] on any device. Returns the same shape.
      - geometry: mirror, small rotation / zoom / shift (keypoint jitter of the face detector)
      - lighting: brightness, contrast and gamma (dim rooms, backlight)
      - optics: Gaussian or motion blur, and low resolution (face far from the camera)
      - sensor: Gaussian noise
      - occlusion: a random patch (hand, glasses, hair)
    """
    n, _, h, w = x.shape
    device = x.device

    # geometry
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_rotate)
    scale = torch.empty(n, device=device).uniform_(*scale_range)
    tx, ty = (torch.rand(2, n, device=device) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      padding_mode="border", align_corners=False)

    # lighting
    mean = x.mean(dim=(2, 3), keepdim=True)
    contrast = torch.empty(n, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    brightness = torch.empty(n, 1, 1, 1, device=device).uniform_(-0.15, 0.15)
    gamma = torch.exp(torch.empty(n, 1, 1, 1, device=device).uniform_(math.log(0.7), math.log(1.5)))
    x = ((x - mean) * contrast + mean + brightness).clamp(1e-4, 1) ** gamma

    # optics: blur (one kernel per image, applied as a grouped convolution)
    k = 7
    kernels = _blur_kernels(n, k, device, blur_prob)
    x = F.conv2d(F.pad(x.view(1, n, h, w), (k // 2,) * 4, mode="replicate"), kernels, groups=n).view(n, 1, h, w)

    # optics: low resolution — downscale and upscale back
    low = torch.rand(n, device=device) < lowres_prob
    if low.any():
        size = int(h * float(torch.empty(1).uniform_(0.35, 0.7)))
        small = F.interpolate(x[low], size=(size, size), mode="bilinear", align_corners=False)
        x[low] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

    # sensor noise
    sigma = torch.empty(n, 1, 1, 1, device=device).uniform_(0.0, 0.05)
    sigma *= (torch.rand(n, 1, 1, 1, device=device) < noise_prob)
    x = (x + torch.randn_like(x) * sigma).clamp(0, 1)

    # occlusion
    size = torch.randint(h // 6, h // 3, (n, 1, 1), device=device)
    cy = torch.randint(0, h, (n, 1, 1), device=device)
    cx = torch.randint(0, w, (n, 1, 1), device=device)
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    box = ((ys - cy).abs() < size // 2) & ((xs - cx).abs() < size // 2)
    box &= torch.rand(n, 1, 1, device=device) < erase_prob
    return torch.where(box.unsqueeze(1), x.mean(dim=(2, 3), keepdim=True), x)
