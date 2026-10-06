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
