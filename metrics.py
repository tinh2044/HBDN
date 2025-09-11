import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
import lpips
import warnings
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Initialize LPIPS once; suppress pickle FutureWarning from inside the library load
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )
    _lpips_model = lpips.LPIPS(net="vgg")


def calculate_psnr(img1, img2, max_val=1.0):
    img1 = img1.data.cpu().numpy().astype(np.float32)
    img2 = img2.data.cpu().numpy().astype(np.float32)
    value = 0
    for i in range(img1.shape[0]):
        psnr = compare_psnr(img1[i], img2[i], data_range=max_val)
        value += psnr

    return value / img1.shape[0]


def calculate_ssim(img1, img2):
    # SSIM expects (B, C, H, W)
    return ssim(img1, img2, data_range=1.0)


def calculate_ms_ssim(img1, img2):
    # MS-SSIM expects (B, C, H, W)
    return ms_ssim(img1, img2, data_range=1.0)


def calculate_lpips(img1, img2, device="cuda"):
    model = _lpips_model.to(device)
    with torch.no_grad():
        a = img1 * 2.0 - 1.0
        b = img2 * 2.0 - 1.0
        v = model(a, b)
        return v.mean().item() if isinstance(v, torch.Tensor) else float(v)


def compute_metrics(
    img_gt, img_pred, device="cuda", scale=2, crop_border=True, y_channel=True
):
    pred = img_pred.clamp(0.0, 1.0)
    gt = img_gt.clamp(0.0, 1.0)

    if crop_border and scale > 0:
        pred = pred[..., scale:-scale, scale:-scale]
        gt = gt[..., scale:-scale, scale:-scale]

    if y_channel and pred.size(1) == 3:
        r1, g1, b1 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        r2, g2, b2 = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3]
        pred = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
        gt = 0.299 * r2 + 0.587 * g2 + 0.114 * b2

    psnr = calculate_psnr(pred, gt)
    ssim_val = calculate_ssim(pred, gt)
    ms_val = calculate_ms_ssim(pred, gt)
    if isinstance(ssim_val, torch.Tensor):
        ssim_val = ssim_val.item()
    if isinstance(ms_val, torch.Tensor):
        ms_val = ms_val.item()
    lp = calculate_lpips(img_gt.clamp(0, 1), img_pred.clamp(0, 1), device)
    return {"psnr": psnr, "ssim": ssim_val, "ms_ssim": ms_val, "lpips": lp}
