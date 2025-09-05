import argparse
import os
import glob
from typing import List
from PIL import Image

import cv2
import torch
import torchvision.transforms as transforms

from net import HCIT
import yaml
from utils import save_img


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_path: str) -> List[str]:
    if os.path.isfile(input_path) and is_image_file(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(paths)
    return []


def load_cfg_model(cfg_path: str, device: torch.device) -> HCIT:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    model = HCIT(**model_cfg).to(device)
    model.eval()
    return model


def clean_state_dict_keys(state_dict):
    # Remove possible prefixes like 'module.' or 'model.'
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module.") :]] = v
        elif k.startswith("model."):
            new_sd[k[len("model.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_weights(model, weight_path: str):
    ckpt = torch.load(weight_path, map_location="cpu")
    # Try common formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            # maybe the dict is already state_dict
            sd = ckpt
    else:
        # Unexpected format
        sd = ckpt
    sd = clean_state_dict_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("[Warning] Missing keys:", missing)
    if len(unexpected) > 0:
        print("[Warning] Unexpected keys:", unexpected)


def enhance_image(model, img, device: torch.device):
    with torch.no_grad():
        out = model(img)

    return out


def main():
    parser = argparse.ArgumentParser("UIE_model Inference")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument(
        "-w", "--weight", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./results_uie", help="Output folder"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/HCIT_light_x2.yaml",
        help="Path to config (model params)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument(
        "--suffix", type=str, default="_uie", help="Suffix for saved images"
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=8,
        help="Pad H,W to be divisible by this value",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Build model and load weights
    model = load_cfg_model(args.cfg_path, device)
    load_weights(model, args.weight)
    model.eval()

    # Use configured img_size for LR resize
    img_size = getattr(model, "img_size", 256)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    # Collect images
    paths = list_images(args.input)
    if len(paths) == 0:
        print(f"No images found in {args.input}")
        return

    for path in paths:
        img_name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        if img is None:
            print(f"[Warning] Failed to read image: {path}")
            continue

        img = transform(img).to(device)
        img = img.unsqueeze(0)
        res = enhance_image(model, img, device)

        name, ext = os.path.splitext(img_name)
        save_name = f"{name}{args.suffix}{ext}"
        save_path = os.path.join(args.output, save_name)
        save_img(res[0], save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
