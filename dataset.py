import os
from pathlib import Path
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class HRDataset(data.Dataset):
    def __init__(self, root, cfg, split="train"):
        self.root = root
        self.cfg = cfg
        self.split = split
        self.scale = cfg["scale"]

        if split == "train":
            self.input_dir = os.path.join(root, cfg["train_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["train_dir"], cfg["target_dir"])
        else:
            self.input_dir = os.path.join(root, cfg["test_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["test_dir"], cfg["target_dir"])

        self.extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        self.image_files = self._get_image_files()

        self.transform = self._get_transforms()

        print(f"HR Dataset[{split}] -> {len(self.image_files)} samples")
        print(
            f"  Input: {self.input_dir} ({len(self._list_files(self.input_dir))} files)"
        )
        print(
            f"  Target: {self.target_dir} ({len(self._list_files(self.target_dir))} files)"
        )

    def _list_files(self, folder):
        if not os.path.exists(folder):
            return []
        files = []
        for fp in Path(folder).rglob("*"):
            if fp.is_file() and fp.suffix.lower() in self.extensions:
                files.append(fp)
        return files

    def _get_image_files(self):
        image_files = []

        if os.path.exists(self.input_dir) and os.path.exists(self.target_dir):
            input_files = sorted(
                [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.lower().endswith(self.extensions)
                ]
            )
            for file in input_files:
                ext = file.split(".")[-1]

                # Map LR filename like 0001x{scale}.png -> HR filename 0001.png
                if "_LRBI_" in file:
                    s = "_LRBI_"
                else:
                    s = ""
                target_filename = file.replace(f"{s}x{self.scale}.{ext}", f".{ext}")
                target_path = os.path.join(self.target_dir, target_filename)
                if os.path.exists(target_path):
                    # store filenames only; construct full paths in __getitem__
                    image_files.append((file, target_filename))
                else:
                    print(f"Target file not found: {target_path}")

        return image_files

    def _get_transforms(self):
        lr_size = self.cfg.get("image_size", 256)
        hr_size = lr_size * self.scale

        # For SR, always produce LR-sized inputs and HR-sized targets
        transform_lr = transforms.Compose(
            [
                transforms.Resize((lr_size, lr_size)),
                transforms.ToTensor(),
            ]
        )
        transform_hr = transforms.Compose(
            [
                transforms.Resize((hr_size, hr_size)),
                transforms.ToTensor(),
            ]
        )

        return {"lr": transform_lr, "hr": transform_hr}

    def _apply_pair_augment(self, lr_img, hr_img):
        """Apply identical random augmentations to LR and HR images (PIL) during training.
        Augs: Random H/V flips, random 90-degree rotation.
        """
        if self.split != "train":
            return lr_img, hr_img

        # Horizontal flip
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)

        # Vertical flip
        if random.random() < 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)

        # 90-degree rotation (0/90/180/270) with 50% chance to rotate
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            angle = 90 * k
            lr_img = TF.rotate(lr_img, angle=angle)
            hr_img = TF.rotate(hr_img, angle=angle)

        return lr_img, hr_img

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename, target_filename = self.image_files[idx]

        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, target_filename)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Paired augmentations first (on PIL), then deterministic resize+ToTensor
        input_image, target_image = self._apply_pair_augment(input_image, target_image)

        input_tensor = self.transform["lr"](input_image)
        target_tensor = self.transform["hr"](target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
            "idx": idx,
        }

    def data_collator(self, batch):
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        filenames = [item["filename"] for item in batch]
        indices = [item["idx"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "filenames": filenames,
            "indices": indices,
        }


def get_training_set(root, cfg):
    return HRDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    return HRDataset(root, cfg, split="test")
