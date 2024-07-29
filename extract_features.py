import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os
from diffusers.models import AutoencoderKL
from datasets_prep.dataset import create_dataset

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda")

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(f'{args.features_path}/{args.dataset}/{args.image_size}_features', exist_ok=True)
    os.makedirs(f'{args.features_path}/{args.dataset}/{args.image_size}_labels', exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    dataset = create_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
        x = x.detach().cpu().numpy()    # (1, 4, 32, 32)
        np.save(f'{args.features_path}/{args.dataset}/{args.image_size}_features/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.features_path}/{args.dataset}/{args.image_size}_labels/{train_steps}.npy', y)
            
        train_steps += 1
        if train_steps % 100 == 0:
            print(train_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cifar10","celeba","lsun","coco","imagenet"], required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="features")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
