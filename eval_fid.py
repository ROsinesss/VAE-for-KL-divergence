"""
FID (Fréchet Inception Distance) evaluation script
Lower FID scores indicate better quality and diversity of generated images
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_fid import fid_score
from tqdm import tqdm
import shutil

from model import create_vae
from config import Config


def generate_samples(model, num_samples, save_dir, config):
    """
    Generate samples from trained VAE and save to directory

    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        save_dir: Directory to save generated images
        config: Configuration object
    """
    model.eval()

    # Create directory for generated samples
    os.makedirs(save_dir, exist_ok=True)

    # Generate samples in batches
    batch_size = config.fid_batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f'Generating {num_samples} samples...')
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            # Calculate batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Generate samples
            samples = model.sample(current_batch_size, config.device)
            samples = samples.cpu()

            # Save each sample as PNG
            from torchvision.utils import save_image
            for j in range(current_batch_size):
                img_idx = i * batch_size + j
                save_image(samples[j], f'{save_dir}/sample_{img_idx:05d}.png')


def prepare_real_images(save_dir, config, num_images=10000):
    """
    Prepare real CIFAR-10 images for FID calculation

    Args:
        save_dir: Directory to save real images
        config: Configuration object
        num_images: Number of real images to use
    """
    # Create directory for real images
    os.makedirs(save_dir, exist_ok=True)

    # Load CIFAR-10 test set
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f'Preparing {num_images} real images...')
    from torchvision.utils import save_image
    for i in tqdm(range(min(num_images, len(dataset)))):
        img, _ = dataset[i]
        save_image(img, f'{save_dir}/real_{i:05d}.png')


def calculate_fid(real_dir, fake_dir, device='cuda', batch_size=50):
    """
    Calculate FID score between real and generated images

    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing generated images
        device: Device to use for calculation
        batch_size: Batch size for FID calculation

    Returns:
        fid_value: FID score
    """
    print('Calculating FID score...')
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048  # Use Inception features
    )
    return fid_value


def evaluate_fid(checkpoint_path, config, num_samples=10000):
    """
    Complete FID evaluation pipeline

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        num_samples: Number of samples to generate for FID

    Returns:
        fid_value: FID score
    """
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device

    # Load model
    print(f'Loading model from {checkpoint_path}...')
    model = create_vae(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')

    # Create temporary directories
    real_dir = '/tmp/fid_real'
    fake_dir = '/tmp/fid_fake'

    try:
        # Prepare real images
        if not os.path.exists(real_dir) or len(os.listdir(real_dir)) < num_samples:
            print('Preparing real images...')
            prepare_real_images(real_dir, config, num_samples)
        else:
            print(f'Using existing real images from {real_dir}')

        # Generate fake images
        print('Generating fake images...')
        if os.path.exists(fake_dir):
            shutil.rmtree(fake_dir)
        generate_samples(model, num_samples, fake_dir, config)

        # Calculate FID
        fid_value = calculate_fid(real_dir, fake_dir, device=str(device), batch_size=config.fid_batch_size)

        print(f'\n{"="*50}')
        print(f'FID Score: {fid_value:.2f}')
        print(f'{"="*50}\n')

        # Clean up fake images directory
        shutil.rmtree(fake_dir)

        return fid_value

    except Exception as e:
        print(f'Error during FID evaluation: {e}')
        # Clean up in case of error
        if os.path.exists(fake_dir):
            shutil.rmtree(fake_dir)
        raise


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate VAE with FID score')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate for FID calculation')
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Evaluate FID
    fid_value = evaluate_fid(args.checkpoint, config, args.num_samples)

    # Interpretation guide
    print('\nFID Score Interpretation:')
    print('  < 10: Excellent quality (similar to real images)')
    print('  10-20: Very good quality')
    print('  20-50: Good quality')
    print('  50-100: Fair quality')
    print('  > 100: Poor quality')


if __name__ == '__main__':
    main()
