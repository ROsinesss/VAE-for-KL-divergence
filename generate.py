"""
Generate samples from a trained VAE model
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid

from model import create_vae
from config import Config


def generate_and_save_samples(model, num_samples, save_path, config, grid_size=None):
    """
    Generate samples and save as grid

    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        save_path: Path to save the generated samples
        config: Configuration object
        grid_size: Tuple (nrow, ncol) for grid layout, if None will use square grid
    """
    model.eval()

    with torch.no_grad():
        # Generate samples
        samples = model.sample(num_samples, config.device)
        samples = samples.cpu()

        # Create grid
        if grid_size is None:
            nrow = int(np.sqrt(num_samples))
        else:
            nrow = grid_size[0]

        # Save as grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)

        # Save image
        save_image(grid, save_path)
        print(f'Saved generated samples to {save_path}')

        # Also display with matplotlib
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'Generated Samples (n={num_samples})')
        plt.tight_layout()

        # Save matplotlib version
        matplotlib_path = save_path.replace('.png', '_matplotlib.png')
        plt.savefig(matplotlib_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved matplotlib version to {matplotlib_path}')


def interpolate_latent_space(model, num_steps, save_path, config):
    """
    Interpolate between two random points in latent space

    Args:
        model: Trained VAE model
        num_steps: Number of interpolation steps
        save_path: Path to save the interpolation
        config: Configuration object
    """
    model.eval()

    with torch.no_grad():
        # Sample two random latent vectors
        z1 = torch.randn(1, config.latent_dim).to(config.device)
        z2 = torch.randn(1, config.latent_dim).to(config.device)

        # Create interpolation
        alphas = torch.linspace(0, 1, num_steps)
        interpolated_samples = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            sample = model.decoder(z_interp)
            interpolated_samples.append(sample)

        # Stack samples
        samples = torch.cat(interpolated_samples, dim=0).cpu()

        # Save as grid
        grid = make_grid(samples, nrow=num_steps, padding=2, normalize=False)
        save_image(grid, save_path)
        print(f'Saved latent space interpolation to {save_path}')


def generate_reconstructions(model, test_loader, num_images, save_path, config):
    """
    Show reconstructions of real images

    Args:
        model: Trained VAE model
        test_loader: Test data loader
        num_images: Number of images to reconstruct
        save_path: Path to save reconstructions
        config: Configuration object
    """
    model.eval()

    with torch.no_grad():
        # Get a batch of real images
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        images = images[:num_images].to(config.device)

        # Reconstruct
        recon_images, _, _ = model(images)

        # Combine original and reconstructed
        comparison = torch.cat([images.cpu(), recon_images.cpu()])

        # Save as grid (original on top, reconstructed on bottom)
        nrow = num_images
        grid = make_grid(comparison, nrow=nrow, padding=2, normalize=False)

        save_image(grid, save_path)
        print(f'Saved reconstructions to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_latest.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save generated samples')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'interpolate', 'reconstruct'],
                       help='Generation mode')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of interpolation steps (for interpolate mode)')
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = create_vae(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate based on mode
    if args.mode == 'sample':
        print(f'Generating {args.num_samples} samples...')
        save_path = os.path.join(args.output_dir, 'samples.png')
        generate_and_save_samples(model, args.num_samples, save_path, config)

    elif args.mode == 'interpolate':
        print(f'Generating latent space interpolation with {args.num_steps} steps...')
        save_path = os.path.join(args.output_dir, 'interpolation.png')
        interpolate_latent_space(model, args.num_steps, save_path, config)

    elif args.mode == 'reconstruct':
        print(f'Generating reconstructions for {args.num_samples} images...')
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        # Load test set
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.CIFAR10(
            root=config.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)

        save_path = os.path.join(args.output_dir, 'reconstructions.png')
        generate_reconstructions(model, test_loader, args.num_samples, save_path, config)

    print(f'\nDone! Output saved to {args.output_dir}')


if __name__ == '__main__':
    main()
