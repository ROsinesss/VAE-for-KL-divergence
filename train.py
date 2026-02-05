"""
Training script for VAE on CIFAR-10
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import create_vae
from loss import vae_loss
from config import Config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_cifar10_loaders(config):
    """
    Create CIFAR-10 data loaders

    Args:
        config: Configuration object

    Returns:
        train_loader, test_loader: DataLoader objects
    """
    # Data transformations - normalize to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization - keep in [0, 1] for sigmoid output
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, config, epoch):
    """
    Train for one epoch

    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        config: Configuration object
        epoch: Current epoch number

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(config.device)

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(
            recon_batch, data, mu, logvar,
            kl_weight=config.kl_weight,
            reconstruction_loss_type=config.reconstruction_loss_type
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()

        # Update progress bar
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                'loss': loss.item() / len(data),
                'recon': recon_loss.item() / len(data),
                'kl': kl_loss.item() / len(data)
            })

    # Calculate average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_kl_loss = train_kl_loss / len(train_loader.dataset)

    return avg_loss, avg_recon_loss, avg_kl_loss


def test_epoch(model, test_loader, config):
    """
    Evaluate on test set

    Args:
        model: VAE model
        test_loader: Test data loader
        config: Configuration object

    Returns:
        avg_loss: Average loss on test set
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.device)

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(
                recon_batch, data, mu, logvar,
                kl_weight=config.kl_weight,
                reconstruction_loss_type=config.reconstruction_loss_type
            )

            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

    # Calculate average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_kl_loss = test_kl_loss / len(test_loader.dataset)

    return avg_loss, avg_recon_loss, avg_kl_loss


def save_samples(model, epoch, config):
    """
    Generate and save sample images

    Args:
        model: VAE model
        epoch: Current epoch
        config: Configuration object
    """
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = model.sample(config.num_samples, config.device)
        samples = samples.cpu()

        # Create grid of images
        n = int(np.sqrt(config.num_samples))
        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                img = samples[idx].permute(1, 2, 0).numpy()
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

        plt.tight_layout()
        os.makedirs(f'{config.save_dir}/samples', exist_ok=True)
        plt.savefig(f'{config.save_dir}/samples/epoch_{epoch:03d}.png')
        plt.close()


def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, config):
    """
    Save model checkpoint

    Args:
        model: VAE model
        optimizer: Optimizer
        epoch: Current epoch
        train_losses: Training loss history
        test_losses: Test loss history
        config: Configuration object
    """
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'config': config
    }
    torch.save(checkpoint, f'{config.save_dir}/checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, f'{config.save_dir}/checkpoint_latest.pt')


def plot_losses(train_losses, test_losses, config):
    """
    Plot training and test losses

    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        config: Configuration object
    """
    plt.figure(figsize=(15, 5))

    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses['total'], label='Train')
    plt.plot(test_losses['total'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss')
    plt.grid(True)

    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses['recon'], label='Train')
    plt.plot(test_losses['recon'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.title('Reconstruction Loss')
    plt.grid(True)

    # KL loss
    plt.subplot(1, 3, 3)
    plt.plot(train_losses['kl'], label='Train')
    plt.plot(test_losses['kl'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.title('KL Divergence')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{config.save_dir}/loss_curves.png')
    plt.close()


def main():
    """Main training function"""
    # Load configuration
    config = Config()

    # Set random seed
    set_seed(config.seed)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f'Using device: {device}')

    # Create data loaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_loaders(config)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Create model
    print('Creating VAE model...')
    model = create_vae(config).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training history
    train_losses = {'total': [], 'recon': [], 'kl': []}
    test_losses = {'total': [], 'recon': [], 'kl': []}

    # Training loop
    print(f'\nStarting training for {config.num_epochs} epochs...')
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, config, epoch)
        train_losses['total'].append(train_loss)
        train_losses['recon'].append(train_recon)
        train_losses['kl'].append(train_kl)

        # Test
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, config)
        test_losses['total'].append(test_loss)
        test_losses['recon'].append(test_recon)
        test_losses['kl'].append(test_kl)

        # Print epoch summary
        print(f'\nEpoch {epoch}: Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        print(f'  Recon - Train: {train_recon:.4f}, Test: {test_recon:.4f}')
        print(f'  KL - Train: {train_kl:.4f}, Test: {test_kl:.4f}')

        # Generate samples
        if epoch % config.sample_interval == 0:
            print(f'Generating samples...')
            save_samples(model, epoch, config)

        # Save checkpoint
        if epoch % config.save_interval == 0:
            print(f'Saving checkpoint...')
            save_checkpoint(model, optimizer, epoch, train_losses, test_losses, config)

        # Plot losses
        plot_losses(train_losses, test_losses, config)

    # Final checkpoint
    print('Saving final checkpoint...')
    save_checkpoint(model, optimizer, config.num_epochs, train_losses, test_losses, config)

    print('\nTraining complete!')
    print(f'Checkpoints saved to: {config.save_dir}')


if __name__ == '__main__':
    main()
