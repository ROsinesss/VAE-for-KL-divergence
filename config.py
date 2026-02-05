"""
Configuration file for VAE training hyperparameters
"""

class Config:
    # Model architecture
    latent_dim = 128  # Latent space dimension
    hidden_dims = [32, 64, 128, 256]  # Encoder/decoder hidden dimensions

    # Training parameters
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Loss parameters
    kl_weight = 0.00025  # Beta parameter for beta-VAE (start small to stabilize training)
    reconstruction_loss_type = 'mse'  # 'mse' or 'bce'

    # Dataset
    dataset_name = 'cifar10'  # CIFAR-10 dataset
    image_size = 32
    num_channels = 3
    data_dir = './data'

    # Training options
    num_workers = 4
    device = 'cuda'  # 'cuda' or 'cpu'
    seed = 42

    # Checkpoint and logging
    save_dir = './checkpoints'
    log_interval = 100  # Log every N batches
    save_interval = 10  # Save checkpoint every N epochs
    sample_interval = 5  # Generate samples every N epochs
    num_samples = 64  # Number of samples to generate

    # FID evaluation
    fid_batch_size = 50
    num_fid_samples = 10000  # Number of samples for FID calculation
