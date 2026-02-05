"""
VAE Model Architecture for Image Generation
Optimized for CIFAR-10 (32x32x3 images)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network that maps images to latent space parameters (mu, logvar)
    """
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None):
        super(Encoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build encoder layers
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate the size after convolutions
        # For CIFAR-10 (32x32): 32 -> 16 -> 8 -> 4 -> 2
        self.flatten_size = hidden_dims[-1] * 2 * 2

        # Latent space parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        """
        Encode input image to latent parameters

        Args:
            x: Input images [batch_size, 3, 32, 32]

        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that maps latent vectors back to images
    """
    def __init__(self, latent_dim=128, hidden_dims=None, out_channels=3):
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Reverse the hidden dimensions for decoder
        hidden_dims = list(reversed(hidden_dims))

        # Calculate the size after convolutions
        self.flatten_size = hidden_dims[0] * 2 * 2

        # Project latent vector to feature maps
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        self.decoder = nn.Sequential(*modules)

        # Final layer to output image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        self.hidden_dims = hidden_dims

    def forward(self, z):
        """
        Decode latent vector to image

        Args:
            z: Latent vectors [batch_size, latent_dim]

        Returns:
            recon_x: Reconstructed images [batch_size, 3, 32, 32]
        """
        x = self.fc(z)
        x = x.view(-1, self.hidden_dims[0], 2, 2)
        x = self.decoder(x)
        recon_x = self.final_layer(x)

        return recon_x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder model
    """
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Initialize encoder and decoder
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, hidden_dims, in_channels)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        where epsilon ~ N(0, 1)

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through VAE

        Args:
            x: Input images [batch_size, 3, 32, 32]

        Returns:
            recon_x: Reconstructed images [batch_size, 3, 32, 32]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def sample(self, num_samples, device):
        """
        Generate new samples from the learned distribution

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            samples: Generated images [num_samples, 3, 32, 32]
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Decode
        samples = self.decoder(z)

        return samples

    def reconstruct(self, x):
        """
        Reconstruct input images (without sampling)

        Args:
            x: Input images

        Returns:
            recon_x: Reconstructed images
        """
        mu, logvar = self.encoder(x)
        # Use mean for reconstruction (no sampling)
        recon_x = self.decoder(mu)
        return recon_x


def create_vae(config):
    """
    Factory function to create VAE model from config

    Args:
        config: Configuration object with model parameters

    Returns:
        model: VAE model
    """
    model = VAE(
        in_channels=config.num_channels,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims
    )
    return model
