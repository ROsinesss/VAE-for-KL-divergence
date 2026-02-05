"""
Loss functions for VAE training
This file contains all loss components that can be easily modified for experimentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(recon_x, x, loss_type='mse'):
    """
    Reconstruction loss between original and reconstructed images

    Args:
        recon_x: Reconstructed images from decoder
        x: Original input images
        loss_type: Type of reconstruction loss ('mse' or 'bce')

    Returns:
        Reconstruction loss value
    """
    if loss_type == 'mse':
        # Mean Squared Error - better for normalized images
        return F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        # Binary Cross Entropy - better for images in [0, 1]
        return F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_divergence_loss(mu, logvar):
    """
    KL divergence loss between learned latent distribution and standard normal

    KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of the latent distribution
        logvar: Log variance of the latent distribution

    Returns:
        KL divergence loss value
    """
    # KL divergence formula for Gaussian distributions
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, reconstruction_loss_type='mse'):
    """
    Complete VAE loss = Reconstruction Loss + KL_weight * KL Divergence

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term (beta in beta-VAE)
        reconstruction_loss_type: Type of reconstruction loss

    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
    """
    # Calculate individual loss components
    recon_loss = reconstruction_loss(recon_x, x, reconstruction_loss_type)
    kl_loss = kl_divergence_loss(mu, logvar)

    # Combine losses with KL weight
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def elbo_loss(recon_x, x, mu, logvar, reconstruction_loss_type='mse'):
    """
    Evidence Lower Bound (ELBO) - standard VAE objective
    This is equivalent to vae_loss with kl_weight=1.0

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        reconstruction_loss_type: Type of reconstruction loss

    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
    """
    return vae_loss(recon_x, x, mu, logvar, kl_weight=1.0,
                   reconstruction_loss_type=reconstruction_loss_type)


# You can add custom loss functions here for experimentation
# For example:

def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0, reconstruction_loss_type='mse'):
    """
    Beta-VAE loss with explicit beta parameter
    Higher beta encourages better disentanglement but may hurt reconstruction

    Args:
        beta: Weight for KL divergence (typically > 1.0 for disentanglement)
    """
    return vae_loss(recon_x, x, mu, logvar, kl_weight=beta,
                   reconstruction_loss_type=reconstruction_loss_type)


def annealed_vae_loss(recon_x, x, mu, logvar, current_epoch, total_epochs,
                     max_kl_weight=1.0, reconstruction_loss_type='mse'):
    """
    VAE loss with KL annealing - gradually increase KL weight during training
    This helps with posterior collapse in early training

    Args:
        current_epoch: Current training epoch
        total_epochs: Total number of epochs for annealing
        max_kl_weight: Maximum KL weight after annealing
    """
    # Linear annealing
    kl_weight = max_kl_weight * min(1.0, current_epoch / (total_epochs * 0.5))
    return vae_loss(recon_x, x, mu, logvar, kl_weight=kl_weight,
                   reconstruction_loss_type=reconstruction_loss_type)
