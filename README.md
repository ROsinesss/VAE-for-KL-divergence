# VAE for Image Generation (CIFAR-10)

A Variational Autoencoder (VAE) implementation for image generation, optimized for small datasets like CIFAR-10. This project is designed to achieve single-digit FID scores with limited hardware resources.

## Features

- **Efficient VAE architecture** optimized for 32x32 images (CIFAR-10)
- **Modular loss functions** in separate file for easy experimentation
- **FID evaluation** to measure generation quality
- **Comprehensive training pipeline** with checkpointing and visualization
- **Low hardware requirements** - works on modest GPUs

## Project Structure

```
VAE-for-KL-divergence/
├── model.py           # VAE architecture (Encoder + Decoder)
├── loss.py            # Loss functions (reconstruction + KL divergence)
├── train.py           # Training script
├── eval_fid.py        # FID evaluation script
├── config.py          # Configuration and hyperparameters
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ROsinesss/VAE-for-KL-divergence.git
cd VAE-for-KL-divergence
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the VAE on CIFAR-10:
```bash
python train.py
```

The training script will:
- Automatically download CIFAR-10 dataset
- Train the VAE for 200 epochs
- Save checkpoints every 10 epochs
- Generate sample images every 5 epochs
- Plot loss curves

### Evaluation

Calculate FID score on trained model:
```bash
python eval_fid.py --checkpoint checkpoints/checkpoint_latest.pt
```

Or evaluate a specific checkpoint:
```bash
python eval_fid.py --checkpoint checkpoints/checkpoint_epoch_200.pt --num_samples 10000
```

## Configuration

Edit `config.py` to customize hyperparameters:

### Model Architecture
```python
latent_dim = 128              # Latent space dimension
hidden_dims = [32, 64, 128, 256]  # Encoder/decoder hidden dimensions
```

### Training Parameters
```python
batch_size = 128
num_epochs = 200
learning_rate = 1e-3
weight_decay = 1e-5
```

### Loss Parameters
```python
kl_weight = 0.00025          # KL divergence weight (beta parameter)
reconstruction_loss_type = 'mse'  # 'mse' or 'bce'
```

## Loss Functions

The `loss.py` file contains modular loss functions that you can easily modify:

### Available Loss Functions

1. **`vae_loss`** - Standard VAE loss with configurable KL weight
2. **`elbo_loss`** - Evidence Lower Bound (standard VAE objective)
3. **`beta_vae_loss`** - Beta-VAE for better disentanglement
4. **`annealed_vae_loss`** - KL annealing to prevent posterior collapse

### Customizing Losses

To experiment with different loss functions, simply modify the import in `train.py`:

```python
from loss import vae_loss  # Change this to your desired loss function
```

Or create your own loss function in `loss.py`:

```python
def my_custom_loss(recon_x, x, mu, logvar, **kwargs):
    # Your custom loss implementation
    recon_loss = reconstruction_loss(recon_x, x, 'mse')
    kl_loss = kl_divergence_loss(mu, logvar)

    # Custom combination
    total_loss = recon_loss + your_weight * kl_loss

    return total_loss, recon_loss, kl_loss
```

## Model Architecture

### Encoder
- Input: 32×32×3 images (CIFAR-10)
- 4 convolutional blocks with stride 2
- Output: Mean (μ) and log-variance (log σ²) of latent distribution

### Decoder
- Input: Latent vector (128-dimensional)
- 4 transposed convolutional blocks
- Output: Reconstructed 32×32×3 image

### Key Features
- Batch normalization for stable training
- LeakyReLU activation
- Reparameterization trick for backpropagation
- Sigmoid output for [0, 1] image range

## Training Tips

### Achieving Single-Digit FID

1. **Start with low KL weight**: Begin with `kl_weight = 0.00025` to avoid posterior collapse
2. **Train longer**: 200+ epochs recommended for best results
3. **Monitor losses**: Check that reconstruction loss decreases steadily
4. **Adjust β gradually**: Increase `kl_weight` if needed for better latent space structure

### Common Issues

**Problem**: Blurry reconstructions
- **Solution**: Decrease `kl_weight` or increase training time

**Problem**: Posterior collapse (KL → 0)
- **Solution**: Use lower `kl_weight` or try `annealed_vae_loss`

**Problem**: High FID score
- **Solution**: Train longer, adjust architecture (more `hidden_dims`), or tune learning rate

## Output Files

Training generates the following outputs:

```
checkpoints/
├── checkpoint_latest.pt       # Latest checkpoint
├── checkpoint_epoch_010.pt    # Periodic checkpoints
├── checkpoint_epoch_020.pt
├── ...
├── loss_curves.png           # Training curves
└── samples/
    ├── epoch_005.png         # Generated samples per epoch
    ├── epoch_010.png
    └── ...
```

## FID Score Interpretation

- **< 10**: Excellent quality (similar to real images) ✓ **Target**
- **10-20**: Very good quality
- **20-50**: Good quality
- **50-100**: Fair quality
- **> 100**: Poor quality

## Hardware Requirements

**Minimum**:
- GPU: 4GB VRAM (e.g., GTX 1050 Ti)
- RAM: 8GB
- Storage: 2GB for dataset + checkpoints

**Recommended**:
- GPU: 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 5GB

## Advanced Usage

### Custom Dataset

To use a different dataset, modify `train.py`:

```python
def get_custom_loaders(config):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dataset = YourDataset(transform=transform)
    # ... rest of the code
```

### Learning Rate Scheduling

Add a scheduler in `train.py`:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

# In training loop after each epoch:
scheduler.step()
```

### Monitoring with TensorBoard

Add TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/vae_experiment')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/test', test_loss, epoch)
```

## Citation

If you use this code for your research, please cite:

```
@misc{vae-cifar10,
  author = {ROsinesss},
  title = {VAE for Image Generation},
  year = {2026},
  url = {https://github.com/ROsinesss/VAE-for-KL-divergence}
}
```

## References

- [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500)

## License

MIT License - feel free to use this code for your projects!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
