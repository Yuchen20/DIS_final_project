import os
import sys
import random
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add models to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from pix2pix_improved import ImprovedPix2PixModel
from data_loader import DiffusionDataset
# python src/train_pix2pix_standalone.py --save_dir /path/to/save --experiment_name my_experiment

### https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/LICENSE

@dataclass
class Pix2PixTrainingConfig:
    """Configuration for Pix2Pix training"""
    # Model parameters
    input_nc: int = 5
    output_nc: int = 5
    ngf: int = 64
    ndf: int = 64
    norm: str = 'batch'
    use_dropout: bool = True
    lambda_L1: float = 100.0
    gan_mode: str = 'vanilla'
    
    # Training parameters
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    num_epochs: int = 10
    batch_size: int = 2
    num_workers: int = 4
    
    # Logging and saving
    save_dir: str = '/rds/user/ym429/hpc-work/dissertation/results/pix2pix_standalone'
    experiment_name: str = 'pix2pix_experiment'
    log_freq: int = 100  # Log images every N steps
    save_freq: int = 5   # Save model every N epochs
    eval_freq: int = 1   # Evaluate every N epochs
    
    # Data parameters
    base_csv_dir: str = "/home/ym429/rds/hpc-work/dissertation"
    img_size: tuple = (512, 512)
    source_channels: list = None
    target_channels: list = None
    train_plates: list = None
    valid_plates: list = None
    
    # Training settings
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cuda', or 'cpu'
    
    def __post_init__(self):
        # Set default values for data parameters
        if self.source_channels is None:
            self.source_channels = [7, 7, 7, 7, 7]
        if self.target_channels is None:
            self.target_channels = [1, 2, 3, 4, 5]
        if self.train_plates is None:
            self.train_plates = ["BR00116991", "BR00116992", "BR00116995", "BR00117024"]
        if self.valid_plates is None:
            self.valid_plates = ["BR00116993", "BR00117025"]


def set_seed(seed: int):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_collator(batch):
    """Form batches from the dataset"""
    sources = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return {'source': sources, 'target': targets}


def create_datasets(config: Pix2PixTrainingConfig):
    """Create training and validation datasets"""
    # Construct full CSV paths
    train_csv_list = [os.path.join(config.base_csv_dir, f"unique_paths_{p}.csv") 
                      for p in config.train_plates]
    valid_csv_list = [os.path.join(config.base_csv_dir, f"unique_paths_{p}.csv") 
                      for p in config.valid_plates]

    # Create datasets
    train_dataset = DiffusionDataset(
        csv_file_list=train_csv_list,
        source_channels=config.source_channels,
        target_channels=config.target_channels,
        img_size=config.img_size
    )
    
    val_dataset = DiffusionDataset(
        csv_file_list=valid_csv_list,
        source_channels=config.source_channels,
        target_channels=config.target_channels,
        img_size=config.img_size
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: Pix2PixTrainingConfig):
    """Create training and validation dataloaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=data_collator,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=data_collator,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return train_loader, val_loader


def log_images_to_wandb(model, step: int, prefix: str = 'train'):
    """Log images to wandb"""
    with torch.no_grad():
        visuals = model.get_current_visuals()
        
        if 'real_A' in visuals and 'fake_B' in visuals and 'real_B' in visuals:
            # Move to CPU and convert to numpy (first item in batch)
            real_A = visuals['real_A'][0].cpu().numpy()
            fake_B = visuals['fake_B'][0].cpu().numpy() 
            real_B = visuals['real_B'][0].cpu().numpy()
            
            # Normalize for display
            def normalize_for_display(img):
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return np.clip(img_norm, 0, 1)
            
            real_A = normalize_for_display(real_A)
            fake_B = normalize_for_display(fake_B)
            real_B = normalize_for_display(real_B)
            
            # Create visualization
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            
            for i in range(5):
                axes[0, i].imshow(real_A[i], cmap='viridis', vmin=0, vmax=1)
                axes[0, i].set_title(f'Input Ch {i+1}')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(fake_B[i], cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title(f'Generated Ch {i+1}')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(real_B[i], cmap='viridis', vmin=0, vmax=1)
                axes[2, i].set_title(f'Target Ch {i+1}')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f'{prefix}/images': wandb.Image(fig)}, step=step)
            plt.close(fig)


def save_checkpoint(model, optimizer_G, optimizer_D, epoch: int, step: int, 
                   config: Pix2PixTrainingConfig, is_best: bool = False):
    """Save model checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'netG_state_dict': model.netG.state_dict(),
        'netD_state_dict': model.netD.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(config.save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save latest checkpoint
    latest_path = os.path.join(config.save_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_path: str, model, optimizer_G, optimizer_D, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.netG.load_state_dict(checkpoint['netG_state_dict'])
    model.netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step']


def evaluate_model(model, val_loader, device, epoch: int):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss_G = 0.0
    total_loss_D = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_loader):
            # Move to device
            sources = inputs['source'].to(device)
            targets = inputs['target'].to(device)
            
            # Set input for model
            model.set_input({'source': sources, 'target': targets})
            
            # Forward pass
            fake_B = model.netG(model.real_A)
            
            # Compute losses
            loss_G, _, _ = model.compute_generator_loss(model.real_A, model.real_B, fake_B)
            loss_D, _, _ = model.compute_discriminator_loss(model.real_A, model.real_B, fake_B)
            
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            num_batches += 1
            
            # Log validation images for first batch
            if batch_idx == 0:
                log_images_to_wandb(model, epoch, prefix='val')
    
    avg_loss_G = total_loss_G / num_batches
    avg_loss_D = total_loss_D / num_batches
    
    model.train()
    return avg_loss_G, avg_loss_D


def train_pix2pix(config: Pix2PixTrainingConfig, resume_from: Optional[str] = None):
    """Main training function for Pix2Pix"""
    
    # Set up device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize wandb
    wandb.init(
        project='pix2pix-standalone',
        name=config.experiment_name,
        config=config.__dict__,
        dir=config.save_dir
    )
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    gpu_ids = [i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    model = ImprovedPix2PixModel(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf,
        ndf=config.ndf,
        norm=config.norm,
        use_dropout=config.use_dropout,
        lambda_L1=config.lambda_L1,
        gan_mode=config.gan_mode,
        gpu_ids=gpu_ids
    )
    
    print(f"Model device: {model.device}")
    
    # Create optimizers
    optimizer_G = torch.optim.Adam(
        model.netG.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    optimizer_D = torch.optim.Adam(
        model.netD.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        start_epoch, global_step = load_checkpoint(
            resume_from, model, optimizer_G, optimizer_D, device
        )
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch_idx, inputs in enumerate(progress_bar):
            # Move to device
            sources = inputs['source'].to(device)
            targets = inputs['target'].to(device)
            
            # Set input for model
            model.set_input({'source': sources, 'target': targets})
            
            # ========================
            # Update Discriminator
            # ========================
            optimizer_D.zero_grad()
            
            # Generate fake images (detached for discriminator)
            with torch.no_grad():
                fake_B = model.netG(model.real_A)
            
            # Compute discriminator loss
            loss_D, loss_D_real, loss_D_fake = model.compute_discriminator_loss(
                model.real_A, model.real_B, fake_B
            )
            
            # Backward pass for discriminator
            loss_D.backward()
            optimizer_D.step()
            
            # ========================
            # Update Generator
            # ========================
            optimizer_G.zero_grad()
            
            # Generate fake images (with gradients for generator)
            fake_B = model.netG(model.real_A)
            
            # Compute generator loss
            loss_G, loss_G_GAN, loss_G_L1 = model.compute_generator_loss(
                model.real_A, model.real_B, fake_B
            )
            
            # Backward pass for generator
            loss_G.backward()
            optimizer_G.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'G_L1': f'{loss_G_L1.item():.4f}'
            })
            
            # Log to wandb
            if global_step % config.log_freq == 0:
                wandb.log({
                    'train/loss_G': loss_G.item(),
                    'train/loss_G_GAN': loss_G_GAN.item(),
                    'train/loss_G_L1': loss_G_L1.item(),
                    'train/loss_D': loss_D.item(),
                    'train/loss_D_real': loss_D_real.item(),
                    'train/loss_D_fake': loss_D_fake.item(),
                    'epoch': epoch,
                }, step=global_step)
                
                # Log images
                log_images_to_wandb(model, global_step, prefix='train')
            
            global_step += 1
        
        # Evaluation
        if (epoch + 1) % config.eval_freq == 0:
            print(f"\nEvaluating at epoch {epoch + 1}...")
            val_loss_G, val_loss_D = evaluate_model(model, val_loader, device, epoch)
            
            wandb.log({
                'val/loss_G': val_loss_G,
                'val/loss_D': val_loss_D,
                'epoch': epoch,
            }, step=global_step)
            
            print(f"Validation - G_loss: {val_loss_G:.4f}, D_loss: {val_loss_D:.4f}")
            
            # Save best model
            is_best = val_loss_G < best_val_loss
            if is_best:
                best_val_loss = val_loss_G
                print(f"New best validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(
                model, optimizer_G, optimizer_D, epoch + 1, global_step,
                config, is_best=(epoch + 1) % config.eval_freq == 0 and val_loss_G < best_val_loss
            )
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
    
    # Save final checkpoint
    save_checkpoint(model, optimizer_G, optimizer_D, config.num_epochs, global_step, config)
    
    print("Training completed!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model')
    parser.add_argument('--save_dir', type=str, 
                       default='/rds/user/ym429/hpc-work/dissertation/results/pix2pix',
                       help='Directory to save checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default='pix2pix_experiment',
                       help='Name for the experiment')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--lambda_L1', type=float, default=100.0,
                       help='Weight for L1 loss')
    
    args = parser.parse_args()
    
    # Create config
    config = Pix2PixTrainingConfig(
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_L1=args.lambda_L1
    )
    
    # Start training
    train_pix2pix(config, resume_from=args.resume_from)


if __name__ == '__main__':
    main() 