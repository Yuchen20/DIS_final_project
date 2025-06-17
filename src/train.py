import os, sys
import random
import numpy as np
import copy
import torch
import wandb
import argparse
from dataclasses import dataclass, asdict
from transformers import Trainer, TrainingArguments
from data_loader import DiffusionDataset
# from models.unet import UNetModelSwin
from noise_scheduling import CFG, DiffusionScheduler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path
from plain_unet import UNet
from unet import UNetModelSwin
from pix2pix import Pix2PixModel
from pix2pix_improved import ImprovedPix2PixModel
import lpips
from safetensors.torch import load_file as load_safetensors
from collections import OrderedDict
from ldm.models.autoencoder import VQModelTorch
import torchvision.transforms as transforms

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# 1. Configuration
@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 5e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    weight_decay: float = 0
    lr_scheduler_type: str = 'cosine'
    warmup_steps: int = 5000
    logging_steps: int = 10
    evaluation_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    load_best_model_at_end: bool = True
    fp16: bool = False
    report_to: str = 'wandb'
    output_dir: str = 'results/pix2pix'
    # diffusion params
    kappa: float = 2.0
    p: float = 0.3
    eta_T: float = 0.99
    T: int = 15
    use_loss_coef: bool = False
    use_lpips: bool = False
    # training mode
    use_diffusion: bool = False
    use_pix2pix: bool = False
    use_consistency_distillation: bool = True
    use_latent_diffusion: bool = False  # New flag for latent diffusion
    # consistency distillation params
    cd_ema_decay: float = 0.9999  # μ in the paper
    cd_pretrained_path: str = "/rds/user/ym429/hpc-work/dissertation/results/rescell-15step-no_loss_coef-LPIPS/checkpoint-69120"  # Path to pretrained diffusion model
    cd_lambda_weight: float = 0.2  # Weight function λ(t_n)
    cd_target_reg_weight: float = 1.0  # Weight for target regularization term



# 2. Set seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 3. Data collator to form batches
def data_collator(batch):
    sources = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return {'source': sources, 'target': targets}

def load_hf_checkpoint(model_path, device='cpu'):
    """Helper function to load Hugging Face Trainer checkpoint"""
    if os.path.isdir(model_path):
        # For Hugging Face Trainer saved models
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        bin_path = os.path.join(model_path, 'pytorch_model.bin')

        if os.path.exists(safetensors_path):
            state_dict = load_safetensors(safetensors_path, device=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")  # remove the prefix from torch.compile
                new_state_dict[new_key] = v
            return new_state_dict
        elif os.path.exists(bin_path):
            checkpoint = torch.load(bin_path, map_location=device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            else:
                return checkpoint
        else:
            raise ValueError(f"Could not find model weights at {safetensors_path} or {bin_path}")
    else:
        # For regular checkpoint files
        checkpoint = torch.load(model_path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            return checkpoint

# 4. Custom Trainer overriding loss computation
class DiffusionTrainer(Trainer):
    def __init__(self, scheduler: DiffusionScheduler, use_loss_coef: bool = False, use_lpips: bool = True, *args, **kwargs):
        self.scheduler = scheduler
        self.use_loss_coef = use_loss_coef
        self.use_lpips = use_lpips
        self._eval_batch_counter = 0
        
        # Initialize LPIPS loss only if use_lpips is True
        if self.use_lpips:
            self.lpips_loss = lpips.LPIPS(net='vgg').to('cuda:0')
            for params in self.lpips_loss.parameters():
                params.requires_grad_(False)
            self.lpips_loss.eval()
        else:
            self.lpips_loss = None
            
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device  # Get the device from input tensors
        batch_size = sources.size(0)
        
        # sample random t for each example
        ts = torch.randint(1, self.scheduler.config.T + 1, (batch_size,), device=device)
        
        # generate noisy inputs and coefs
        noisy = []
        coefs = []
        for i, t in enumerate(ts):
            noisy_img = self.scheduler.get_noisy_image(int(t.item()), targets[i], sources[i])
            noisy.append(noisy_img)
            coef = self.scheduler.get_loss_coef(int(t.item()))
            coefs.append(coef.to(device))  # Ensure coef is on the same device
        
        noisy = torch.stack(noisy)
        coefs = torch.stack(coefs).view(batch_size, 1, 1, 1)

        # forward pass
        outputs = model(noisy, ts, lq = sources.to(device))

        # Compute per-sample losses
        per_sample_losses = []
        
        for b in range(batch_size):
            # MSE loss per sample
            sample_mse = (outputs[b] - targets[b]).pow(2).mean()
            
            # Start with MSE loss
            sample_loss = sample_mse
            
            # Add LPIPS loss only if use_lpips is True
            if self.use_lpips and self.lpips_loss is not None:
                sample_lpips_sum = 0.0
                for c in range(outputs.shape[1]):
                    # LPIPS expects 3-channel or 1-channel, so repeat to 3 channels
                    out_img = outputs[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)  # (1,3,H,W)
                    tgt_img = targets[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                    lpips_val = self.lpips_loss(out_img.to(device), tgt_img.to(device))
                    sample_lpips_sum += lpips_val
                sample_lpips = sample_lpips_sum / outputs.shape[1]
                
                # Add LPIPS to the loss
                sample_loss = sample_loss + sample_lpips
            
            # Apply loss coefficient if enabled
            if self.use_loss_coef:
                sample_loss = sample_loss * coefs[b].squeeze()
            
            per_sample_losses.append(sample_loss)
        
        # Average across batch
        loss = torch.stack(per_sample_losses).mean()
            
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at step {self.state.global_step}")
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
    
        # log images every 100 steps
        step = self.state.global_step
        if step and step % 200 == 0:
            # Get first item in batch
            noisy_img = noisy[0].detach().cpu().numpy()  # [5, 512, 512]
            denoised_img = outputs[0].detach().cpu().numpy()  # [5, 512, 512]
            target_img = targets[0].detach().cpu().numpy()  # [5, 512, 512]
            
            # Create figure with 3 rows (noisy, denoised, target) and 5 columns (channels)
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            
            # Plot noisy channels
            for i in range(5):
                axes[0, i].imshow(noisy_img[i], cmap='viridis')
                axes[0, i].set_title(f'Noisy Channel {i+1}')
                axes[0, i].axis('off')
            
            # Plot denoised channels
            for i in range(5):
                axes[1, i].imshow(denoised_img[i], cmap='viridis')
                axes[1, i].set_title(f'Denoised Channel {i+1}')
                axes[1, i].axis('off')
            
            # Plot target channels
            for i in range(5):
                axes[2, i].imshow(target_img[i], cmap='viridis')
                axes[2, i].set_title(f'Target Channel {i+1}')
                axes[2, i].axis('off')

            # set title as timestemp
            fig.suptitle(f'Timestep: {ts[0]}', fontsize=16)
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                'channel_comparison': wandb.Image(fig),
                'loss': loss.item()
            }, step=step)
            
            plt.close(fig)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override prediction step to handle dictionary inputs and diffusion process"""
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device  # Get the device from input tensors
        batch_size = sources.size(0)
        
        with torch.no_grad():
            # For validation, we'll use a fixed timestep (e.g., T/2) for all samples
            t = torch.full((batch_size,), self.scheduler.config.T // 2, device=device)
            
            # Generate noisy images
            noisy = []
            for i in range(batch_size):
                noisy_img = self.scheduler.get_noisy_image(int(t[i].item()), targets[i], sources[i])
                noisy.append(noisy_img)
            noisy = torch.stack(noisy)
            
            # Get model predictions
            outputs = model(noisy, t, lq = sources.to(device))
            
            # Compute loss
            loss = ((outputs - targets).pow(2)).mean()

            # for every 100 validation steps do full diffusion
            self._eval_batch_counter += 1
            if self._eval_batch_counter % 100 == 1:
                intermediate_results = []
                with torch.no_grad():
                    # only take the first image in the batch
                    sources = sources[0].to(device).unsqueeze(0)
                    targets = targets[0].to(device).unsqueeze(0)

                    x = self.scheduler.get_noisy_image(self.scheduler.config.T, sources, sources)
                    x = x.to(device)
                    batch_size = x.shape[0]

                    for t in range(self.scheduler.config.T, 0, -1):
                        # Create a batch of timesteps
                        timesteps = torch.full((batch_size,), t, device=device)
                        output = self.model(x, timesteps, lq = sources)
                        coef_1 = self.scheduler.get_eta_t(t - 1) / self.scheduler.get_eta_t(t)
                        coef_2 = self.scheduler.get_alpha_t(t) / self.scheduler.get_eta_t(t)
                        coef_3 = self.scheduler.config.kappa * self.scheduler.get_eta_t(t - 1) / self.scheduler.get_eta_t(t) * self.scheduler.get_alpha_t(t)

                        coef_1, coef_2, coef_3 = coef_1.to(device), coef_2.to(device), coef_3.to(device)

                        x = coef_1 * x + coef_2 * output + coef_3 * torch.randn_like(x, device=device)

                        intermediate_results.append(
                            (x.detach().cpu().numpy(), output.detach().cpu().numpy())
                        )
                    n_steps = len(intermediate_results)
                    fig, axes = plt.subplots(10, n_steps + 1, figsize=(2.2*n_steps, 20))
                    
                    for j in range(10):
                        for i, (x, output) in enumerate(intermediate_results):
                            x_img = x[0]  # Shape: (5, 512, 512)
                            output_img = output[0]  # Shape: (5, 512, 512)
                            target_img = targets[0].detach().cpu().numpy() # Shape: (5, 512, 512)
                            
                            if j < 5:
                                axes[j, i].imshow(x_img[j], cmap='viridis')
                            else:
                                axes[j, i].imshow(output_img[j-5], cmap='viridis')
                            axes[j, i].axis('off')

                    for j in range(10):
                        # the target image  
                        axes[j, n_steps].imshow(target_img[j % 5], cmap='viridis')
                        axes[j, n_steps].axis('off')
                    
                    plt.tight_layout()
                    
                    # Log diffusion process to wandb with custom step
                    val_step = self._eval_batch_counter
                    wandb.log({
                        'val/diffusion_process': wandb.Image(fig),
                        'val_step': val_step,
                    })


            
        return (loss, None, None)  # Only return loss for validation

class NormalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sources = inputs['source']
        targets = inputs['target']
        
        # Direct prediction without diffusion
        outputs = model(sources)
        
        # MSE loss
        loss = (outputs - targets).pow(2).mean()
        
        # Log images every 100 steps
        step = self.state.global_step
        if step and step % 1000 == 0:
            # Get first item in batch
            source_img = sources[0].detach().cpu().numpy()  # [5, 512, 512]
            pred_img = outputs[0].detach().cpu().numpy()    # [5, 512, 512]
            target_img = targets[0].detach().cpu().numpy()  # [5, 512, 512]
            
            # Create figure with 2 rows (gt and pred) and 5 columns (channels)
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            
            # Plot ground truth channels
            for i in range(5):
                axes[0, i].imshow(target_img[i], cmap='viridis')
                axes[0, i].set_title(f'GT Channel {i+1}')
                axes[0, i].axis('off')
            
            # Plot predicted channels
            for i in range(5):
                axes[1, i].imshow(pred_img[i], cmap='viridis')
                axes[1, i].set_title(f'Pred Channel {i+1}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                'channel_comparison': wandb.Image(fig),
                'loss': loss.item()
            }, step=step)
            
            plt.close(fig)
        
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_preds):
        """Compute MSE metric for evaluation"""
        predictions, labels = eval_preds
        mse = ((predictions - labels) ** 2).mean()
        return {"eval_mse": mse}

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override prediction step to handle dictionary inputs"""
        sources = inputs['source']
        targets = inputs['target']
        
        with torch.no_grad():
            outputs = model(sources)
            loss = (outputs - targets).pow(2).mean()
            
        return (loss, None, None)  # Only return loss, no need to store outputs and targets

class Pix2PixTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_batch_counter = 0
        self._step_counter = 0
        
    def create_optimizer(self):
        """Create optimizers for both generator and discriminator"""
        # Create optimizer for generator only - we'll handle discriminator separately
        lr = min(self.args.learning_rate, 2e-4)
        
        self.optimizer = torch.optim.Adam(
            self.model.netG.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        
        # Create discriminator optimizer separately
        self.optimizer_D = torch.optim.Adam(
            self.model.netD.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Clean training step following the pix2pix pattern"""
        model.train()
        
        # Set input for the model
        model.set_input(inputs)
        
        # ========================
        # Update Discriminator
        # ========================
        self.optimizer_D.zero_grad()
        
        # Forward pass through generator (detached for discriminator)
        with torch.no_grad():
            fake_B = model.netG(model.real_A)
        
        # Compute discriminator loss
        loss_D, loss_D_real, loss_D_fake = model.compute_discriminator_loss(
            model.real_A, model.real_B, fake_B
        )
        
        # Backward pass for discriminator
        loss_D.backward()
        self.optimizer_D.step()
        
        # ========================
        # Update Generator
        # ========================
        # Forward pass for generator (with gradients)
        fake_B = model.netG(model.real_A)
        
        # Compute generator loss
        loss_G, loss_G_GAN, loss_G_L1 = model.compute_generator_loss(
            model.real_A, model.real_B, fake_B
        )
        
        # Log images occasionally
        self._step_counter += 1
        if self._step_counter % 100 == 0:
            self._log_training_images(model, loss_G, loss_G_GAN, loss_G_L1, loss_D, loss_D_real, loss_D_fake)
        
        # Return generator loss - this will be handled by Trainer for backward pass
        return loss_G
    
    def _log_training_images(self, model, loss_G, loss_G_GAN, loss_G_L1, loss_D, loss_D_real, loss_D_fake):
        """Log training images to wandb"""
        with torch.no_grad():
            visuals = model.get_current_visuals()
            
            if 'real_A' in visuals and 'fake_B' in visuals and 'real_B' in visuals:
                # Move to CPU and convert to numpy
                real_A = visuals['real_A'][0].cpu().numpy()  # First item in batch
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
                
                # Plot channels
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
                losses = model.get_current_losses()
                wandb.log({
                    'pix2pix_training': wandb.Image(fig),
                    'loss_G': float(loss_G.item()),
                    'loss_G_GAN': float(loss_G_GAN.item()), 
                    'loss_G_L1': float(loss_G_L1.item()),
                    'loss_D': float(loss_D.item()),
                    'loss_D_real': float(loss_D_real.item()),
                    'loss_D_fake': float(loss_D_fake.item())
                }, step=self._step_counter)
                
                plt.close(fig)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for evaluation"""
        model.set_input(inputs)
        
        with torch.no_grad():
            fake_B = model.netG(model.real_A)
            loss_G, _, _ = model.compute_generator_loss(model.real_A, model.real_B, fake_B)
        
        return (loss_G, fake_B) if return_outputs else loss_G

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override prediction step for evaluation"""
        model.set_input(inputs)
        self._eval_batch_counter += 1
        
        with torch.no_grad():
            fake_B = model.netG(model.real_A)
            loss_G, loss_G_GAN, loss_G_L1 = model.compute_generator_loss(
                model.real_A, model.real_B, fake_B
            )
            
            # Log validation images occasionally
            if self._eval_batch_counter % 50 == 0:
                self._log_validation_images(model, loss_G, loss_G_GAN, loss_G_L1)
        
        return (loss_G, None, None)
    
    def _log_validation_images(self, model, loss_G, loss_G_GAN, loss_G_L1):
        """Log validation images to wandb"""
        with torch.no_grad():
            visuals = model.get_current_visuals()
            
            if 'real_A' in visuals and 'fake_B' in visuals and 'real_B' in visuals:
                # Move to CPU and convert to numpy
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
                
                wandb.log({
                    'pix2pix_validation': wandb.Image(fig),
                    'val/loss_G': float(loss_G.item()),
                    'val/loss_G_GAN': float(loss_G_GAN.item()),
                    'val/loss_G_L1': float(loss_G_L1.item())
                }, step=self._eval_batch_counter)
                
                plt.close(fig)

class ConsistencyDistillationTrainer(Trainer):
    """Trainer for Consistency Distillation following Algorithm 2 from the paper"""
    
    def __init__(
            self, 
            scheduler: DiffusionScheduler, 
            ema_decay: float = 0.9999, 
            lambda_weight: float = 1.0, 
            target_reg_weight: float = 0.1, 
            *args, **kwargs
        ):
        self.scheduler = scheduler
        self.ema_decay = ema_decay
        self.lambda_weight = lambda_weight
        self.target_reg_weight = target_reg_weight
        self._eval_batch_counter = 0
        
        # Initialize LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='vgg').to('cuda:0')
        for params in self.lpips_loss.parameters():
            params.requires_grad_(False)
        self.lpips_loss.eval()
        
        super().__init__(*args, **kwargs)
        
        # Initialize EMA model (θ^- in the paper)

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)


    def _update_ema_model(self, model):
        """Update EMA model parameters: θ^- ← stopgrad(μθ^- + (1-μ)θ)"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _ode_solver_step(self, x_tn_plus_1, f_theta_tn_plus_1, t_n):
        """
        Compute x^φ_{t_n} using the proper diffusion step (same as inference)
        This uses the same step as in SwinUNetPredictor.predict():
        x = coef_1 * x + coef_2 * output
        """
        t_n_plus_1 = t_n + 1
        
        # Calculate coefficients for the diffusion step (same as inference.py)
        # Going from t_{n+1} to t_n (one step back in diffusion process)
        coef_1 = self.scheduler.get_eta_t(t_n) / self.scheduler.get_eta_t(t_n_plus_1)
        coef_2 = self.scheduler.get_alpha_t(t_n_plus_1) / self.scheduler.get_eta_t(t_n_plus_1)
        
        # Ensure coefficients are on the right device
        coef_1, coef_2 = coef_1.to(x_tn_plus_1.device), coef_2.to(x_tn_plus_1.device)
        
        # Apply the diffusion step: x^φ_{t_n} = coef_1 * x_{t_{n+1}} + coef_2 * output
        # Note: we don't add noise (coef_3 term) during training as in the paper
        x_phi_tn = coef_1 * x_tn_plus_1 + coef_2 * f_theta_tn_plus_1
        
        return x_phi_tn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute consistency distillation loss according to Algorithm 2"""
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device
        batch_size = sources.size(0)
        
        # Sample n ~ U[1, N-1]
        t_n = torch.randint(1, self.scheduler.config.T, (batch_size,), device=device)
        t_n_1 = t_n + 1

        x_t_n_1 = []
        for i in range(batch_size):
            # Convert tensor to integer for indexing
            t_n_1_i = int(t_n_1[i].item())
            x_t_n_1.append(self.scheduler.get_noisy_image(t_n_1_i, targets[i], sources[i]))

        x_t_n_1 = torch.stack(x_t_n_1, dim=0)

        ## Online Prediction
        online_predictions = model(x_t_n_1, t_n_1, lq=sources)

        ## Offline Prediction
        ### Solver to get x tn with teacher
        teacher_predictions = self.teacher_model(x_t_n_1, t_n_1, lq=sources)
        x_phi_tn = []
        for i in range(batch_size):
            # Convert tensor to integer for indexing
            t_n_i = int(t_n[i].item())
            x_phi_tn.append(self._ode_solver_step(x_t_n_1[i], teacher_predictions[i], t_n_i))

        x_phi_tn = torch.stack(x_phi_tn, dim=0)

        ### Predict f_theta_tn_plus_1 with the EMA model
        offline_predictions = self.ema_model(x_phi_tn, t_n, lq=sources)

        # compute the loss
        MSE_loss = (online_predictions - offline_predictions).pow(2).mean()
        
        # Compute LPIPS loss for each channel separately
        LPIPS_loss = 0.0
        for c in range(online_predictions.shape[1]):
            # Convert single channel to 3 channels for LPIPS
            online_channel = online_predictions[:, c:c+1].repeat(1, 3, 1, 1)
            offline_channel = offline_predictions[:, c:c+1].repeat(1, 3, 1, 1)
            # Take mean to ensure scalar output
            channel_lpips = self.lpips_loss(online_channel, offline_channel).mean()
            LPIPS_loss += channel_lpips
        LPIPS_loss = LPIPS_loss / online_predictions.shape[1]  # Average over channels

        # log the mse and lpips loss
        wandb.log({
            'train/MSE_loss': MSE_loss.item(),
            'train/LPIPS_loss': LPIPS_loss.item(),
        }, step=self.state.global_step)

        loss = MSE_loss + LPIPS_loss

        if self.state.global_step % 100 == 0:
            self.log_images(sources, targets, x_t_n_1, teacher_predictions, x_phi_tn, online_predictions, offline_predictions, t_n, t_n_1)

        return loss

    def log_images(self, source, target, noised_image, teacher_prediction, x_phi_tn, online_prediction, offline_prediction, t_n, t_n_1):
        # plot the source, target, noised_image, teacher_prediction, x_phi_tn, online_prediction, offline_prediction
        # Only take the first sample from the batch for visualization
        fig, axes = plt.subplots(5, 7, figsize=(14, 10))

        images_list = [
            ('source', source[0]),  # Take first sample in batch
            ('x t_n+1', noised_image[0]),
            ('x t_n Recovered by teacher', x_phi_tn[0]),
            ('teacher_prediction', teacher_prediction[0]),
            ('online_prediction', online_prediction[0]),
            ('offline_prediction', offline_prediction[0]),
            ('target', target[0]),
        ]

        for i in range(5):  # 5 channels
            for idx, (image_name, image) in enumerate(images_list):  # 7 image types
                # Now image has shape (5, 512, 512), so image[i] gives us (512, 512)
                axes[i, idx].imshow(image[i].detach().cpu().numpy(), cmap='viridis')
                axes[i, idx].set_title(f'{image_name} Ch {i+1}')
                axes[i, idx].axis('off')

        fig.suptitle(f'Consistency Distillation Training (t_n={t_n[0]}, t_n+1={t_n_1[0]})', fontsize=16)
        plt.tight_layout()
        wandb.log({
            'cd_training_comparison': wandb.Image(fig),
        }, step=self.state.global_step)
        plt.close(fig)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to update EMA model after each step"""
        self._update_ema_model(model)

        loss = super().training_step(model, inputs, num_items_in_batch)
        
        return loss
    
    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override prediction step for evaluation"""
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device
        
        with torch.no_grad():
            # For evaluation, use the model directly without noise
            outputs = model(sources, torch.zeros(sources.size(0), device=device), lq=sources)
            loss = (outputs - targets).pow(2).mean()

            # log the mse, ssim, psnr
            # Calculate metrics for the batch
            try:
                mse = (outputs - targets).pow(2).mean()
                
                # Convert tensors to numpy for SSIM and PSNR calculation
                outputs_np = outputs.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()
                
                # Initialize metric lists for each channel
                channel_metrics = {
                    'ssim': [[] for _ in range(outputs_np.shape[1])],  # 5 channels
                    'psnr': [[] for _ in range(outputs_np.shape[1])]
                }
                
                # Calculate metrics per sample and per channel
                for batch_idx in range(outputs_np.shape[0]):  # Iterate over batch
                    for channel_idx in range(outputs_np.shape[1]):  # Iterate over channels
                        pred_channel = outputs_np[batch_idx, channel_idx]
                        target_channel = targets_np[batch_idx, channel_idx]
                        
                        # Normalize to 0-1 range
                        pred_norm = (pred_channel - pred_channel.min()) / (pred_channel.max() - pred_channel.min())
                        target_norm = (target_channel - target_channel.min()) / (target_channel.max() - target_channel.min())
                        
                        # Calculate SSIM
                        ssim_val = ssim(pred_norm, target_norm, data_range=1.0)
                        channel_metrics['ssim'][channel_idx].append(ssim_val)
                        
                        # Calculate PSNR
                        psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
                        channel_metrics['psnr'][channel_idx].append(psnr_val)
                
                # Calculate average metrics per channel across batch
                avg_ssim_per_channel = [np.mean(channel_metrics['ssim'][i]) for i in range(outputs_np.shape[1])]
                avg_psnr_per_channel = [np.mean(channel_metrics['psnr'][i]) for i in range(outputs_np.shape[1])]
                
                # Overall averages across all channels
                overall_avg_ssim = np.mean(avg_ssim_per_channel)
                overall_avg_psnr = np.mean(avg_psnr_per_channel)
                
                # Log metrics
                log_dict = {
                    'val/mse': mse.item(),
                    'val/ssim_overall': overall_avg_ssim,
                    'val/psnr_overall': overall_avg_psnr
                }
                
                # Log per-channel metrics
                for i in range(len(avg_ssim_per_channel)):
                    log_dict[f'val/ssim_ch{i+1}'] = avg_ssim_per_channel[i]
                    log_dict[f'val/psnr_ch{i+1}'] = avg_psnr_per_channel[i]
                
                wandb.log(log_dict, step=self._eval_batch_counter)
                
            except Exception as e:
                # do nothing
                pass
            
            # Occasionally visualize full denoising process
            self._eval_batch_counter += 1
            if self._eval_batch_counter % 100 == 1:
                # Take first image
                source = sources[0].unsqueeze(0)
                target = targets[0].unsqueeze(0)
                
                # Start from pure noise
                one_step_scheduler = DiffusionScheduler(CFG(2.0, 0.3, 0.99, 2))

                t = torch.tensor([1], device=device)

                # get the noised image
                x_t_n_1 = one_step_scheduler.get_noisy_image(t, source, source)
                
                # Single-step denoising (consistency model property)
                denoised = model(x_t_n_1, t, lq=source)
                
                fig, axes = plt.subplots(3, 5, figsize=(10, 6))
                
                # Plot source
                for j in range(5):
                    axes[0, j].imshow(source[0, j].cpu().numpy(), cmap='viridis')
                    axes[0, j].set_title(f'Source Ch {j+1}')
                    axes[0, j].axis('off')
                
                # Plot denoised
                for j in range(5):
                    axes[1, j].imshow(denoised[0, j].cpu().numpy(), cmap='viridis')
                    axes[1, j].set_title(f'Denoised Ch {j+1}')
                    axes[1, j].axis('off')

                for j in range(5):
                    axes[2, j].imshow(x_t_n_1[0, j].cpu().numpy(), cmap='viridis')
                    axes[2, j].set_title(f'Noised Ch {j+1}')
                    axes[2, j].axis('off')

                
                plt.tight_layout()
                
                wandb.log({
                    'val/cd_single_step_denoising': wandb.Image(fig),
                    'val_step': self._eval_batch_counter,
                })
                
                plt.close(fig)
        
        return (loss, None, None)

class LatentDiffusionTrainer(Trainer):
    """Trainer for Latent Diffusion using AutoencoderDC"""
    
    def __init__(self, scheduler: DiffusionScheduler, use_loss_coef: bool = False, use_lpips: bool = True, *args, **kwargs):
        self.scheduler = scheduler
        self.use_loss_coef = use_loss_coef
        self.use_lpips = use_lpips
        self._eval_batch_counter = 0
        
        # Initialize VQModelTorch and freeze it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # VQ autoencoder configuration from the config file
        ddconfig = {
            "double_z": False,
            "z_channels": 3,
            "resolution": 512,
            "in_channels": 5,
            "out_ch": 5,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "padding_mode": "zeros"
        }
        
        self.autoencoder = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            remap=None,
            sane_index_shape=False
        ).to(device).eval()
        
        # Load checkpoint
        ckpt_path = "/rds/user/ym429/hpc-work/dissertation/Model/model_400000.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # Remove any 'loss' related keys if they exist
        keys_to_remove = [key for key in state_dict.keys() if 'loss' in key]
        for key in keys_to_remove:
            del state_dict[key]
        
        self.autoencoder.load_state_dict(state_dict)
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)
        
        # Transform for autoencoder (expects normalized input)
        self.transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5),
        ])
        
        # Initialize LPIPS loss only if use_lpips is True
        if self.use_lpips:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
            for params in self.lpips_loss.parameters():
                params.requires_grad_(False)
            self.lpips_loss.eval()
        else:
            self.lpips_loss = None
            
        super().__init__(*args, **kwargs)

    def encode_to_latent(self, images):
        """Encode images to latent space using VQModelTorch"""
        device = images.device
        
        # Apply transform to entire batch
        # Convert from (B, C, H, W) to (B, H, W, C) for transform, then back
        images_hwc = images.permute(0, 2, 3, 1)  # (B, H, W, C)
        images_transformed = self.transform(images_hwc)  # Apply normalization
        images_transformed = images_transformed.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        images_transformed = images_transformed.to(device, torch.float32)
        
        # Encode entire batch to latent
        # Note: for encoding, we can use no_grad since we don't need gradients to flow back to input images
        with torch.no_grad():
            latents = self.autoencoder.encode(images_transformed)  # VQ model returns tensor directly
        
        return latents  # (batch_size, latent_channels, latent_height, latent_width)

    def decode_from_latent(self, latents):
        """Decode latents back to image space using VQModelTorch"""
        device = latents.device
        
        # Decode entire batch from latent
        # Note: autoencoder parameters are frozen, but we need gradients to flow through for training
        # VQ model expects float32
        latents_fp32 = latents.to(torch.float32)
        decoded = self.autoencoder.decode(latents_fp32)  # VQ model returns tensor directly
        # Denormalize: from [-1, 1] to [0, 1]
        decoded = decoded * 0.5 + 0.5
        # Keep in float32 for consistency with the rest of the pipeline
        if decoded.dtype != torch.float32:
            decoded = decoded.to(torch.float32)
        
        return decoded  # (batch_size, channels, height, width)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device
        batch_size = sources.size(0)
        
        # Encode sources and targets to latent space
        sources_latent = self.encode_to_latent(sources)
        targets_latent = self.encode_to_latent(targets)
        
        # Sample random t for each example
        ts = torch.randint(1, self.scheduler.config.T + 1, (batch_size,), device=device)
        
        # Generate noisy latents and coefs
        noisy_latents = []
        coefs = []
        for i, t in enumerate(ts):
            noisy_latent = self.scheduler.get_noisy_image(int(t.item()), targets_latent[i], sources_latent[i])
            noisy_latents.append(noisy_latent)
            coef = self.scheduler.get_loss_coef(int(t.item()))
            coefs.append(coef.to(device))
        
        noisy_latents = torch.stack(noisy_latents)
        coefs = torch.stack(coefs).view(batch_size, 1, 1, 1)

        # Forward pass in latent space
        outputs_latent = model(noisy_latents, ts, lq=sources_latent.to(device))

        # Decode outputs back to image space for loss computation
        outputs = self.decode_from_latent(outputs_latent)
        
        # Compute per-sample losses in image space
        per_sample_losses = []
        
        for b in range(batch_size):
            # MSE loss per sample
            sample_mse = (outputs[b] - targets[b]).pow(2).mean()
            
            # Start with MSE loss
            sample_loss = sample_mse
            
            # Add LPIPS loss only if use_lpips is True
            if self.use_lpips and self.lpips_loss is not None:
                sample_lpips_sum = 0.0
                for c in range(outputs.shape[1]):
                    # LPIPS expects 3-channel, so repeat to 3 channels
                    out_img = outputs[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)  # (1,3,H,W)
                    tgt_img = targets[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                    lpips_val = self.lpips_loss(out_img.to(device), tgt_img.to(device))
                    sample_lpips_sum += lpips_val
                sample_lpips = sample_lpips_sum / outputs.shape[1]
                
                # Add LPIPS to the loss
                sample_loss = sample_loss + sample_lpips
            
            # Apply loss coefficient if enabled
            if self.use_loss_coef:
                sample_loss = sample_loss * coefs[b].squeeze()
            
            per_sample_losses.append(sample_loss)
        
        # Average across batch
        loss = torch.stack(per_sample_losses).mean()
        
        # Debug: Check if loss requires gradients
        if not loss.requires_grad:
            print(f"Warning: Loss does not require gradients at step {self.state.global_step}")
            print(f"outputs_latent requires_grad: {outputs_latent.requires_grad}")
            print(f"outputs requires_grad: {outputs.requires_grad}")
            # Create a dummy loss that requires gradients
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
            
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at step {self.state.global_step}")
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
    
        # Log images every 200 steps
        step = self.state.global_step
        if step and step % 20 == 0:
            # Get first item in batch
            noisy_img = noisy_latents[0].detach().cpu().numpy()  # Latent space
            denoised_img = outputs[0].detach().cpu().numpy()    # Image space
            target_img = targets[0].detach().cpu().numpy()      # Image space
            source_img = sources[0].detach().cpu().numpy()      # Image space
            
            # Create figure with 4 rows and 5 columns (for 5 channels)
            fig, axes = plt.subplots(4, 5, figsize=(15, 12))
            
            # Plot source channels
            for i in range(5):
                axes[0, i].imshow(source_img[i], cmap='viridis')
                axes[0, i].set_title(f'Source Channel {i+1}')
                axes[0, i].axis('off')
            
            # Plot some latent channels (first 3 of latent channels, repeated to fill 5 columns)
            for i in range(5):
                latent_idx = i % min(3, noisy_img.shape[0])  # Cycle through available latent channels
                axes[1, i].imshow(noisy_img[latent_idx], cmap='viridis')
                axes[1, i].set_title(f'Noisy Latent Ch {latent_idx+1}')
                axes[1, i].axis('off')
            
            # Plot denoised channels
            for i in range(5):
                axes[2, i].imshow(denoised_img[i], cmap='viridis')
                axes[2, i].set_title(f'Denoised Channel {i+1}')
                axes[2, i].axis('off')
            
            # Plot target channels
            for i in range(5):
                axes[3, i].imshow(target_img[i], cmap='viridis')
                axes[3, i].set_title(f'Target Channel {i+1}')
                axes[3, i].axis('off')

            fig.suptitle(f'Latent Diffusion - Timestep: {ts[0]}', fontsize=16)
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                'latent_channel_comparison': wandb.Image(fig),
                'loss': loss.item()
            }, step=step)
            
            plt.close(fig)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        """Override prediction step to handle latent space diffusion process"""
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device
        batch_size = sources.size(0)
        
        with torch.no_grad():
            # Encode to latent space
            sources_latent = self.encode_to_latent(sources)
            targets_latent = self.encode_to_latent(targets)
            
            # For validation, use a fixed timestep
            t = torch.full((batch_size,), self.scheduler.config.T // 2, device=device)
            
            # Generate noisy latents
            noisy_latents = []
            for i in range(batch_size):
                noisy_latent = self.scheduler.get_noisy_image(int(t[i].item()), targets_latent[i], sources_latent[i])
                noisy_latents.append(noisy_latent)
            noisy_latents = torch.stack(noisy_latents)
            
            # Get model predictions in latent space
            outputs_latent = model(noisy_latents, t, lq=sources_latent.to(device))
            
            # Decode back to image space
            outputs = self.decode_from_latent(outputs_latent)
            
            # Compute loss in image space
            loss = ((outputs - targets).pow(2)).mean()

            # Occasionally do full diffusion process
            self._eval_batch_counter += 1
            if self._eval_batch_counter % 100 == 1:
                # Only take the first image in the batch
                source_single = sources[0].to(device).unsqueeze(0)
                target_single = targets[0].to(device).unsqueeze(0)
                
                # Encode to latent space
                source_latent = self.encode_to_latent(source_single)
                
                # Start from noisy latent
                x_latent = self.scheduler.get_noisy_image(self.scheduler.config.T, source_latent, source_latent)
                x_latent = x_latent.to(device)
                batch_size_single = x_latent.shape[0]

                intermediate_results = []
                for t in range(self.scheduler.config.T, 0, -1):
                    timesteps = torch.full((batch_size_single,), t, device=device)
                    output_latent = self.model(x_latent, timesteps, lq=source_latent)
                    
                    # Apply diffusion step in latent space
                    coef_1 = self.scheduler.get_eta_t(t - 1) / self.scheduler.get_eta_t(t)
                    coef_2 = self.scheduler.get_alpha_t(t) / self.scheduler.get_eta_t(t)
                    coef_3 = self.scheduler.config.kappa * self.scheduler.get_eta_t(t - 1) / self.scheduler.get_eta_t(t) * self.scheduler.get_alpha_t(t)

                    coef_1, coef_2, coef_3 = coef_1.to(device), coef_2.to(device), coef_3.to(device)
                    x_latent = coef_1 * x_latent + coef_2 * output_latent + coef_3 * torch.randn_like(x_latent, device=device)

                    # Decode current state to image space for visualization
                    x_decoded = self.decode_from_latent(x_latent)
                    output_decoded = self.decode_from_latent(output_latent)
                    
                    intermediate_results.append(
                        (x_decoded.detach().cpu().numpy(), output_decoded.detach().cpu().numpy())
                    )
                
                # Create visualization
                n_steps = len(intermediate_results)
                fig, axes = plt.subplots(10, n_steps + 1, figsize=(2.2*n_steps, 20))
                
                for j in range(10):
                    for i, (x, output) in enumerate(intermediate_results):
                        x_img = x[0]  # Shape: (5, 512, 512)
                        output_img = output[0]  # Shape: (5, 512, 512)
                        target_img = target_single[0].detach().cpu().numpy()  # Shape: (5, 512, 512)
                        
                        if j < 5:
                            axes[j, i].imshow(x_img[j], cmap='viridis')
                        else:
                            axes[j, i].imshow(output_img[j-5], cmap='viridis')
                        axes[j, i].axis('off')

                    # Target image
                    axes[j, n_steps].imshow(target_img[j % 5], cmap='viridis')
                    axes[j, n_steps].axis('off')
                
                plt.tight_layout()
                
                # Log diffusion process to wandb
                val_step = self._eval_batch_counter
                wandb.log({
                    'val/latent_diffusion_process': wandb.Image(fig),
                    'val_step': val_step,
                })
                
                plt.close(fig)
            
        return (loss, None, None)

# 5. Main training function
def main(args=None):
    # Start with default config
    config = TrainConfig(output_dir='/rds/user/ym429/hpc-work/dissertation/results/rescell-15step-distillation')
    
    # Override config with command line arguments if provided
    if args is not None:
        # Update config with provided arguments
        for key, value in vars(args).items():
            if hasattr(config, key):
                # For boolean flags with action='store_true', they will be False by default
                # and True if the flag is provided
                if key in ['use_diffusion', 'use_pix2pix', 'use_consistency_distillation', 'use_latent_diffusion', 'load_best_model_at_end', 'fp16']:
                    setattr(config, key, value)
                elif value is not None:  # For other arguments, only override if explicitly provided
                    setattr(config, key, value)
    
    set_seed(config.seed)
    # init WandB
    wandb.init(
        project='diffusion-denoise',
        config=asdict(config),
        dir=config.output_dir + '/wandb'
    )
    
    # Define custom metrics for wandb
    wandb.define_metric("val_step")
    wandb.define_metric("val/diffusion_process", step_metric="val_step")

    ## print the config file
    print(config)

    # Data loading setup (from data_processing.ipynb guide)
    if config.use_latent_diffusion:
        # Use 5 channels for VQ latent diffusion: all channels
        source_channels = [6, 6, 6, 7, 8]
        target_channels = [1, 2, 3, 4, 5]
    else:
        source_channels = [7, 7, 7, 7, 7]
        target_channels = [1, 2, 3, 4, 5]
    num_workers = min(24, os.cpu_count() // 2)  # Limit workers to 24 or less

    # Define plate identifiers
    train_plates = ["BR00116991", "BR00116992", "BR00116995", "BR00117024"]
    valid_plates = ["BR00116993", "BR00117025"]

    # Construct full CSV paths on the RDS
    base_csv_dir = "/home/ym429/rds/hpc-work/dissertation"
    train_csv_list = [os.path.join(base_csv_dir, f"unique_paths_{p}.csv") for p in train_plates]
    valid_csv_list = [os.path.join(base_csv_dir, f"unique_paths_{p}.csv") for p in valid_plates]

    # Create datasets
    train_dataset = DiffusionDataset(
        csv_file_list=train_csv_list,
        source_channels=source_channels,
        target_channels=target_channels,
        img_size=(512, 512)
    )
    eval_dataset = DiffusionDataset(
        csv_file_list=valid_csv_list,
        source_channels=source_channels,
        target_channels=target_channels,
        img_size=(512, 512)
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy=config.evaluation_strategy,
        # eval_steps=100,
        save_strategy=config.save_strategy,
        # save_steps=100,
        load_best_model_at_end=config.load_best_model_at_end,
        fp16=config.fp16,
        report_to=config.report_to,
        dataloader_num_workers=num_workers,
        max_grad_norm=1.0,
        save_total_limit=2,
    )

    if config.use_latent_diffusion:
        # For latent diffusion with VQ: 3 channels, 128x128 spatial resolution
        model = UNetModelSwin(
            image_size=128,  # Latent spatial size (512/4=128 with VQ compression)
            in_channels=3,   # VQ latent channels (embed_dim=3)
            model_channels=160,
            out_channels=3,  # VQ latent channels for output
            attention_resolutions=[64, 32, 16, 8],  # Scaled for 128x128
            dropout=0,
            channel_mult=[1, 2, 2, 4],
            num_res_blocks=[2, 2, 2, 2],
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=192,
            window_size=8,
            mlp_ratio=4,
            cond_lq=True,
            lq_size=128  # Latent condition size
        )
        print(f"using latent diffusion with VQ swin unet (3 channels, 128x128)")
        scheduler = DiffusionScheduler(CFG(config.kappa, config.p, config.eta_T, config.T))
        trainer = LatentDiffusionTrainer(
            scheduler=scheduler,
            use_loss_coef=config.use_loss_coef,
            use_lpips=config.use_lpips,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    elif config.use_diffusion:

        model = UNetModelSwin(
            image_size=512,
            in_channels=len(source_channels),
            model_channels=160,
            out_channels=len(target_channels),
            attention_resolutions=[64, 32, 16, 8],
            dropout=0,
            channel_mult=[1, 2, 2, 4],
            num_res_blocks=[2, 2, 2, 2],
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,  # Assuming default, since num_heads is not mentioned in the config
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=192,
            window_size=8,
            mlp_ratio=4,
            cond_lq=True,
            lq_size=512
        )
        print(f"using swin unet")
        scheduler = DiffusionScheduler(CFG(config.kappa, config.p, config.eta_T, config.T))
        trainer = DiffusionTrainer(
            scheduler=scheduler,
            use_loss_coef=config.use_loss_coef,
            use_lpips=config.use_lpips,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    elif config.use_consistency_distillation:
        # Create the same model architecture as diffusion
        model = UNetModelSwin(
            image_size=512,
            in_channels=len(source_channels),
            model_channels=160,
            out_channels=len(target_channels),
            attention_resolutions=[64, 32, 16, 8],
            dropout=0,
            channel_mult=[1, 2, 2, 4],
            num_res_blocks=[2, 2, 2, 2],
            conv_resample=True,
            dims=2,
            use_fp16=False,
            num_heads=1,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=False,
            swin_depth=2,
            swin_embed_dim=192,
            window_size=8,
            mlp_ratio=4,
            cond_lq=True,
            lq_size=512
        )
        
        # Load pretrained weights from diffusion model
        if config.cd_pretrained_path:
            print(f"Loading pretrained diffusion model from {config.cd_pretrained_path}")
            try:
                state_dict = load_hf_checkpoint(config.cd_pretrained_path, device='cpu')
                model.load_state_dict(state_dict)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Training from scratch instead.")
        
        print(f"using consistency distillation with swin unet")
        scheduler = DiffusionScheduler(CFG(config.kappa, config.p, config.eta_T, config.T))
        trainer = ConsistencyDistillationTrainer(
            scheduler=scheduler,
            ema_decay=config.cd_ema_decay,
            lambda_weight=config.cd_lambda_weight,
            target_reg_weight=config.cd_target_reg_weight,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
    elif config.use_pix2pix:
        # Determine GPU IDs
        gpu_ids = []
        if torch.cuda.is_available():
            gpu_ids = [i for i in range(torch.cuda.device_count())]
        
        model = ImprovedPix2PixModel(
            input_nc=5,  # 5 input channels
            output_nc=5,  # 5 output channels
            ngf=64,
            ndf=64,
            norm='batch',
            use_dropout=True,
            lambda_L1=100.0,
            gan_mode='vanilla',
            gpu_ids=gpu_ids
        )
        print(f"using improved pix2pix with UNet generator and PatchGAN discriminator")
        print(f"GPU IDs: {gpu_ids}")
        
        # Disable fp16 for GAN training as it can cause instability
        training_args.fp16 = False
        print("Disabled fp16 for Pix2Pix training (can cause instability)")
        
        trainer = Pix2PixTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    else:
        model = UNet(in_channels=len(source_channels), out_channels=len(target_channels))
        print(f"using plain unet")
        try:
            model = torch.compile(model)
        except Exception:
            pass
        trainer = NormalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None, help='Batch size for evaluation')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default=None, help='Learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps')
    parser.add_argument('--logging_steps', type=int, default=None, help='Logging steps')
    parser.add_argument('--evaluation_strategy', type=str, default=None, help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default=None, help='Save strategy')
    parser.add_argument('--load_best_model_at_end', action='store_true', help='Load best model at end')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--report_to', type=str, default=None, help='Report to')
    
    # Diffusion parameters
    parser.add_argument('--kappa', type=float, default=None, help='Diffusion kappa parameter')
    parser.add_argument('--p', type=float, default=None, help='Diffusion p parameter')
    parser.add_argument('--eta_T', type=float, default=None, help='Diffusion eta_T parameter')
    parser.add_argument('--T', type=int, default=None, help='Diffusion timesteps')
    parser.add_argument('--use_loss_coef', action='store_true', help='Use loss coef')
    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS loss')
    
    # Training mode flags
    parser.add_argument('--use_diffusion', action='store_true', help='Use diffusion training')
    parser.add_argument('--use_pix2pix', action='store_true', help='Use pix2pix training')
    parser.add_argument('--use_consistency_distillation', action='store_true', help='Use consistency distillation training')
    parser.add_argument('--use_latent_diffusion', action='store_true', help='Use latent diffusion training with AutoencoderDC')
    parser.add_argument('--cd_ema_decay', type=float, default=None, help='Consistency distillation ema decay')
    parser.add_argument('--cd_pretrained_path', type=str, default=None, help='Consistency distillation pretrained path')
    parser.add_argument('--cd_lambda_weight', type=float, default=None, help='Consistency distillation lambda weight')
    parser.add_argument('--cd_target_reg_weight', type=float, default=None, help='Consistency distillation target regularization weight')
    args = parser.parse_args()

    main(args)

    # accelerate launch src/train.py --output_dir /rds/user/ym429/hpc-work/dissertation/results/rescell-15-step-latent-125 --use_latent_diffusion

# accelerate launch src/train.py  --output_dir /home/ym429/rds/hpc-work/dissertation/results/

# accelerate launch src/train.py  --output_dir /rds/user/ym429/hpc-work/dissertation/results/rescell-15-step-distillation --use_consistency_distillation --cd_lambda_weight 1.0 --cd_target_reg_weight 0.0 --num_train_epochs 3 --cd_pretrained_path /rds/user/ym429/hpc-work/dissertation/results/rescell-15-step/checkpoint-69120


## down arrow : ssim ()