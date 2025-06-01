import os, sys
import random
import numpy as np
import torch
import wandb
from dataclasses import dataclass, asdict
from transformers import Trainer, TrainingArguments
from data_loader import DiffusionDataset
# from models.unet import UNetModelSwin
from noise_scheduling import CFG, DiffusionScheduler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from plain_unet import UNet
from unet import UNetModelSwin
from pix2pix import Pix2PixModel
from pix2pix_improved import ImprovedPix2PixModel
import lpips


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
    # training mode
    use_diffusion: bool = False
    use_pix2pix: bool = True
    use_consistency_distillation: bool = False
    # consistency distillation params
    cd_ema_decay: float = 0.999  # μ in the paper
    cd_pretrained_path: str = None  # Path to pretrained diffusion model
    cd_lambda_weight: float = 1.0  # Weight function λ(t_n)



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

# 4. Custom Trainer overriding loss computation
class DiffusionTrainer(Trainer):
    def __init__(self, scheduler: DiffusionScheduler, *args, **kwargs):
        self.scheduler = scheduler
        self._eval_batch_counter = 0
        # Initialize LPIPS loss (VGG backbone, on cuda:0 by default)
        self.lpips_loss = lpips.LPIPS(net='vgg').to('cuda:0')
        for params in self.lpips_loss.parameters():
            params.requires_grad_(False)
        self.lpips_loss.eval()
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

        diff = outputs - targets
        squared_diff = diff.pow(2)
        mse_loss = squared_diff.mean()

        # LPIPS expects shape (N,3,H,W) or (N,1,H,W), so flatten channel dim to batch for 5ch
        # We'll compute mean LPIPS over all channels
        lpips_total = 0.0
        for b in range(batch_size):
            lpips_sum = 0.0
            for c in range(outputs.shape[1]):
                # LPIPS expects 3-channel or 1-channel, so unsqueeze channel
                out_img = outputs[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)  # (1,3,H,W)
                tgt_img = targets[b, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                lpips_val = self.lpips_loss(out_img.to(device), tgt_img.to(device))
                lpips_sum += lpips_val
            lpips_total += lpips_sum / outputs.shape[1]
        lpips_loss = lpips_total / batch_size

        # Combine MSE and LPIPS (equal weighting)
        loss = mse_loss + lpips_loss

        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at step {self.state.global_step}")
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
    
        # log images every 100 steps
        step = self.state.global_step
        if step and step % 50 == 0:
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
    
    def __init__(self, scheduler: DiffusionScheduler, ema_decay: float = 0.999, 
                 lambda_weight: float = 1.0, *args, **kwargs):
        self.scheduler = scheduler
        self.ema_decay = ema_decay
        self.lambda_weight = lambda_weight
        self._eval_batch_counter = 0
        
        # Initialize LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='vgg').to('cuda:0')
        for params in self.lpips_loss.parameters():
            params.requires_grad_(False)
        self.lpips_loss.eval()
        
        super().__init__(*args, **kwargs)
        
        # Create EMA model (θ^- in the paper)
        self.ema_model = None
        
    def _init_ema_model(self):
        """Initialize EMA model as a copy of the current model"""
        if self.ema_model is None:
            import copy
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
    
    def _update_ema_model(self):
        """Update EMA model parameters: θ^- ← stopgrad(μθ^- + (1-μ)θ)"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _ode_solver_step(self, x_tn_plus_1, t_n_plus_1, t_n, sources):
        """
        Compute x^φ_{t_n} using the proper diffusion step (same as inference)
        This uses the same step as in SwinUNetPredictor.predict():
        x = coef_1 * x + coef_2 * output
        """
        device = x_tn_plus_1.device
        batch_size = x_tn_plus_1.shape[0]
        
        # Create timestep tensors
        t_n_plus_1_tensor = torch.full((batch_size,), t_n_plus_1, device=device)
        
        # Get model prediction at t_{n+1}
        with torch.no_grad():
            phi_output = self.model(x_tn_plus_1, t_n_plus_1_tensor, lq=sources)
        
        # Calculate coefficients for the diffusion step (same as inference.py)
        # Going from t_{n+1} to t_n (one step back in diffusion process)
        coef_1 = self.scheduler.get_eta_t(t_n) / self.scheduler.get_eta_t(t_n_plus_1)
        coef_2 = self.scheduler.get_alpha_t(t_n_plus_1) / self.scheduler.get_eta_t(t_n_plus_1)
        
        # Ensure coefficients are on the right device
        coef_1, coef_2 = coef_1.to(device), coef_2.to(device)
        
        # Apply the diffusion step: x^φ_{t_n} = coef_1 * x_{t_{n+1}} + coef_2 * output
        # Note: we don't add noise (coef_3 term) during training as in the paper
        x_phi_tn = coef_1 * x_tn_plus_1 + coef_2 * phi_output
        
        return x_phi_tn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute consistency distillation loss according to Algorithm 2"""
        sources = inputs['source']
        targets = inputs['target']
        device = sources.device
        batch_size = sources.size(0)
        
        # Initialize EMA model if not done
        if self.ema_model is None:
            self._init_ema_model()
        
        # Sample n ~ U[1, N-1]
        n = torch.randint(1, self.scheduler.config.T, (batch_size,), device=device)
        
        # Compute t_n and t_{n+1}
        total_loss = 0.0
        
        for i in range(batch_size):
            n_i = int(n[i].item())
            t_n = n_i
            t_n_plus_1 = n_i + 1
            
            # Sample x_{t_{n+1}} ~ N(x; t²_{n+1}I)
            x_tn_plus_1 = self.scheduler.get_noisy_image(t_n_plus_1, targets[i], sources[i])
            
            # Compute x^φ_{t_n} using ODE solver
            x_phi_tn = self._ode_solver_step(
                x_tn_plus_1.unsqueeze(0), 
                t_n_plus_1, 
                t_n, 
                sources[i].unsqueeze(0)
            )
            
            # Compute f_θ(x_{t_{n+1}}, t_{n+1})
            t_n_plus_1_tensor = torch.tensor([t_n_plus_1], device=device)
            f_theta_tn_plus_1 = model(
                x_tn_plus_1.unsqueeze(0), 
                t_n_plus_1_tensor, 
                lq=sources[i].unsqueeze(0)
            )
            
            # Compute f_{θ^-}(x^φ_{t_n}, t_n) using EMA model
            t_n_tensor = torch.tensor([t_n], device=device)
            with torch.no_grad():
                f_theta_minus_tn = self.ema_model(
                    x_phi_tn, 
                    t_n_tensor, 
                    lq=sources[i].unsqueeze(0)
                )
            
            # Compute distance d(f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ^-}(x^φ_{t_n}, t_n))
            # Using MSE + LPIPS as the distance metric
            
            # MSE loss
            mse_loss = (f_theta_tn_plus_1 - f_theta_minus_tn).pow(2).mean()
            
            # LPIPS loss (average over channels)
            lpips_sum = 0.0
            for c in range(f_theta_tn_plus_1.shape[1]):
                out_img = f_theta_tn_plus_1[0, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                tgt_img = f_theta_minus_tn[0, c].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                lpips_val = self.lpips_loss(out_img, tgt_img)
                lpips_sum += lpips_val
            lpips_loss = lpips_sum / f_theta_tn_plus_1.shape[1]
            
            # Combine losses with lambda weighting
            loss_i = self.lambda_weight * (mse_loss + lpips_loss)
            total_loss += loss_i
        
        # Average loss over batch
        loss = total_loss / batch_size
        
        # Log images every 100 steps
        step = self.state.global_step
        if step and step % 100 == 0:
            with torch.no_grad():
                # Visualize the consistency training process
                i = 0  # First item in batch
                n_i = int(n[i].item())
                t_n_plus_1 = n_i + 1
                
                x_tn_plus_1 = self.scheduler.get_noisy_image(t_n_plus_1, targets[i], sources[i])
                t_n_plus_1_tensor = torch.tensor([t_n_plus_1], device=device)
                f_theta_tn_plus_1 = model(
                    x_tn_plus_1.unsqueeze(0), 
                    t_n_plus_1_tensor, 
                    lq=sources[i].unsqueeze(0)
                )[0].detach().cpu().numpy()
                
                target_img = targets[i].detach().cpu().numpy()
                x_tn_plus_1_img = x_tn_plus_1.detach().cpu().numpy()
                
                fig, axes = plt.subplots(3, 5, figsize=(20, 12))
                
                # Plot noisy input
                for j in range(5):
                    axes[0, j].imshow(x_tn_plus_1_img[j], cmap='viridis')
                    axes[0, j].set_title(f'Noisy (t={t_n_plus_1}) Ch {j+1}')
                    axes[0, j].axis('off')
                
                # Plot model output
                for j in range(5):
                    axes[1, j].imshow(f_theta_tn_plus_1[j], cmap='viridis')
                    axes[1, j].set_title(f'Model Output Ch {j+1}')
                    axes[1, j].axis('off')
                
                # Plot target
                for j in range(5):
                    axes[2, j].imshow(target_img[j], cmap='viridis')
                    axes[2, j].set_title(f'Target Ch {j+1}')
                    axes[2, j].axis('off')
                
                fig.suptitle(f'Consistency Distillation Training (n={n_i})', fontsize=16)
                plt.tight_layout()
                
                wandb.log({
                    'cd_training_comparison': wandb.Image(fig),
                    'cd_loss': loss.item()
                }, step=step)
                
                plt.close(fig)
        
        return (loss, None) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to update EMA model after each step"""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Update EMA model after gradient step
        self._update_ema_model()
        
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
            
            # Occasionally visualize full denoising process
            self._eval_batch_counter += 1
            if self._eval_batch_counter % 100 == 1:
                # Take first image
                source = sources[0].unsqueeze(0)
                target = targets[0].unsqueeze(0)
                
                # Start from pure noise
                x = torch.randn_like(target, device=device)
                
                # Single-step denoising (consistency model property)
                t = torch.tensor([1], device=device)
                denoised = model(x, t, lq=source)
                
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                
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
                
                plt.tight_layout()
                
                wandb.log({
                    'val/cd_single_step_denoising': wandb.Image(fig),
                    'val_step': self._eval_batch_counter,
                })
                
                plt.close(fig)
        
        return (loss, None, None)

# 5. Main training function
def main(custom_config=None):
    # config = TrainConfig()
    
    if custom_config is not None:
        config = custom_config
    else:
        config = TrainConfig(output_dir='/rds/user/ym429/hpc-work/dissertation/results/pix2pix')
    
    set_seed(config.seed)
    # init WandB
    wandb.init(
        project='diffusion-denoise',
        config=asdict(config),
        dir='/rds/user/ym429/hpc-work/dissertation/results/pix2pix/wandb'
    )
    
    # Define custom metrics for wandb
    wandb.define_metric("val_step")
    wandb.define_metric("val/diffusion_process", step_metric="val_step")

    # Data loading setup (from data_processing.ipynb guide)
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

    if config.use_diffusion:

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
            checkpoint = torch.load(config.cd_pretrained_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Successfully loaded pretrained weights")
        else:
            print("WARNING: No pretrained path specified for consistency distillation!")
            print("Training from scratch, which is not recommended.")
        
        print(f"using consistency distillation with swin unet")
        scheduler = DiffusionScheduler(CFG(config.kappa, config.p, config.eta_T, config.T))
        trainer = ConsistencyDistillationTrainer(
            scheduler=scheduler,
            ema_decay=config.cd_ema_decay,
            lambda_weight=config.cd_lambda_weight,
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
    main()

# accelerate launch src/train.py  --output_dir /home/ym429/rds/hpc-work/dissertation/results/




## down arrow : ssim ()