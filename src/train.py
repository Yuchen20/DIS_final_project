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


# 1. Configuration
@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 5e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    weight_decay: float = 0.005
    lr_scheduler_type: str = 'cosine'
    warmup_steps: int = 5000
    logging_steps: int = 10
    evaluation_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    load_best_model_at_end: bool = True
    fp16: bool = True
    report_to: str = 'wandb'
    output_dir: str = 'results'
    # diffusion params
    kappa: float = 2.0
    p: float = 0.3
    eta_T: float = 0.999
    T: int = 15
    # training mode
    use_diffusion: bool = True

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
        outputs = model(noisy, ts)

        
        diff = outputs - targets
        squared_diff = diff.pow(2)
        squared_diff = torch.clamp(squared_diff, min=1e-8, max=1e8)
        loss = (squared_diff * coefs).mean()

        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at step {self.state.global_step}")
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
    
        # log images every 100 steps
        step = self.state.global_step
        if step and step % 500 == 0:
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
            coef = self.scheduler.get_loss_coef(int(t[0].item())).to(device)  # Ensure coef is on the same device
            
            # Generate noisy images
            noisy = []
            for i in range(batch_size):
                noisy_img = self.scheduler.get_noisy_image(int(t[i].item()), targets[i], sources[i])
                noisy.append(noisy_img)
            noisy = torch.stack(noisy)
            
            # Get model predictions
            outputs = model(noisy, t)
            
            # Compute loss
            loss = ((outputs - targets).pow(2)).mean()
            
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

# 5. Main training function
def main():
    # config = TrainConfig()
    config = TrainConfig(use_diffusion=True, output_dir='/home/ym429/rds/hpc-work/dissertation/results/')
    set_seed(config.seed)
    # init WandB
    wandb.init(
        project='diffusion-denoise',
        config=asdict(config),
        dir='/home/ym429/rds/hpc-work/dissertation/results/wandb'  # Set wandb directory
    )

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

    if config.use_diffusion:
        model = UNetModelSwin(
            image_size=512, 
            in_channels=5, 
            model_channels=128, 
            out_channels=5, 
            num_res_blocks=2, 
            attention_resolutions=(256, 128, 64), 
            cond_lq=False
        )
        print(f"using swin unet")
    else:
        model = UNet(in_channels=len(source_channels), out_channels=len(target_channels))
        print(f"using plain unet")
        try:
            model = torch.compile(model)
        except Exception:
            pass


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
    )

    if config.use_diffusion:
        # Initialize diffusion scheduler
        scheduler = DiffusionScheduler(CFG(config.kappa, config.p, config.eta_T, config.T))
        trainer = DiffusionTrainer(
            scheduler=scheduler,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    else:
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