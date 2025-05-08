import os, sys
import random
import numpy as np
import torch
import wandb
from dataclasses import dataclass, asdict
from transformers import Trainer, TrainingArguments
from data_loader import DiffusionDataset
from noise_scheduling import CFG, DiffusionScheduler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from plain_unet import UNet



# 1. Configuration
@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 5e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    weight_decay: float = 0.0
    lr_scheduler_type: str = 'cosine'
    warmup_steps: int = 5000
    logging_steps: int = 10
    evaluation_strategy: str = 'steps'
    save_strategy: str = 'steps'
    load_best_model_at_end: bool = True
    fp16: bool = True
    report_to: str = 'wandb'
    output_dir: str = 'results'
    # diffusion params
    kappa: float = 2.0
    p: float = 0.5
    eta_T: float = 0.999
    T: int = 50
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
        batch_size = sources.size(0)
        # sample random t for each example
        ts = torch.randint(1, self.scheduler.config.T + 1, (batch_size,), device=sources.device)
        # generate noisy inputs and coefs
        noisy = []
        coefs = []
        for i, t in enumerate(ts):
            noisy_img = self.scheduler.get_noisy_image(int(t.item()), targets[i], sources[i])
            noisy.append(noisy_img)
            coefs.append(self.scheduler.get_loss_coef(int(t.item())))
        noisy = torch.stack(noisy)
        coefs = torch.stack(coefs).view(batch_size, 1, 1, 1)
        # forward pass
        outputs = model(noisy, ts)
        # mse loss scaled by coef
        loss = ((outputs - targets).pow(2) * coefs).mean()

        # log images every 100 steps
        step = self.state.global_step
        if step and step % 1000 == 0:
            # log first item in batch
            noisy_img = noisy[0].detach().cpu().numpy()
            denoised_img = outputs[0].detach().cpu().numpy()
            # reshape for wandb (C,H,W) or (H,W,C)
            # convert to H,W,C
            noisy_img = np.transpose(noisy_img, (1,2,0))
            denoised_img = np.transpose(denoised_img, (1,2,0))
            wandb.log({
                'noisy_image': wandb.Image(noisy_img, caption=f'step_{step}_noisy'),
                'denoised_image': wandb.Image(denoised_img, caption=f'step_{step}_denoised')
            }, step=step)

        return (loss, outputs) if return_outputs else loss

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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to handle dictionary inputs"""
        sources = inputs['source']
        targets = inputs['target']
        
        with torch.no_grad():
            outputs = model(sources)
            loss = (outputs - targets).pow(2).mean()
            
        return (loss, outputs, targets)

# 5. Main training function
def main():
    # config = TrainConfig()
    config = TrainConfig(use_diffusion=False)
    set_seed(config.seed)
    # init WandB
    wandb.init(project='diffusion-denoise', config=asdict(config))

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

    model = UNet(in_channels=5, out_channels=5)

    # speed up with torch.compile
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
        eval_steps=100,
        save_strategy=config.save_strategy,
        save_steps=100,
        load_best_model_at_end=config.load_best_model_at_end,
        fp16=config.fp16,
        report_to=config.report_to,
        dataloader_num_workers=num_workers
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