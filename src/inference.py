import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DiffusionDataset, visualize_image_channels
import argparse
from tqdm import tqdm
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import datetime
from abc import ABC, abstractmethod
from noise_scheduling import DiffusionScheduler, CFG
import wandb
from safetensors.torch import load_file as load_safetensors

# Assume `model_weights` is loaded from safetensors
from collections import OrderedDict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from plain_unet import UNet
from unet import UNetModelSwin
from pix2pix import Pix2PixModel

class BasePredictor(ABC):
    """Base class for model predictors"""
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()

    @abstractmethod
    def load_model(self, model_path):
        """Load the model from checkpoint"""
        pass

    def _load_hf_checkpoint(self, model_path):
        """Helper method to load Hugging Face Trainer checkpoint"""
        if os.path.isdir(model_path):
            # For Hugging Face Trainer saved models
            safetensors_path = os.path.join(model_path, 'model.safetensors')
            bin_path = os.path.join(model_path, 'pytorch_model.bin')

            if os.path.exists(safetensors_path):
                state_dict = load_safetensors(safetensors_path, device=self.device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k.replace("_orig_mod.", "")  # remove the prefix
                    new_state_dict[new_key] = v

                return new_state_dict
            elif os.path.exists(bin_path):
                return torch.load(bin_path, map_location=self.device)
            else:
                raise ValueError(f"Could not find model weights at {safetensors_path} or {bin_path}")
        else:
            # For regular checkpoint files
            return torch.load(model_path, map_location=self.device)

    @abstractmethod
    def predict(self, sources):
        """Run inference on input sources"""
        pass

    def log_images(self, sources, predictions, targets, step):
        """Log images to wandb for all predictors"""
        if step % 50 == 0:
            # Get first image from batch
            source_img = sources[0].detach().cpu().numpy()
            pred_img = predictions[0].detach().cpu().numpy()
            target_img = targets[0].detach().cpu().numpy()
            
            # Create figure with 3 rows (source, prediction, target) and 5 columns (channels)
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            
            # Plot source channels
            for j in range(5):
                axes[0, j].imshow(source_img[j], cmap='viridis')
                axes[0, j].set_title(f'Source Channel {j+1}')
                axes[0, j].axis('off')
            
            # Plot prediction channels
            for j in range(5):
                axes[1, j].imshow(pred_img[j], cmap='viridis')
                axes[1, j].set_title(f'Pred Channel {j+1}')
                axes[1, j].axis('off')
            
            # Plot target channels
            for j in range(5):
                axes[2, j].imshow(target_img[j], cmap='viridis')
                axes[2, j].set_title(f'Target Channel {j+1}')
                axes[2, j].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                'inference_comparison': wandb.Image(fig),
            }, step=step)
            
            plt.close(fig)

class UNetPredictor(BasePredictor):
    """Predictor for standard UNet model"""
    def load_model(self, model_path):
        model = UNet(in_channels=5, out_channels=5)
        checkpoint = self._load_hf_checkpoint(model_path)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model

    def predict(self, sources):
        with torch.no_grad():
            sources = sources.to(self.device)
            return self.model(sources)

    def log_images(self, sources, predictions, targets, step):
        """Log images to wandb"""
        if step % 50 == 0:
            # Get first image from batch
            source_img = sources[0].detach().cpu().numpy()
            pred_img = predictions[0].detach().cpu().numpy()
            target_img = targets[0].detach().cpu().numpy()
            
            # Create figure with 3 rows (source, prediction, target) and 5 columns (channels)
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            
            # Plot source channels
            for j in range(5):
                axes[0, j].imshow(source_img[j], cmap='viridis')
                axes[0, j].set_title(f'Source Channel {j+1}')
                axes[0, j].axis('off')
            
            # Plot prediction channels
            for j in range(5):
                axes[1, j].imshow(pred_img[j], cmap='viridis')
                axes[1, j].set_title(f'Pred Channel {j+1}')
                axes[1, j].axis('off')
            
            # Plot target channels
            for j in range(5):
                axes[2, j].imshow(target_img[j], cmap='viridis')
                axes[2, j].set_title(f'Target Channel {j+1}')
                axes[2, j].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                'inference_comparison': wandb.Image(fig),
            }, step=step)
            
            plt.close(fig)

class SwinUNetPredictor(BasePredictor):
    """Predictor for Swin UNet model with diffusion process"""
    def load_model(self, model_path):
        model = UNetModelSwin(
            image_size=512,
            in_channels=5,
            model_channels=128,
            out_channels=5,
            num_res_blocks=2,
            attention_resolutions=(256, 128, 64),
            cond_lq=False
        )
        checkpoint = self._load_hf_checkpoint(model_path)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model

    def predict(self, sources):
        config = CFG(kappa=2.0, p=0.3, eta_T=0.999, T=15)
        scheduler = DiffusionScheduler(config)
        
        # Initialize x with noisy image
        t = config.T
        x = scheduler.get_noisy_image(t, sources, sources)
        
        # Store intermediate results for logging
        intermediate_results = []
        
        with torch.no_grad():
            x = x.to(self.device)
            for t in range(config.T + 1, 1, -1):
                t = torch.tensor(t, device=self.device)
                output = self.model(x, t)
                coef_1 = scheduler.get_eta_t(t - 1) / scheduler.get_eta_t(t)
                coef_2 = scheduler.get_alpha_t(t) / scheduler.get_eta_t(t)
                coef_1, coef_2 = coef_1.to(self.device), coef_2.to(self.device)
                x = coef_1 * x + coef_2 * output
                intermediate_results.append(x.cpu().numpy())
        
        return x, intermediate_results

    def log_images(self, sources, predictions, targets, step):
        """Override log_images to include diffusion process"""
        # First call parent class's log_images
        super().log_images(sources, predictions, targets, step)
        
        # Then log diffusion process if we have intermediate results
        if step % 50 == 0 and hasattr(self, 'intermediate_results'):
            # Create figure for diffusion process
            n_steps = len(self.intermediate_results)
            fig, axes = plt.subplots(n_steps, 5, figsize=(20, 4*n_steps))
            
            for i, step_result in enumerate(self.intermediate_results):
                for j in range(5):
                    axes[i, j].imshow(step_result[0, j], cmap='viridis')
                    axes[i, j].set_title(f'Step {n_steps-i} Channel {j+1}')
                    axes[i, j].axis('off')
            
            plt.tight_layout()
            
            # Log diffusion process to wandb
            wandb.log({
                'diffusion_process': wandb.Image(fig),
            }, step=step)
            
            plt.close(fig)

class Pix2PixPredictor(BasePredictor):
    """Predictor for Pix2Pix model"""
    def load_model(self, model_path):
        model = Pix2PixModel(
            input_nc=3,  # 3 input channels
            output_nc=3,  # 3 output channels
            ngf=64,
            ndf=64,
            norm='batch',
            use_dropout=False  # No dropout during inference
        )
        checkpoint = self._load_hf_checkpoint(model_path)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model.generator  # Only use generator for inference

    def predict(self, sources):
        with torch.no_grad():
            sources = sources.to(self.device)
            return self.model(sources)

def get_predictor(model_type, model_path, device):
    """Factory function to get the appropriate predictor"""
    if model_type == 'unet':
        return UNetPredictor(model_path, device)
    elif model_type == 'swin_unet':
        return SwinUNetPredictor(model_path, device)
    elif model_type == 'pix2pix':
        return Pix2PixPredictor(model_path, device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with UNet model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['unet', 'swin_unet', 'pix2pix'], help='Type of model to use')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    return parser.parse_args()

def calculate_metrics(pred, target):
    """Calculate MSE, SSIM, and PSNR for each channel"""
    metrics = {
        'mse': [],
        'ssim': [],
        'psnr': []
    }
    
    for i in range(pred.shape[0]):  # For each channel
        pred_channel = pred[i]
        target_channel = target[i]
        
        # MSE
        mse = np.mean((pred_channel - target_channel) ** 2)
        
        # SSIM (convert to 0-1 range for SSIM calculation)
        pred_norm = (pred_channel - pred_channel.min()) / (pred_channel.max() - pred_channel.min())
        target_norm = (target_channel - target_channel.min()) / (target_channel.max() - target_channel.min())
        ssim_val = ssim(pred_norm, target_norm, data_range=1.0)
        
        # PSNR
        psnr_val = psnr(target_channel, pred_channel, data_range=target_channel.max() - target_channel.min())
        
        metrics['mse'].append(mse)
        metrics['ssim'].append(ssim_val)
        metrics['psnr'].append(psnr_val)
    
    return metrics

def save_metrics_to_csv(metrics, filename, prefix, channel_idx):
    """Save metrics to CSV file"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'image_name', 'channel', 'mse', 'ssim', 'psnr'])
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            timestamp,
            prefix,
            channel_idx,
            metrics['mse'][channel_idx],
            metrics['ssim'][channel_idx],
            metrics['psnr'][channel_idx]
        ])

def save_predictions(pred, target, prefix, output_dir):
    """Save prediction and target as separate npz files"""
    # Create directory    
    os.makedirs(output_dir, exist_ok=True)  
    prefix = prefix.split('/')[-1]
    # Save prediction and target as separate npz files
    npz_path_target = os.path.join(output_dir, f'{prefix}_target.npy')
    npz_path_pred = os.path.join(output_dir, f'{prefix}_pred.npy')

    np.save(npz_path_target, target)
    np.save(npz_path_pred, pred)

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project='diffusion-inference',
        config=vars(args),
        dir=args.output_dir
    )
    
    # Define test plates
    test_plates = ["BR00116994", "BR00117026", "BR00118041", "BR00118045"]
    base_csv_dir = "/home/ym429/rds/hpc-work/dissertation"
    test_csv_list = [os.path.join(base_csv_dir, f"unique_paths_{p}.csv") for p in test_plates]
    
    # Create test dataset
    test_dataset = DiffusionDataset(
        csv_file_list=test_csv_list,
        source_channels=[7, 7, 7, 7, 7],  # 3 input channels
        target_channels=[1, 2, 3, 4, 5],  # 3 output channels
        img_size=(512, 512),
        get_prefix=True
    )
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Get predictor
    predictor = get_predictor(args.model_type, args.model_path, args.device)
    
    # Create metrics CSV file
    metrics_file = os.path.join(args.output_dir, 'metrics.csv')
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)

    
    # Run inference
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
        sources, target, prefix = batch
        
        # Get model predictions using the predictor
        if isinstance(predictor, SwinUNetPredictor):
            output, intermediate_results = predictor.predict(sources)
            predictor.intermediate_results = intermediate_results
        else:
            output = predictor.predict(sources)
        
        # Log images
        predictor.log_images(sources, output, target, batch_idx)
        
        # Move to CPU and convert to numpy
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        
        # Process each image in the batch
        for i in range(output.shape[0]):
            # Calculate metrics for this image
            metrics = calculate_metrics(output[i], target[i])
            
            # Save metrics for each channel
            for ch_idx in range(5):  # 3 channels for pix2pix
                save_metrics_to_csv(metrics, metrics_file, prefix[i], ch_idx)
            
            # Save predictions and targets
            save_predictions(output[i], target[i], prefix[i], os.path.join(args.output_dir, 'predictions'))
    
    # Close wandb
    wandb.finish()

if __name__ == '__main__':
    main() 


# To run the script:
## For UNet:
# accelerate run src/inference.py --model_path /home/ym429/rds/hpc-work/dissertation/results/Unet/model.safetensors --model_type unet --output_dir /home/ym429/rds/hpc-work/dissertation/inference_results/unet

## For Swin UNet:
# python inference.py --model_path /home/ym429/rds/hpc-work/dissertation/models/swin_unet_checkpoint.pth --model_type swin_unet --output_dir /home/ym429/rds/hpc-work/dissertation/inference_results/swin_unet

## For Pix2Pix:
# python inference.py --model_path /path/to/pix2pix_checkpoint.pth --model_type pix2pix --output_dir /home/ym429/rds/hpc-work/dissertation/inference_results/pix2pix
