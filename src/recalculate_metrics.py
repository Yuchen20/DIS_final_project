import os
import numpy as np
import csv
import argparse
import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# --- Copy of the updated calculate_metrics function ---
def calculate_metrics(pred, target):
    """Calculate MSE, SSIM, and PSNR for each channel"""
    metrics = {
        'mse': [],
        'ssim': [],
        'psnr': [],
        'mse_255': [],
        'ssim_255': [],
        'psnr_255': []
    }
    
    for i in range(pred.shape[0]):  # For each channel
        pred_channel = pred[i] 
        target_channel = target[i]

        pred_norm = (pred_channel - pred_channel.min()) / (pred_channel.max() - pred_channel.min())
        target_norm = (target_channel - target_channel.min()) / (target_channel.max() - target_channel.min())
        
        # MSE
        mse = np.mean((pred_norm - target_norm) ** 2)
        
        # SSIM (convert to 0-1 range for SSIM calculation)
        ssim_val = ssim(pred_norm, target_norm, data_range=1.0)
        
        # PSNR
        psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
        
        metrics['mse'].append(mse)
        metrics['ssim'].append(ssim_val)
        metrics['psnr'].append(psnr_val)

        # normalize the target and pred to 0-255
        pred_norm_255 = (pred_channel - pred_channel.min()) / (pred_channel.max() - pred_channel.min()) * 255
        target_norm_255 = (target_channel - target_channel.min()) / (target_channel.max() - target_channel.min()) * 255
        
        mse_255 = np.mean((pred_norm_255 - target_norm_255) ** 2)
        ssim_255 = ssim(pred_norm_255, target_norm_255, data_range=255)
        psnr_255 = psnr(target_norm_255, pred_norm_255, data_range=255)

        metrics['mse_255'].append(mse_255)
        metrics['ssim_255'].append(ssim_255)
        metrics['psnr_255'].append(psnr_255)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Recalculate metrics for pred/target .npy pairs in a folder.')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing *_pred.npy and *_target.npy files')
    parser.add_argument('--output', type=str, default='new_metric.csv', help='Output CSV file name')
    args = parser.parse_args()

    folder = args.folder
    output_csv = args.output

    # Find all *_pred.npy files
    pred_files = [f for f in os.listdir(folder) if f.endswith('_pred.npy')]
    pred_files.sort()  # Optional: sort for consistency

    # Prepare CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_name', 'channel', 'mse', 'ssim', 'psnr', 'mse_255', 'ssim_255', 'psnr_255'])

        for pred_file in tqdm(pred_files):
            prefix = pred_file[:-9]  # Remove '_pred.npy'
            target_file = prefix + '_target.npy'
            pred_path = os.path.join(folder, pred_file)
            target_path = os.path.join(folder, target_file)

            if not os.path.exists(target_path):
                print(f"Warning: Target file not found for {pred_file}, skipping.")
                continue

            pred = np.load(pred_path)
            target = np.load(target_path)

            metrics = calculate_metrics(pred, target)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for ch_idx in range(pred.shape[0]):
                writer.writerow([
                    timestamp,
                    prefix,
                    ch_idx,
                    metrics['mse'][ch_idx],
                    metrics['ssim'][ch_idx],
                    metrics['psnr'][ch_idx],
                    metrics['mse_255'][ch_idx],
                    metrics['ssim_255'][ch_idx],
                    metrics['psnr_255'][ch_idx]
                ])

if __name__ == '__main__':
    main() 

# python src/recalculate_metrics.py --folder /rds/user/ym429/hpc-work/dissertation/inference_results/rescell_wlosscoef_bad_random/predictions --output /rds/user/ym429/hpc-work/dissertation/inference_results/rescell_wlosscoef_bad_random/new_metric.csv