import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    """Generate Gaussian window"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    """Create 2D Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """SSIM Loss as used in DeepHCS"""
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        
        return self._ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # Adjust window channel to match input
        if img1.size(1) != channel:
            window = window.expand(img1.size(1), 1, window_size, window_size)
            
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss as used in DeepHCS"""
    def __init__(self, window_size=11, size_average=True, channel=1, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # Default weights from the paper
        if weights is None:
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        else:
            self.weights = weights
            
        self.levels = len(self.weights)
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        
        return self._ms_ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

    def _ms_ssim(self, img1, img2, window, window_size, channel, size_average=True):
        weights = torch.FloatTensor(self.weights).to(img1.device)
        mssim = []
        mcs = []
        
        for i in range(self.levels):
            ssim_val, cs = self._ssim_cs(img1, img2, window, window_size, channel, size_average)
            mssim.append(ssim_val)
            mcs.append(cs)
            
            # Downsample for next level
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        
        # Compute MS-SSIM
        pow1 = mcs ** weights
        pow2 = mssim ** weights
        
        # Use only contrast sensitivity for all but the last level
        output = torch.prod(pow1[:-1]) * pow2[-1]
        
        return 1.0 - output  # Return as loss (lower is better)

    def _ssim_cs(self, img1, img2, window, window_size, channel, size_average=True):
        # Adjust window channel to match input
        if img1.size(1) != channel:
            window = window.expand(img1.size(1), 1, window_size, window_size)
            
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        luminance = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
        contrast_structure = (2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
        
        ssim_val = luminance * contrast_structure
        cs = contrast_structure

        if size_average:
            return ssim_val.mean(), cs.mean()
        else:
            return ssim_val.mean(1).mean(1).mean(1), cs.mean(1).mean(1).mean(1)


def ssim_loss(img1, img2, window_size=11, size_average=True):
    """Functional SSIM loss"""
    ssim_module = SSIMLoss(window_size, size_average, img1.size(1))
    return 1.0 - ssim_module(img1, img2)


def ms_ssim_loss(img1, img2, window_size=11, size_average=True, weights=None):
    """Functional MS-SSIM loss"""
    ms_ssim_module = MSSSIMLoss(window_size, size_average, img1.size(1), weights)
    return ms_ssim_module(img1, img2) 