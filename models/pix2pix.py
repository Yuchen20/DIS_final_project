import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from networks import UnetGenerator, NLayerDiscriminator, GANLoss

class Pix2PixModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ndf=64, norm='batch', use_dropout=False):
        super(Pix2PixModel, self).__init__()
        
        # Generator (U-Net)
        self.generator = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=8,  # For 512x512 images
            ngf=ngf,
            norm_layer=nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d,
            use_dropout=use_dropout
        )
        
        # Discriminator (PatchGAN)
        self.discriminator = NLayerDiscriminator(
            input_nc=input_nc + output_nc,  # Concatenated input and output
            ndf=ndf,
            n_layers=3,
            norm_layer=nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
        )
        
        # Loss functions
        self.criterionGAN = GANLoss('vanilla')
        self.criterionL1 = nn.L1Loss()
        
    def forward(self, x):
        """Forward pass for generator"""
        return self.generator(x)
    
    def get_discriminator_output(self, real_A, real_B, fake_B):
        """Get discriminator outputs for both real and fake pairs"""
        # Concatenate input and output for discriminator
        real_AB = torch.cat((real_A, real_B), 1)
        fake_AB = torch.cat((real_A, fake_B), 1)
        
        # Get discriminator outputs
        pred_real = self.discriminator(real_AB)
        pred_fake = self.discriminator(fake_AB.detach())
        
        return pred_real, pred_fake
    
    def compute_generator_loss(self, real_A, real_B, fake_B):
        """Compute generator losses (GAN + L1)"""
        # Get discriminator output for fake pair
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        
        # GAN loss
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # L1 loss
        loss_G_L1 = self.criterionL1(fake_B, real_B) * 100.0  # lambda_L1 = 100
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_G_L1
        
        return loss_G, loss_G_GAN, loss_G_L1
    
    def compute_discriminator_loss(self, real_A, real_B, fake_B):
        """Compute discriminator losses"""
        # Get discriminator outputs
        pred_real, pred_fake = self.get_discriminator_output(real_A, real_B, fake_B)
        
        # Compute losses
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Total discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        return loss_D, loss_D_real, loss_D_fake 