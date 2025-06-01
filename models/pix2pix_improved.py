import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from networks import define_G, define_D, GANLoss, init_weights


class ImprovedPix2PixModel(nn.Module):
    """
    Improved Pix2Pix model based on the official implementation.
    This class implements the pix2pix model for learning a mapping from input images to output images.
    
    The model training uses:
    - UNet256 generator
    - PatchGAN discriminator  
    - Vanilla GAN loss + L1 loss
    
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    
    def __init__(self, 
                 input_nc=5, 
                 output_nc=5, 
                 ngf=64, 
                 ndf=64, 
                 norm='batch', 
                 use_dropout=True,
                 init_type='normal',
                 init_gain=0.02,
                 lambda_L1=100.0,
                 gan_mode='vanilla',
                 gpu_ids=[]):
        """
        Initialize the Pix2Pix model.
        
        Parameters:
            input_nc (int)    -- the number of channels in input images
            output_nc (int)   -- the number of channels in output images  
            ngf (int)         -- the number of filters in the last conv layer of generator
            ndf (int)         -- the number of filters in the first conv layer of discriminator
            norm (str)        -- the type of normalization layers used in the network
            use_dropout (bool)-- if use dropout layers
            init_type (str)   -- the name of the initialization method
            init_gain (float) -- scaling factor for normal, xavier and orthogonal
            lambda_L1 (float) -- weight for L1 loss
            gan_mode (str)    -- the type of GAN objective: vanilla | lsgan | wgangp
            gpu_ids (list)    -- which GPUs the network runs on
        """
        super(ImprovedPix2PixModel, self).__init__()
        
        # Store configuration
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.lambda_L1 = lambda_L1
        self.gpu_ids = gpu_ids
        
        # Initialize networks first
        self.netG = define_G(input_nc, output_nc, ngf, 'unet_256', norm, 
                            use_dropout, init_type, init_gain, gpu_ids)
        self.netD = define_D(input_nc + output_nc, ndf, 'basic', 3, norm, 
                            init_type, init_gain, gpu_ids)
        
        # Detect device after networks are created
        # The networks might be automatically placed on CUDA by define_G/define_D
        self.device = next(self.netG.parameters()).device
        
        # Initialize loss functions and move to the correct device
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()
        
        # Initialize loss storage for tracking
        self.loss_G_GAN = 0
        self.loss_G_L1 = 0  
        self.loss_D_real = 0
        self.loss_D_fake = 0
        
        # Store input/output for visualization
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        
    def set_input(self, input_dict):
        """Unpack input data and move to device."""
        self.real_A = input_dict['source'].to(self.device)
        self.real_B = input_dict['target'].to(self.device)
        
    def forward(self, x=None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if x is not None:
            # Direct forward for inference (used by Trainer)
            return self.netG(x)
        else:
            # Use stored real_A (used by optimize_parameters)
            self.fake_B = self.netG(self.real_A)
            return self.fake_B
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # Combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B (L1 loss)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        
        # Combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        return self.loss_G
    
    def optimize_parameters_manual(self, optimizer_G, optimizer_D):
        """
        Manual optimization step (used outside of Trainer for reference)
        This shows how the training should work conceptually.
        """
        # Forward pass
        self.forward()
        
        # Update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        optimizer_D.zero_grad()                  # set D's gradients to zero
        loss_D = self.backward_D()               # calculate gradients for D
        loss_D.backward()
        optimizer_D.step()                       # update D's weights
        
        # Update G  
        self.set_requires_grad(self.netD, False) # D requires no gradients when optimizing G
        optimizer_G.zero_grad()                  # set G's gradients to zero
        loss_G = self.backward_G()               # calculate gradients for G
        loss_G.backward()
        optimizer_G.step()                       # update G's weights
        
        return loss_G, loss_D
    
    def compute_generator_loss(self, real_A, real_B, fake_B):
        """
        Compute generator losses for use with HuggingFace Trainer.
        This method is called by the custom Trainer.
        """
        # Store inputs for potential visualization
        self.real_A = real_A
        self.real_B = real_B  
        self.fake_B = fake_B
        
        # GAN loss: G(A) should fool the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # L1 loss: G(A) should be close to B
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_G_L1
        
        # Store losses for logging
        self.loss_G_GAN = loss_G_GAN
        self.loss_G_L1 = loss_G_L1
        
        return loss_G, loss_G_GAN, loss_G_L1
    
    def compute_discriminator_loss(self, real_A, real_B, fake_B):
        """
        Compute discriminator losses for use with HuggingFace Trainer.
        This method is called by the custom Trainer.
        """
        # Fake: D should classify G(A,B) as fake
        fake_AB = torch.cat((real_A, fake_B.detach()), 1)  # detach to avoid backprop through G
        pred_fake = self.netD(fake_AB)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real: D should classify (A,B) as real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Total discriminator loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        
        # Store losses for logging
        self.loss_D_real = loss_D_real
        self.loss_D_fake = loss_D_fake
        
        return loss_D, loss_D_real, loss_D_fake
    
    def get_current_losses(self):
        """Return training losses for logging"""
        return {
            'G_GAN': float(self.loss_G_GAN) if hasattr(self, 'loss_G_GAN') else 0.0,
            'G_L1': float(self.loss_G_L1) if hasattr(self, 'loss_G_L1') else 0.0,
            'D_real': float(self.loss_D_real) if hasattr(self, 'loss_D_real') else 0.0,
            'D_fake': float(self.loss_D_fake) if hasattr(self, 'loss_D_fake') else 0.0
        }
    
    def get_current_visuals(self):
        """Return current images for visualization"""
        visual_ret = {}
        if self.real_A is not None:
            visual_ret['real_A'] = self.real_A
        if self.fake_B is not None:
            visual_ret['fake_B'] = self.fake_B  
        if self.real_B is not None:
            visual_ret['real_B'] = self.real_B
        return visual_ret
    
    def eval(self):
        """Set the model to evaluation mode"""
        self.netG.eval()
        self.netD.eval()
        return self
    
    def train(self, mode=True):
        """Set the model to training mode"""
        self.netG.train(mode)
        self.netD.train(mode)
        return self 