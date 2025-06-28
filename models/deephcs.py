import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with 3 conv layers as used in DeepHCS"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = out + residual  # Skip connection
        return out


class TransformNetwork(nn.Module):
    """Transform Network - First stage of DeepHCS"""
    def __init__(self, in_channels=5, out_channels=5):
        super(TransformNetwork, self).__init__()
        nc = 64  # Base number of channels
        
        # Encoder
        self.d1_1 = nn.Conv2d(in_channels, nc, kernel_size=3, padding=1, bias=True)
        self.res1 = ResidualBlock(nc)
        self.d1_2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        
        self.d2_1 = nn.Conv2d(nc, nc*2, kernel_size=3, padding=1, bias=True)
        self.res2 = ResidualBlock(nc*2)
        self.d2_2 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        
        self.d3_1 = nn.Conv2d(nc*2, nc*4, kernel_size=3, padding=1, bias=True)
        self.res3 = ResidualBlock(nc*4)
        self.d3_2 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        
        self.d4_1 = nn.Conv2d(nc*4, nc*8, kernel_size=3, padding=1, bias=True)
        self.res4 = ResidualBlock(nc*8)
        self.d4_2 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        
        # Middle
        self.m1_1 = nn.Conv2d(nc*8, nc*16, kernel_size=3, padding=1, bias=True)
        self.res5 = ResidualBlock(nc*16)
        self.m1_2 = nn.Conv2d(nc*16, nc*16, kernel_size=3, padding=1, bias=True)
        
        # Decoder
        self.u1_1 = nn.Conv2d(nc*16, nc*8, kernel_size=3, padding=1, bias=True)
        self.u1_2 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        self.res6 = ResidualBlock(nc*8)
        self.u1_6 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        
        self.u2_1 = nn.Conv2d(nc*8, nc*4, kernel_size=3, padding=1, bias=True)
        self.u2_2 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        self.res7 = ResidualBlock(nc*4)
        self.u2_6 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        
        self.u3_1 = nn.Conv2d(nc*4, nc*2, kernel_size=3, padding=1, bias=True)
        self.u3_2 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        self.res8 = ResidualBlock(nc*2)
        self.u3_6 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        
        self.u4_1 = nn.Conv2d(nc*2, nc, kernel_size=3, padding=1, bias=True)
        self.u4_2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        self.res9 = ResidualBlock(nc)
        self.u4_6 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        
        # Output
        self.out = nn.Conv2d(nc, out_channels, kernel_size=3, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.d1_1(x))
        x = self.res1(x)
        d4 = self.relu(self.d1_2(x))
        x = self.pool(d4)
        
        x = self.relu(self.d2_1(x))
        x = self.res2(x)
        d5 = self.relu(self.d2_2(x))
        x = self.pool(d5)
        
        x = self.relu(self.d3_1(x))
        x = self.res3(x)
        d6 = self.relu(self.d3_2(x))
        x = self.pool(d6)
        
        x = self.relu(self.d4_1(x))
        x = self.res4(x)
        d7 = self.relu(self.d4_2(x))
        x = self.pool(d7)
        
        # Middle
        x = self.relu(self.m1_1(x))
        x = self.res5(x)
        x = self.relu(self.m1_2(x))
        
        # Decoder with skip connections
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u1_1(x))
        x = x + d7  # Skip connection
        x = self.relu(self.u1_2(x))
        x = self.res6(x)
        x = self.relu(self.u1_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u2_1(x))
        x = x + d6  # Skip connection
        x = self.relu(self.u2_2(x))
        x = self.res7(x)
        x = self.relu(self.u2_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u3_1(x))
        x = x + d5  # Skip connection
        x = self.relu(self.u3_2(x))
        x = self.res8(x)
        x = self.relu(self.u3_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u4_1(x))
        x = x + d4  # Skip connection
        x = self.relu(self.u4_2(x))
        x = self.res9(x)
        x = self.relu(self.u4_6(x))
        
        # Output
        x = self.relu(self.out(x))
        
        return x


class RefineNetwork(nn.Module):
    """Refinement Network - Second stage of DeepHCS"""
    def __init__(self, in_channels=10, out_channels=5):  # in_channels = 5 (transform output) + 5 (original input)
        super(RefineNetwork, self).__init__()
        nc = 64  # Base number of channels
        
        # Encoder
        self.d1_1 = nn.Conv2d(in_channels, nc, kernel_size=3, padding=1, bias=True)
        self.res1 = ResidualBlock(nc)
        self.d1_2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        
        self.d2_1 = nn.Conv2d(nc, nc*2, kernel_size=3, padding=1, bias=True)
        self.res2 = ResidualBlock(nc*2)
        self.d2_2 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        
        self.d3_1 = nn.Conv2d(nc*2, nc*4, kernel_size=3, padding=1, bias=True)
        self.res3 = ResidualBlock(nc*4)
        self.d3_2 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        
        self.d4_1 = nn.Conv2d(nc*4, nc*8, kernel_size=3, padding=1, bias=True)
        self.res4 = ResidualBlock(nc*8)
        self.d4_2 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        
        # Middle
        self.m1_1 = nn.Conv2d(nc*8, nc*16, kernel_size=3, padding=1, bias=True)
        self.res5 = ResidualBlock(nc*16)
        self.m1_2 = nn.Conv2d(nc*16, nc*16, kernel_size=3, padding=1, bias=True)
        
        # Decoder
        self.u1_1 = nn.Conv2d(nc*16, nc*8, kernel_size=3, padding=1, bias=True)
        self.u1_2 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        self.res6 = ResidualBlock(nc*8)
        self.u1_6 = nn.Conv2d(nc*8, nc*8, kernel_size=3, padding=1, bias=True)
        
        self.u2_1 = nn.Conv2d(nc*8, nc*4, kernel_size=3, padding=1, bias=True)
        self.u2_2 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        self.res7 = ResidualBlock(nc*4)
        self.u2_6 = nn.Conv2d(nc*4, nc*4, kernel_size=3, padding=1, bias=True)
        
        self.u3_1 = nn.Conv2d(nc*4, nc*2, kernel_size=3, padding=1, bias=True)
        self.u3_2 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        self.res8 = ResidualBlock(nc*2)
        self.u3_6 = nn.Conv2d(nc*2, nc*2, kernel_size=3, padding=1, bias=True)
        
        self.u4_1 = nn.Conv2d(nc*2, nc, kernel_size=3, padding=1, bias=True)
        self.u4_2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        self.res9 = ResidualBlock(nc)
        self.u4_6 = nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True)
        
        # Output
        self.out = nn.Conv2d(nc, out_channels, kernel_size=3, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        x = self.relu(self.d1_1(x))
        x = self.res1(x)
        d4 = self.relu(self.d1_2(x))
        x = self.pool(d4)
        
        x = self.relu(self.d2_1(x))
        x = self.res2(x)
        d5 = self.relu(self.d2_2(x))
        x = self.pool(d5)
        
        x = self.relu(self.d3_1(x))
        x = self.res3(x)
        d6 = self.relu(self.d3_2(x))
        x = self.pool(d6)
        
        x = self.relu(self.d4_1(x))
        x = self.res4(x)
        d7 = self.relu(self.d4_2(x))
        x = self.pool(d7)
        
        # Middle
        x = self.relu(self.m1_1(x))
        x = self.res5(x)
        x = self.relu(self.m1_2(x))
        
        # Decoder with skip connections
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u1_1(x))
        x = x + d7  # Skip connection
        x = self.relu(self.u1_2(x))
        x = self.res6(x)
        x = self.relu(self.u1_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u2_1(x))
        x = x + d6  # Skip connection
        x = self.relu(self.u2_2(x))
        x = self.res7(x)
        x = self.relu(self.u2_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u3_1(x))
        x = x + d5  # Skip connection
        x = self.relu(self.u3_2(x))
        x = self.res8(x)
        x = self.relu(self.u3_6(x))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.relu(self.u4_1(x))
        x = x + d4  # Skip connection
        x = self.relu(self.u4_2(x))
        x = self.res9(x)
        x = self.relu(self.u4_6(x))
        
        # Output
        x = self.relu(self.out(x))
        
        return x


class DeepHCSModel(nn.Module):
    """Complete DeepHCS Model with Transform and Refinement Networks"""
    def __init__(self, in_channels=5, out_channels=5):
        super(DeepHCSModel, self).__init__()
        self.transform_net = TransformNetwork(in_channels, out_channels)
        self.refine_net = RefineNetwork(in_channels + out_channels, out_channels)
        
    def forward(self, x, use_refinement=True):
        # First stage: Transform Network
        y_transform = self.transform_net(x)
        
        if not use_refinement:
            return y_transform
        
        # Second stage: Refinement Network
        # Concatenate original input and transform output
        refine_input = torch.cat([y_transform, x], dim=1)
        y_refined = self.refine_net(refine_input)
        
        return y_refined, y_transform


if __name__ == "__main__":
    # Test the model
    model = DeepHCSModel(in_channels=5, out_channels=5)
    x = torch.randn(1, 5, 512, 512)
    
    # Test transform only
    y_transform = model(x, use_refinement=False)
    print(f"Transform output shape: {y_transform.shape}")
    
    # Test full model
    y_refined, y_transform = model(x, use_refinement=True)
    print(f"Refined output shape: {y_refined.shape}")
    print(f"Transform output shape: {y_transform.shape}") 