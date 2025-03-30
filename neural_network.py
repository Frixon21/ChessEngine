#neural_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


from board_encoder import TOTAL_PLANES # Should now 22



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that adaptively recalibrates channel-wise
    feature responses by explicitly modeling interdependencies between channels.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: Two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # Output values between 0 and 1 to scale channels
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale (Excitation applied to input x)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    A single residual block with two 3x3 convolutions, BatchNorm, ReLU,
    and a Squeeze-and-Excitation block.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) # Bias often False with BN
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # <<< Add SEBlock >>>
        self.se = SEBlock(channels) # Using default reduction=16

    def forward(self, x):
        residual = x
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv block
        out = self.bn2(self.conv2(out))
        # <<< Apply SEBlock >>>
        out = self.se(out)
        # Add residual connection
        out += residual
        # Final activation
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    """
    Leela-like network with SE-ResNet blocks.
      - Input: tensor of shape (batch_size, TOTAL_PLANES, 8, 8)
      - Initial Conv Layer projects TOTAL_PLANES -> channels
      - Tower of SE-Residual blocks.
      - Policy and Value heads.
    """
    def __init__(self, num_blocks=20, channels=256, num_moves=4352):
        super(ChessNet, self).__init__()
        
        # Initial convolution + BN (accepts TOTAL_PLANES)
        self.conv_initial = nn.Conv2d(TOTAL_PLANES, channels, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(channels)

        # Residual tower using the NEW ResidualBlock (with SE)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head (structure remains the same)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves) # 128 features flattened

        # Value head (structure remains the same)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256) # 64 features flattened -> 256 hidden
        self.value_fc2 = nn.Linear(256, 1) # 256 hidden -> 1 output value
        
    def forward(self, x):
        # Input: (batch_size, TOTAL_PLANES, 8, 8)
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        # Pass through the tower of SE-Residual blocks
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1); p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1); v = F.relu(self.value_fc1(v)); v = torch.tanh(self.value_fc2(v))

        return p, v