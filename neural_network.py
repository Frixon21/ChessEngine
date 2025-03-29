#neural_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A single residual block with two 3x3 convolutions, each followed by batch normalization.
    The input is added to the output (residual connection) and passed through a ReLU.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    """
    A simplified AlphaZero-like network for chess:
      - Input: tensor of shape (batch_size, 12, 8, 8)
      - An initial convolutional layer to project 12 channels to a higher-dimensional space.
      - A tower of residual blocks.
      - Two heads: a policy head and a value head.
    
    Parameters:
      num_blocks: Number of residual blocks.
      channels: Number of channels after the initial convolution.
      num_moves: Number of output move logits (4352 in your scheme).
    """
    def __init__(self, num_blocks=10, channels=256, num_moves=4352):
        super(ChessNet, self).__init__()
       # Initial convolution + BN
        self.conv_initial = nn.Conv2d(12, channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        
        # Policy head: 1x1 convolution, BN, then fully connected layer to num_moves
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)
        
        # Value head: 1x1 convolution, BN, then two fully connected layers
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # x: (batch_size, 12, 8, 8)
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        x = self.res_blocks(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # flatten
        p = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
    