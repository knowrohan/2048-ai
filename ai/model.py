import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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

class ZeroNet(nn.Module):
    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head yields logits for 4 possible directions
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 4 * 4, 4)
        
        # Value head yields a scalar estimation in [-1, 1]
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 4 * 4, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        x has shape (Batch, 1, 4, 4)
        Returns:
            policy_logits: (Batch, 4)
            value: (Batch, 1) in range [-1, 1]
        """
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value
