import torch.nn as nn
import torch
from typing import Tuple
import math
from config import config

def set_device():
    """Set the device for PyTorch operations"""
    # Check if CUDA is available and set the device accordingly.
    # If not, default to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=config['sigma_init']):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class DQN(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], action_dim: int):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = input_dim

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out_size(input_dim)

        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, action_dim)
        )

    def _get_conv_out_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.conv(dummy)
            return int(torch.prod(torch.tensor(out.shape[1:])))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()