import torch.nn as nn
import torch
from typing import Tuple

def set_device():
    """Set the device for PyTorch operations"""
    # Check if CUDA is available and set the device accordingly.
    # If not, default to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class DQN(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], action_dim: int):
        """Initialize DQN with convolutional layers"""
        # Set up the parent class and define three convolutional layers.
        # Use progressively smaller kernels and increasing channels.
        # Calculate the flattened size after convolutions for fully connected layers.
        # Add two fully connected layers to map features to Q-values.

        super(DQN, self).__init__()

        self.device = set_device()

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

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate output size of convolutional layers"""
        # Pass a dummy tensor through the convolutional layers.
        # Calculate the total number of features after flattening.

        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.conv(dummy)

            return int(torch.prod(torch.tensor(out.shape[1:])))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Apply convolutional layers with ReLU activations.
        # Flatten the feature maps for fully connected processing.
        # Apply fully connected layers and return raw Q-values.

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
