import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # 14x14
        self.bn1 = nn.BatchNorm2d(32)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1) # 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.conv = nn.Conv2d(16, 1, kernel_size=3, padding=1) 

    def forward(self, x):
        x = F.relu(self.lin1(x)).view(-1, 64, 7, 7)
        x = F.relu(self.bn1(self.ct1(x)))
        x = F.relu(self.bn2(self.ct2(x)))
        return torch.tanh(self.conv(x)) # Tanh output is standard for GANs

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64*7*7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc1(x))
