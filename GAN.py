import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, image_dims) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dims,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1), #fake or real one output
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim,image_dims) -> None: #z-dimension -> dimension of latent noise that the generator takes as the input
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,image_dims),
            nn.Tanh(), # we use tanh here to make the output of the pixel values are b/w -1 and 1 
        )
    
    def forward(self,x):
        return self.gen(x)
    