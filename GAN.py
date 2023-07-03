import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1), #fake or real one output
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.disc(x)
    

