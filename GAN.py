import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
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
    
#hyperparameter

device ="cuda" if torch.cuda.is_available() else "cpu"
print(device)

lr = 3e-4
z_dim = 64
image_dims = 28*28*1
batch_size = 32
num_epochs = 50


disc = Discriminator(image_dims).to(device)
gen = Generator(z_dim,image_dims).to(device)

fixed_noise = torch.randn((batch_size,z_dim)).to(device)

transforms = transforms.Compose(
    #applying series of transformations
    [transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    ## Tranforming the images to tensors 
    # This transformation normalizes the tensor by subtracting the mean and dividing by the standard deviation.
    # The first argument (0.1307,) represents the mean value for each channel in MNIST,
    #  and the second argument (0.3081,) represents the standard deviation for each channel.

dataset = datasets.MNIST(root="dataset/",transform=transforms,download=True)
loader = DataLoader(dataset,batch_size,shuffle=True)
opt_disc = optim.Adam(disc.parameters(),lr=lr)

opt_gen = optim.Adam(gen.parameters(),lr=lr)
criterion = nn.BCELoss # binary cross entropy 

writer_fake = SummaryWriter(f"run/GAN_MNIST/fake")
writer_real = SummaryWriter(f"run/GAN_MNIST/real")
step =0

for epoch in range(num_epochs):
    for batch_idx  ,(real,_) in enumerate(loader):
        real = real. view (-1, 784).to(device) #converting the shape to a 2D one 
        #(-1) represents an inferred batch size  
        #784 - flattened size of image
        batch_size = real.shape[0]


