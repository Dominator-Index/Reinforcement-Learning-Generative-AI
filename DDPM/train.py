from DDPM import Diffusion, Denoiser, show_image, save_model
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.optim as optim

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
dataset = 'MNIST'
img_size = (32, 32, 3) if dataset == "CFAR10" else (28, 28, 1)  # (width, height, channels)

timestep_embedding_dim = 256
n_layers = 8
hidden_dim = 256
n_timesteps = 1000
beta_minmax=[1e-4, 2e-2]

train_batch_size = 128
inference_batch_size = 1
lr = 5e-5
epochs = 200

seed = 1234

hidden_dims = [hidden_dim for _ in range(n_layers)]
torch.manual_seed(seed)
np.random.seed(seed)

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])
kwargs = {'num_workers': 1, 'pin_memory': True}

dataset_path = "/home/nkd/ouyangzl/Diffusion Model/data"

if dataset == 'CIFAR10':
    train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
else:
    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)
    
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))

   
if __name__ == "__main__":
    model = Denoiser(image_resolution=img_size,
                 hidden_dims=hidden_dims,
                 diffusion_time_embedding_dim=timestep_embedding_dim,
                 n_times=n_timesteps
                 ).to(DEVICE)

    
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps, beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)
    load_checkpoint(diffusion, "/home/nkd/ouyangzl/Diffusion Model/checkpoint/DDPM_77.pth")
    print("Model Loaded !")
    
    
    optimizer = optim.Adam(diffusion.parameters(), lr=lr)
    denoising_loss = nn.MSELoss()
    print("Start training DDPMs...")
    diffusion.train()

    epochs = epochs
    for epoch in range(epochs):
        noise_prediction_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            print('Gradient have been cleared')
            
            x = x.to(DEVICE)
            
            noise_input, epsilon, pred_epsilon = diffusion(x)
            loss = denoising_loss(pred_epsilon, epsilon)
            print(f'Loss: {loss}')
            
            noise_prediction_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            print('Parameters Optimized !')
        
        with torch.no_grad():
            generated_images = diffusion.sample(1)
            
            save_model(diffusion, epoch)
        show_image(generated_images, idx=0, epoch=epoch)
        diffusion.train()

        print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss: ", noise_prediction_loss / batch_idx)
        
    print("Finish!!")