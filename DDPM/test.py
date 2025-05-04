from DDPM import Diffusion, Denoiser, show_image
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

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

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    
if __name__ == "__main__":
    model = Denoiser(image_resolution=img_size,
                 hidden_dims=hidden_dims,
                 diffusion_time_embedding_dim=timestep_embedding_dim,
                 n_times=n_timesteps
                 ).to(DEVICE)

    
    diffusion = Diffusion(model, image_resolution=img_size, n_times=n_timesteps, beta_minmax=beta_minmax, device=DEVICE).to(DEVICE)
    load_checkpoint(diffusion, "/home/nkd/ouyangzl/Diffusion Model/checkpoint/DDPM_199.pth")
    print("Model Loaded !")
    image = diffusion.sample(inference_batch_size)
    
    test_image_dir = "/home/nkd/ouyangzl/Diffusion Model/images/test_image/"
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir, exist_ok=True)
    print("Dir for saving images is made !")
    N = image.shape[0]
    for index in range(N):
        figure = plt.figure()
        image_path = test_image_dir + f"image_{index}"
        plt.imshow(image[index].transpose(0, 1).transpose(1, 2).detach().cpu().numpy())
        plt.show()
        plt.savefig(image_path)
        print("Image is saved !")
        plt.close()
        
    
    