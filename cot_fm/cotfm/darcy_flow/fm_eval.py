import numpy as np
import torch
import torch.optim as optim
import wandb
import argparse
import sys
sys.path.append('../')

from models.fourier_neural_operator import TimeConditionalFNO
from src.si import SI
from src.fm import trainer
from src.util.dataloaders import get_darcy_dataloader, CombinedDataset
# from src.util.darcy_utils import make_datapoint
from src.util.gaussian_process import GPPrior, make_grid
from tqdm import tqdm

import gpytorch
from models.fourier_neural_operator import ConditionalFNO, DiscriminatorFNO
from models.fourier_neural_operator import TimeConditionalFNO
import properscoring as ps
from pysteps.verification.probscores import CRPS as pysteps_crps


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of Darcy Flow model")
    parser.add_argument("--cot_ffm_run_id", type=str, default="g2yfmc5g", help="wandb run id for cot_ffm")
    parser.add_argument("--ffm_run_id", type=str, default="v2il8rng", help="wandb run id for ffm")
    parser.add_argument("--gan_run_id", type=str, default="ar0x5v7c_700", help="wandb run id for gan")
    parser.add_argument("--m", type=int, default=10, help="Number of ensemble samples for evaluation (default: 10)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for eval (default: 5)")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples for evaluation (default: 5000)")
    return parser.parse_args()

class ExperimentConfig:
    def __init__(self, args):
        self.configs = vars(args)

    def load_data(self,):
        test_loaders = []
        u_tests = []
        for i in range(5):
            test_loader = torch.load(f'../data/test_loader_{i}.pt')
            # rebuild test_loader so that it has the right batch size
            y_test = test_loader.dataset.tensors[0][:self.configs["n"]]
            u_test = test_loader.dataset.tensors[1][:self.configs["n"]]
            test_loader = torch.utils.data.DataLoader(
                CombinedDataset(y_test, u_test),
                batch_size=self.configs["batch_size"],
                shuffle=False,
            )
            test_loaders.append(test_loader)
            u_tests.append(u_test)

        gp_d = torch.load('../data/gp_dataset.pt')

        prior_loader = torch.utils.data.DataLoader(
            gp_d,
            batch_size=self.configs["batch_size"]*self.configs["m"],
            shuffle=False,
        )

        return prior_loader, test_loaders, u_tests

    def initialize_ffm_models(self):
        modes = 32
        vis_channels = 1
        hidden_channels = 64
        proj_channels = 256
        x_dim = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fno0 = TimeConditionalFNO(
            modes=modes,
            vis_channels=vis_channels,
            hidden_channels=hidden_channels,
            proj_channels=proj_channels,
            x_dim=x_dim,
        ).to(device)
        fno1 = TimeConditionalFNO(
            modes=modes,
            vis_channels=vis_channels,
            hidden_channels=hidden_channels,
            proj_channels=proj_channels,
            x_dim=x_dim,
        ).to(device)
        fno_dict0 = torch.load(f'../darcy_flow/trained_models/cot_ffm/si_{self.configs["cot_ffm_run_id"]}_final.pth')
        fno_dict1 = torch.load(f'../darcy_flow/trained_models/ffm/si_{self.configs["ffm_run_id"]}_final.pth')   
        fno0.load_state_dict(fno_dict0)
        fno1.load_state_dict(fno_dict1)
        model0 = SI(model=fno0,
                functional=True,
                kernel_length=0.5)
        model1 = SI(model=fno1,
                kernel_length=0.5, 
                functional=True)
        return model0, model1

    def initialize_gan_model(self):
        modes = 32
        vis_channels = 1
        hidden_channels = 64
        proj_channels = 256
        x_dim = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G0 = ConditionalFNO(
            modes=modes,
            vis_channels=vis_channels,
            hidden_channels=hidden_channels,
            proj_channels=proj_channels,
            x_dim=x_dim,
        ).to(device)
        gen_dict = torch.load(f'../darcy_flow/trained_models/wamgan/gen_{self.configs["gan_run_id"]}.pth')
        G0.load_state_dict(gen_dict)
        return G0

def main():
    @torch.no_grad()
    def batched_sampling(test_loader, prior_loader, model, m, algorithm='si'):
        def sample_si(y0, prior_samples, si, m):
            n = len(y0) 
            if m>1:
                y0 = y0.repeat(1,m,1,1).flatten(0,1).unsqueeze(1)
            gen_samples = si.sample(prior_samples,y0, device='cuda', rtol=1e-5, atol=1e-5) # shape (n*m, 1, 40, 40)
            # group by m
            gen_samples = gen_samples.view(n,m,*gen_samples.shape[1:]) # shape (n, m, 1, 40, 40)
            return gen_samples

        def sample_gan(y0, prior_samples, generator, m):
            n = len(y0) 
            if m>1:
                y0 = y0.repeat(1,m,1,1).flatten(0,1).unsqueeze(1)
            gen_samples = generator(prior_samples.to('cuda'), y0.to('cuda'))
            # group by m
            gen_samples = gen_samples.view(n,m,*gen_samples.shape[1:])
            return gen_samples

        if algorithm == 'si':
            sample = sample_si
        elif algorithm == 'gan':
            sample = sample_gan
            
        for i, ((y, _),[prior_samples]) in tqdm(enumerate(zip(test_loader, prior_loader)), total=len(test_loader)):
            if i == 0:
                gen_samples = sample(y, prior_samples, model, m)
            else:
                gen_samples = torch.cat([gen_samples, sample(y, prior_samples, model, m)], dim=0)
        return gen_samples
    
    def compute_mse(gen_samples, true_samples):
        # gen_samples: (n, m, 1, 40, 40)
        # true_samples: (n, 1, 40, 40)
        m = gen_samples.shape[1]
        m_times_true_samples = true_samples.repeat(1,m,1,1).flatten(0,1).unsqueeze(1) # shape (n*m, 1, 40, 40)
        mse = torch.nn.functional.mse_loss(gen_samples.flatten(0,1), m_times_true_samples) # reshaping gen_samples to (n*m, 1, 40, 40)
        return mse
        
    def compute_ensemble_mse(gen_samples, true_samples):
        # gen_samples: (n, m, 1, 40, 40)
        # true_samples: (n, 1, 40, 40)
        gen_ensemble_avg = gen_samples.mean(dim=1) # shape (n, 1, 40, 40)
        ensemble_mse = torch.nn.functional.mse_loss(gen_ensemble_avg, true_samples) 
        return ensemble_mse
    
    def compute_crps(gen_samples, true_samples):
        batch_size = gen_samples.shape[0]
        
        gen_samples = gen_samples.squeeze().numpy()   # (b, n, h, w)
        true_samples = true_samples.squeeze().numpy() # (b, h, w)
        
        crps = []
        for i in tqdm(range(batch_size)):
            gen_samples_i = gen_samples[i]
            true_samples_i = true_samples[i]
            
            _crps = pysteps_crps(gen_samples_i, true_samples_i)  # Expects generated (n, h, w) and true (h, w)
            crps.append(_crps)
            
        avg_crps = np.mean(crps)
        crps_std = np.std(crps)
        return avg_crps, crps_std
        
    args = parse_args()
    config = ExperimentConfig(args)
    
    # Load Data
    prior_loader, test_loaders, u_true = config.load_data()

    # Load model
    cot_ffm, ffm = config.initialize_ffm_models()
    gan = config.initialize_gan_model()

    print(f'Data and model loaded. There are {len(test_loaders[0].dataset)} samples in the test set.')

    # create a list for each stat of each model
    cot_ffm_mse_list = []
    cot_ffm_ensemble_mse_list = []
    cot_ffm_crps_list = []

    ffm_mse_list = []
    ffm_ensemble_mse_list = []
    ffm_crps_list = []

    gan_mse_list = []
    gan_ensemble_mse_list = []
    gan_crps_list = []

    for i, (test_loader, true_samples) in enumerate(zip(test_loaders, u_true)):
        cot_ffm_gen_samples = batched_sampling(test_loader, prior_loader, cot_ffm, m=args.m).cpu()
        torch.save(cot_ffm_gen_samples, f'../function_space_experiment/samples/cot_ffm/samples_id_{args.cot_ffm_run_id}_m_{args.m}_test_{i}.pt')
        cot_ffm_mse = compute_mse(cot_ffm_gen_samples, true_samples)
        print(f"Cot FFM MSE: {cot_ffm_mse}")
        cot_ffm_ensemble_mse = compute_ensemble_mse(cot_ffm_gen_samples, true_samples)
        print(f"Cot FFM Ensemble MSE: {cot_ffm_ensemble_mse}")
        cot_ffm_crps, cot_ffm_crps_std = compute_crps(cot_ffm_gen_samples, true_samples)
        print(f"Cot FFM CRPS: {cot_ffm_crps}")
        print(f"Cot FFM CRPS STD: {cot_ffm_crps_std}")

        cot_ffm_mse_list.append(cot_ffm_mse)
        cot_ffm_ensemble_mse_list.append(cot_ffm_ensemble_mse)
        cot_ffm_crps_list.append(cot_ffm_crps)

        ffm_gen_samples = batched_sampling(test_loader, prior_loader, ffm, m=args.m).cpu()
        torch.save(ffm_gen_samples, f'../function_space_experiment/samples/ffm/samples_id_{args.ffm_run_id}_m_{args.m}_test_{i}.pt')
        ffm_mse = compute_mse(ffm_gen_samples, true_samples)
        print(f"FFM MSE: {ffm_mse}")
        ffm_ensemble_mse = compute_ensemble_mse(ffm_gen_samples, true_samples)
        print(f"FFM Ensemble MSE: {ffm_ensemble_mse}")
        ffm_crps, ffm_crps_std = compute_crps(ffm_gen_samples, true_samples)
        print(f"FFM CRPS: {ffm_crps}")
        print(f"FFM CRPS STD: {ffm_crps_std}")

        ffm_mse_list.append(ffm_mse)
        ffm_ensemble_mse_list.append(ffm_ensemble_mse)
        ffm_crps_list.append(ffm_crps)

        gan_gen_samples = batched_sampling(test_loader, prior_loader, gan, m=args.m, algorithm='gan').cpu()
        torch.save(gan_gen_samples, f'../function_space_experiment/samples/gan/samples_id_{args.gan_run_id}_m_{args.m}_test_{i}.pt')
        gan_mse = compute_mse(gan_gen_samples, true_samples)
        print(f"GAN MSE: {gan_mse}")
        gan_ensemble_mse = compute_ensemble_mse(gan_gen_samples, true_samples)
        print(f"GAN Ensemble MSE: {gan_ensemble_mse}")
        gan_crps, gan_crps_std = compute_crps(gan_gen_samples, true_samples)
        print(f"GAN CRPS: {gan_crps}")
        print(f"GAN CRPS STD: {gan_crps_std}")

        gan_mse_list.append(gan_mse)
        gan_ensemble_mse_list.append(gan_ensemble_mse)
        gan_crps_list.append(gan_crps)

    # compute avg mse and std dev
    cot_ffm_mse = torch.tensor(cot_ffm_mse_list).mean().item()
    cot_ffm_ensemble_mse = torch.tensor(cot_ffm_ensemble_mse_list).mean().item()
    cot_ffm_crps = torch.tensor(cot_ffm_crps_list).mean().item()

    cot_ffm_mse_std = torch.tensor(cot_ffm_mse_list).std().item()
    cot_ffm_ensemble_mse_std = torch.tensor(cot_ffm_ensemble_mse_list).std().item()
    cot_ffm_crps_std = torch.tensor(cot_ffm_crps_list).std().item()

    ffm_mse = torch.tensor(ffm_mse_list).mean().item()
    ffm_ensemble_mse = torch.tensor(ffm_ensemble_mse_list).mean().item()
    ffm_crps = torch.tensor(ffm_crps_list).mean().item()

    ffm_mse_std = torch.tensor(ffm_mse_list).std().item()
    ffm_ensemble_mse_std = torch.tensor(ffm_ensemble_mse_list).std().item()
    ffm_crps_std = torch.tensor(ffm_crps_list).std().item()

    gan_mse = torch.tensor(gan_mse_list).mean().item()
    gan_ensemble_mse = torch.tensor(gan_ensemble_mse_list).mean().item()
    gan_crps = torch.tensor(gan_crps_list).mean().item()

    gan_mse_std = torch.tensor(gan_mse_list).std().item()
    gan_ensemble_mse_std = torch.tensor(gan_ensemble_mse_list).std().item()
    gan_crps_std = torch.tensor(gan_crps_list).std().item()

    # print results

    print(f'COT-FFM MSE: {cot_ffm_mse} +/- {cot_ffm_mse_std}')
    print(f'COT-FFM Ensemble MSE: {cot_ffm_ensemble_mse} +/- {cot_ffm_ensemble_mse_std}')
    print(f'COT-FFM CRPS: {cot_ffm_crps} +/- {cot_ffm_crps_std}')

    print(f'FFM MSE: {ffm_mse} +/- {ffm_mse_std}')
    print(f'FFM Ensemble MSE: {ffm_ensemble_mse} +/- {ffm_ensemble_mse_std}')
    print(f'FFM CRPS: {ffm_crps} +/- {ffm_crps_std}')

    print(f'GAN MSE: {gan_mse} +/- {gan_mse_std}')
    print(f'GAN Ensemble MSE: {gan_ensemble_mse} +/- {gan_ensemble_mse_std}')
    print(f'GAN CRPS: {gan_crps} +/- {gan_crps_std}')

    
if __name__ == "__main__":
    main()