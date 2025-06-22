import torch
import torch.optim as optim
import wandb
import argparse
import sys
sys.path.append('../')

from models.fourier_neural_operator import ConditionalFNO, DiscriminatorFNO
from src.gan import Trainer
# from src.util.dataloaders import get_darcy_dataloader
from src.util.gaussian_process import GPPrior
import gpytorch

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    # parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=1500, help="Number of epochs for training (default: 1500)")
    # parser.add_argument("--y_noise_level", type=float, default=0.025, help="Noise level of y observed (default: 0.025)")
    # parser.add_argument("--n_train", type=int, default=10000, help="Number of training data points (default: 20000)")
    # parser.add_argument("--n_test", type=int, default=5000, help="Number of test data points (default: 5000)")
    parser.add_argument("--monotone_penalty", type=float, default=0.05, help="Monotone penalty (default: 0.05)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam optimizer (default: 2e-4)")
    parser.add_argument("--modes", type=int, default=32, help="Modes for FNO (default: 32)")
    parser.add_argument("--vis_channels", type=int, default=1, help="Visible channels (default: 1)")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels (default: 64)")
    parser.add_argument("--proj_channels", type=int, default=256, help="Projection channels (default: 256)")
    parser.add_argument("--x_dim", type=int, default=2, help="Dimension of input data (default: 2)")
    parser.add_argument("--gp_weight", type=float, default=5, help="Gradient penalty weight (default: 5)")
    parser.add_argument("--full_critic_train", type=int, default=10, help="How often to train the critic (default: 10)")
    parser.add_argument("--ensemble_size", type=int, default=10, help="Number of ensemble samples for evaluation (default: 10)")
    parser.add_argument("--wandb_project_name", type=str, default="WaMGAN", help="Weights & Biases project name (default: WaMGAN)")
    return parser.parse_args()

class ExperimentConfig:
    def __init__(self, args):
        self.configs = vars(args)

    def load_data(self):
        nu = 1.5
        kernel_length = 0.5
        kernel_variance = 1.

        base_kernel = gpytorch.kernels.MaternKernel(nu,eps=1e-10)
        base_kernel.lengthscale = kernel_length
        covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        covar_module.outputscale = kernel_variance
        gp = GPPrior(covar_module)
        # return gp, get_darcy_dataloader(
        #     batch_size=self.configs["batch_size"],
        #     n_train=self.configs["n_train"],
        #     n_test=self.configs["n_test"],
        #     noise_level_y_observed=self.configs["y_noise_level"],
        #     path_prefix = '../data',
        # )
        return gp, torch.load('../data/dataloader.pt')

    def initialize_models(self):
        generator = ConditionalFNO(
            modes=self.configs["modes"],
            vis_channels=self.configs["vis_channels"],
            hidden_channels=self.configs["hidden_channels"],
            proj_channels=self.configs["proj_channels"],
            x_dim=self.configs["x_dim"],
        ).to('cuda')

        discriminator = DiscriminatorFNO(
            modes=self.configs["modes"],
            vis_channels=self.configs["vis_channels"],
            hidden_channels=self.configs["hidden_channels"],
            proj_channels=self.configs["proj_channels"],
            x_dim=self.configs["x_dim"],
        ).to('cuda')

        return generator, discriminator

    def initialize_optimizers(self, generator, discriminator):
        G_optimizer = optim.Adam(generator.parameters(), lr=self.configs["lr"])
        D_optimizer = optim.Adam(discriminator.parameters(), lr=self.configs["lr"])
        return G_optimizer, D_optimizer

    def initialize_wandb_run(self):
        wandb.login()
        run = wandb.init(
            project=self.configs["wandb_project_name"],
            config=self.configs,
        )
        return run

def main():
    
    args = parse_args()
    config = ExperimentConfig(args)

    # Load Data
    gp, dataloader = config.load_data()

    # Initialize models
    generator, discriminator = config.initialize_models()

    # Initialize Optimizers
    G_optimizer, D_optimizer = config.initialize_optimizers(generator, discriminator)

    # Initialize wandb run for logging and experiment tracking
    run = config.initialize_wandb_run()

    # Train model
    trainer = Trainer(
        prior=gp,  
        generator=generator, 
        discriminator=discriminator, 
        gen_optimizer=G_optimizer, 
        dis_optimizer=D_optimizer,
        wandb_run=run,
        device='cuda',
        print_every=50,
        gp_weight=config.configs["gp_weight"],
        monotone_penalty=config.configs["monotone_penalty"],
        penalty_type="monge",
        gradient_penalty_type="two_sided",
        full_critic_train=config.configs["full_critic_train"],
        checkpoint_every=100,
    )

    trainer.train(dataloader, config.configs["num_epochs"])
    torch.save(generator.state_dict(), f'../darcy_flow/trained_models/wamgan/generator_{run.id}.pth')
    torch.save(discriminator.state_dict(), f'../darcy_flow/trained_models/wamgan/discriminator_{run.id}.pth')

    run.finish()
    
if __name__ == "__main__":
    main()