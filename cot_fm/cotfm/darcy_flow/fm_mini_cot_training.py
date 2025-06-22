import torch
import torch.optim as optim
import wandb
import argparse
import sys
sys.path.append('../')

from models.fourier_neural_operator import TimeConditionalFNO
from src.si import SI
from src.fm import trainer
# from src.util.dataloaders import get_darcy_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    # parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--num_epochs", type=int, default=1500, help="Number of epochs for training (default: 1500)")
    # parser.add_argument("--y_noise_level", type=float, default=0.025, help="Noise level of y observed (default: 0.025)")
    # parser.add_argument("--n_train", type=int, default=10000, help="Number of training data points (default: 20000)")
    # parser.add_argument("--n_test", type=int, default=5000, help="Number of test data points (default: 5000)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Adam optimizer (default: 5e-4)")
    parser.add_argument("--modes", type=int, default=32, help="Modes for FNO (default: 32)")
    parser.add_argument("--vis_channels", type=int, default=1, help="Visible channels (default: 1)")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels (default: 64)")
    parser.add_argument("--proj_channels", type=int, default=256, help="Projection channels (default: 256)")
    parser.add_argument("--x_dim", type=int, default=2, help="Dimension of input data (default: 2)")
    parser.add_argument("--minibatch_ot", type=bool, default=False, help="Use minibatch OT (default: False)")
    parser.add_argument("--gp_length_scale", type=float, default=0.5, help="Gaussian process kernel length scale (default: 0.01)")
    parser.add_argument("--gp_kernel_variance", type=float, default=1., help="Gaussian process kernel variance (default: 1.)")
    parser.add_argument("--gamma_type", type=str, default="constant", help="Type of gamma (default: constant)")
    parser.add_argument("--wandb_project_name", type=str, default="minibatch-FFM", help="Weights & Biases project name (default: COT-FM)")
    return parser.parse_args()

class ExperimentConfig:
    def __init__(self, args):
        self.configs = vars(args)

    def load_data(self):
        # return get_darcy_dataloader(
        #     batch_size=self.configs["batch_size"],
        #     n_train=self.configs["n_train"],
        #     n_test=self.configs["n_test"],
        #     noise_level_y_observed=self.configs["y_noise_level"],
        #     path_prefix = '../data',
        # )
        return torch.load('../data/dataloader.pt')

    def initialize_model(self):
        fno = TimeConditionalFNO(
            modes=self.configs["modes"],
            vis_channels=self.configs["vis_channels"],
            hidden_channels=self.configs["hidden_channels"],
            proj_channels=self.configs["proj_channels"],
            x_dim=self.configs["x_dim"],
        ).to('cuda')
        si = SI(model=fno, functional=True, device='cuda', 
                gamma_type=self.configs["gamma_type"],
                kernel_length=self.configs["gp_length_scale"],
                kernel_variance=self.configs["gp_kernel_variance"],)
        return si

    def initialize_optimizer(self, si):
        optimizer = optim.Adam(si.model.parameters(), lr=self.configs["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs["num_epochs"], eta_min=1e-5)
        return optimizer, scheduler

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
    dataloader = config.load_data()

    # Initialize models
    si = config.initialize_model()

    # Initialize Optimizers
    optimizer, scheduler = config.initialize_optimizer(si)

    # Initialize wandb run for logging and experiment tracking
    run = config.initialize_wandb_run()

    # Train model
    trainer(
        model = si,
        optimizer = optimizer,
        scheduler = scheduler,
        train_loader = dataloader,
        max_epochs = args.num_epochs,
        minibatch_cot = True,
        wandb_run = run,
        checkpoint_every = 100,
    )
    torch.save(si.model.state_dict(), f'../darcy_flow/trained_models/cot_ffm/si_{run.id}_final.pth')

    run.finish()

if __name__ == "__main__":
    main()