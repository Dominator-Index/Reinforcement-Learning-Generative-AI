import sys
sys.path.append('../')

import numpy as np
import data.darcySolver as ds
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
from src.util.dataloaders import get_darcy_dataloader
import torch

rng=np.random.default_rng(seed=101)

x_grid=np.linspace(0,1,100)
y_grid=np.linspace(0,1,100)
X,Y=np.meshgrid(x_grid,y_grid)

def matern_one_half(rho):
    def k(d):
        return np.exp(-d/rho)
    return k

def matern_three_half(rho):
    def k(d):
        return (1+np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho)
    return k
    
def matern_five_half(rho):
    def k(d):
        return (1+np.sqrt(5)*d/rho+np.sqrt(5)*d**2/(rho**2))*np.exp(-np.sqrt(5)*d/rho)
    return k


def sample_permeability_fields(
        kernel_function_d,
        num_samples,
        rank=50,
        kernel_points=25,
        scaling=1
        ):
    kernel_x_grid=np.linspace(0,1,kernel_points)
    kernel_y_grid=np.linspace(0,1,kernel_points)


    points=np.array([[x,y] for x in kernel_x_grid for y in kernel_y_grid])
    D=np.array([np.linalg.norm(p1-p2) for p1 in points for p2 in points]).reshape(kernel_points**2,kernel_points**2)

    KMat=kernel_function_d(D)
    eig,P=np.linalg.eigh(KMat)
    #Set them in decreasing order for convenience
    eig=eig[::-1]
    P=P[:,::-1]
    eig_reduced=eig.copy()
    eig_reduced[rank:]=0


    gp_samples=rng.multivariate_normal(np.zeros(len(D)),cov=KMat,size=num_samples)
    fields=[np.exp(scaling*gp_sample).reshape(kernel_points,kernel_points) for gp_sample in gp_samples]

    return kernel_x_grid,kernel_y_grid,fields,KMat

# def sample_permeability_fields(
    #     gp,
    #     num_samples,
    #     kernel_points=25,
    #     ):
    # '''
    # Samples from a Gaussian Process prior using the GPyTorch library
    # '''
    # kernel_x_grid=np.linspace(0,1,kernel_points)
    # kernel_y_grid=np.linspace(0,1,kernel_points)

    # sample_dim = (kernel_points,kernel_points)
    # query_points = make_grid([*sample_dim])
    # log_p_fields = gp.sample(query_points, dims=sample_dim, n_samples=num_samples).float().squeeze().numpy()
    # fields=[np.exp(gp_sample).reshape(kernel_points,kernel_points) for gp_sample in log_p_fields]

    # return kernel_x_grid,kernel_y_grid,fields

def constant_rhs(x):
    return x[0]*0+1

def make_datapoint(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        num_eval=100,
        f_rhs=constant_rhs
        ):

    interpolated_permeability_field=RectBivariateSpline(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        kx=2,
        ky=2
        )

    permeability_function=lambda x:interpolated_permeability_field(x[0],x[1],grid=False)

    num_cells=100
    darcy=ds.DarcySolver(num_x_cells=num_cells,num_y_cells=num_cells)


    sol=darcy.solve(
        permeability=permeability_function,
        f_rhs=f_rhs
        )
    
    x_obs=np.linspace(0,1,num_eval)
    y_obs=np.linspace(0,1,num_eval)
    Xobs,Yobs=np.meshgrid(x_obs,y_obs)
    sol_observed=sol(Xobs,Yobs,grid=False)
    return sol_observed

params="""
n=100000
num_kernel_points=40
kernel = matern_three_half(0.5)
eval_points = 100
"""

n=100000
num_kernel_points=40
print("Sampling Random Fields")

### Using GPytorch
# nu = 1.5
# kernel_length = 0.5
# kernel_variance = 1.

# base_kernel = gpytorch.kernels.MaternKernel(nu,eps=1e-10)
# base_kernel.lengthscale = kernel_length
# covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
# covar_module.outputscale = kernel_variance
# gp = GPPrior(covar_module)

# xgrid, ygrid, permeability_fields = sample_permeability_fields(
#     gp,
#     n,
#     kernel_points=num_kernel_points
#     )

### Using original method - what we use in the paper
xgrid,ygrid,permeability_fields,K=sample_permeability_fields(
    matern_three_half(0.5),
    n,
    kernel_points=num_kernel_points
    )
print("Finished Sampling")

print("Solving Darcy Flow on each Field")
observations=[
    make_datapoint(
    xgrid,
    ygrid,
    field,
    num_eval=100
    ) for field in tqdm(permeability_fields)
]

X_data_observed=np.array(observations)
print("Finished Solving")
print("Saving Results")

with open('data/params.txt', 'w') as f:
    f.write(params)

np.save("data/X_observed.npy",X_data_observed)
np.save("data/true_permeability_fields.npy",np.array(permeability_fields))

dataloader, test_loaders, Y_transform = get_darcy_dataloader(
    batch_size=128,
    n_train=10000,
    n_test=5000,
    noise_level_y_observed=0.025,
    path_prefix = '../data/new',
    shuffle=True,
    coupling='none',
    prod_measure=True
    )

torch.save(dataloader,'data/dataloader.pt')
for i,test_loader in enumerate(test_loaders):
    torch.save(test_loader,f'data/test_loader_{i}.pt')

# pre-compute the U_0 samples for evaluation
rng=np.random.default_rng(seed=105)
n=10*5000*5

_,_,permeability_fields,_=sample_permeability_fields(
    matern_three_half(0.5),
    n,
    kernel_points=num_kernel_points
    )
gp_tensor = torch.tensor(permeability_fields).unsqueeze(1).log().float()
gp_dataset = torch.utils.data.TensorDataset(gp_tensor)
torch.save(gp_dataset, '../data/gp_dataset.pt')
