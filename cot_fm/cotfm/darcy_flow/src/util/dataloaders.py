import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from src.util.gaussian_process import make_grid
from src.util.ot import emd_map, sinkhorn_map, hungarian_matching

class OTSampler(torch.utils.data.Sampler):
    def __init__(self, ot_matrix, replacement=False, shuffle=True):
        self.ot_matrix = ot_matrix
        self.replacement = replacement
        self.shuffle = shuffle

    def __iter__(self):
        indices_0 = torch.arange(self.ot_matrix.shape[0])
        indices_1 = torch.multinomial(self.ot_matrix, 1, replacement=self.replacement).squeeze()
        indices = torch.stack((indices_0, indices_1), dim=1)
        if self.shuffle:
            indices = indices[torch.randperm(indices.shape[0])]
        return iter(indices.tolist())
        
    def __len__(self):
        return self.ot_matrix.shape[0]

class MultiDataset(Dataset):
    '''
    Combines two datasets. Used for building interpolant from one dataset to another.
    '''

    def __init__(self, dataset0, dataset1):
        self.dataset0 = dataset0
        self.dataset1 = dataset1

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        index0, index1 = idx
        return self.dataset0[index0], self.dataset1[index1]
        
class CombinedDataset(Dataset):
    '''
    Combines two datasets of the same length into one. Used for building interpolant from one dataset to another.
    '''

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(dataset1) == len(dataset2), "Datasets must have the same length."

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]
    
class Normalizer():
    def __init__(self,data,eps=1e-5):
        self.eps=eps
        self.mean=np.mean(data,0,keepdims=True)
        self.std=np.std(data,0,keepdims=True)
    def transform(self,data):
        return (data-self.mean)/(self.std+self.eps)
    
    def inverse_transform(self,data):
        return data*(self.std+self.eps)+self.mean


def matern_three_half(rho):
    def k(d):
        return (1+np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho)
    return k
    

def sample_log_permeability_fields(
        gp,
        num_samples,
        ):
    sample_dim = (40,40)
    query_points = make_grid([*sample_dim])
    fields = gp.sample(query_points, dims=sample_dim, n_samples=num_samples).float()
    print(fields.shape)
    return fields

def conditional_optimal_transport(dataset0, dataset1, eps=1e-8, u_flat_size=40*40, y_flat_size=100*100,
                                  y_cost = (lambda x, y: torch.cdist(x, y, p=2,)**2), save_matrix_path=None, 
                                  method='emd', reg=1e-1, kmax=1e6):
    
    u_0s = dataset0.tensors[1].flatten(start_dim=1)
    u_1s = dataset1.tensors[1].flatten(start_dim=1)
    y_0s = dataset0.tensors[0].flatten(start_dim=1)
    y_1s = dataset1.tensors[0].flatten(start_dim=1)

    C_y = y_cost(y_0s, y_1s)
    C_u = torch.cdist(u_0s, u_1s, p=2,)**2

    C_u = (C_u - C_u.min())/(C_u.max() - C_u.min())
    C_y = (C_y - C_y.min())/(C_y.max() - C_y.min())

    C = eps*C_u + C_y
    print('Computing optimal transport...')

    if method == 'hungarian':
        return hungarian_matching(C)
    elif method == 'emd':
        otmatrix = emd_map(C)
    elif method == 'sinkhorn':
        otmatrix = sinkhorn_map(C, reg=reg, kmax=kmax)
    else:
        raise ValueError('Unknown OT method')
    if save_matrix_path is not None:
        torch.save(otmatrix, save_matrix_path)
    otmatrix[otmatrix.sum(1) == 0] = 1/otmatrix.shape[1] # fix rows that sum to zero by substituting them with a uniform distribution
    otmatrix = otmatrix / otmatrix.sum(1, keepdim=True) # turn it into a conditional distribution
    return otmatrix

def optimal_transport(dataset0, dataset1, save_matrix_path=None,
                      method='emd', reg=1e-1, kmax=1e6):
    u_0s = dataset0.tensors[1].flatten(start_dim=1)
    u_1s = dataset1.tensors[1].flatten(start_dim=1)

    C_u = torch.cdist(u_0s, u_1s, p=2,)**2

    C = (C_u - C_u.min())/(C_u.max() - C_u.min())

    if method == 'hungarian':
        return hungarian_matching(C)
    elif method == 'emd':
        otmatrix = emd_map(C)
    elif method == 'sinkhorn':
        otmatrix = sinkhorn_map(C, reg=reg, kmax=kmax)
    else:
        raise ValueError('Unknown OT method')
    if save_matrix_path is not None:
        torch.save(otmatrix, save_matrix_path)
    otmatrix[otmatrix.sum(1) == 0] = 1/otmatrix.shape[1] # fix rows that sum to zero by substituting them with a uniform distribution
    otmatrix = otmatrix / otmatrix.sum(1, keepdim=True) # turn it into a conditional distribution
    return otmatrix


def get_darcy_dataloader(
        batch_size=128,
        noise_level_y_observed=0.025,
        n_train = 20000,
        n_test = 5000,
        shuffle = False,
        path_prefix = '../data',
        coupling = 'none', # 'ot', 'cot', 'none'
        n_test_sets = 5,
        prod_measure = False
        ):
    y_observed=np.load(path_prefix + "/X_observed.npy")
    u_fields=np.log(np.load(path_prefix + "/true_permeability_fields.npy"))
    
    if len(u_fields) < n_train:
        n_train = len(u_fields)
        print("Number of datapoints is larger than the number of available samples. Using all available samples.")
    # Add a uniform noise first before centering/normalizing
    y_observed = y_observed + noise_level_y_observed*np.random.standard_normal(size = y_observed.shape)
    y_Normalizer=Normalizer(y_observed)

    y_normalized = y_Normalizer.transform(y_observed)
    y_data=torch.tensor(y_normalized,dtype=torch.float)
    # How much of the variance is explained by the noise?
    print("Rough Ratio SDnoise/SDsignal:", np.mean(noise_level_y_observed/y_Normalizer.std))

    # set seed
    np.random.seed(0)
    source_idx = np.random.choice(len(y_data), n_train)
    target_idx = np.random.choice(np.setdiff1d(np.arange(len(y_data)), source_idx), n_train)
    test_indices = []
    for i in range(n_test_sets):
        if i == 0:
            exclude = np.concatenate([source_idx, target_idx])
        else:
            exclude = np.concatenate([source_idx, target_idx, np.concatenate(test_indices)])
        test_indices.append(np.random.choice(np.setdiff1d(np.arange(len(y_data)), exclude), n_test))

    y_data_0 = torch.tensor(y_data[source_idx]).unsqueeze(1).float()

    # For COT, we want the source to be the product measure. Hence, we pick different indices for U
    if prod_measure:
        exclude = np.concatenate([source_idx, target_idx, np.concatenate(test_indices)])
        source_idx_u = np.random.choice(np.setdiff1d(np.arange(len(y_data)), exclude), n_train)
        u_fields_0 = torch.tensor(u_fields[source_idx_u]).unsqueeze(1).float()
    else:
        u_fields_0 = torch.tensor(u_fields[source_idx]).unsqueeze(1).float()
    
    y_data_1 = torch.tensor(y_data[target_idx]).unsqueeze(1).float()
    u_fields_1 = torch.tensor(u_fields[target_idx]).unsqueeze(1).float()

    test_loaders = []
    for test_idx in test_indices:
        y_data_test = torch.tensor(y_data[test_idx]).unsqueeze(1).float()
        u_fields_test = torch.tensor(u_fields[test_idx]).unsqueeze(1).float()
        test_dataset = TensorDataset(y_data_test, u_fields_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loaders.append(test_loader)

    target_dataset = TensorDataset(y_data_1, u_fields_1)
    source_dataset = TensorDataset(y_data_0, u_fields_0)

    # Compute full OT couplings
    if coupling == 'cot':
        ot_matrix = conditional_optimal_transport(source_dataset, target_dataset, save_matrix_path=None)
        ot_loader = DataLoader(MultiDataset(source_dataset, target_dataset), 
                            batch_size=batch_size, 
                            sampler=OTSampler(ot_matrix, shuffle=shuffle),
                            num_workers=2, pin_memory=True)
    elif coupling == 'ot':
        ot_matrix = optimal_transport(source_dataset, target_dataset, save_matrix_path=None)
        ot_loader = DataLoader(MultiDataset(source_dataset, target_dataset),
                            batch_size=batch_size, 
                            sampler=OTSampler(ot_matrix, shuffle=shuffle),
                            num_workers=2, pin_memory=True)
    else:
        # For minibatch methods, we use this dataset. Couplings will be computed subsequently.
        ot_loader = DataLoader(CombinedDataset(source_dataset, target_dataset), 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=2, pin_memory=True)
    return ot_loader, test_loaders, y_Normalizer
