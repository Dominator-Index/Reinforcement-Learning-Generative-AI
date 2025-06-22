import torch
from sklearn.datasets import make_moons, make_swiss_roll, make_circles

class SyntheticData:
    def __init__(self, dataset):
        self.dataset = dataset

        dataset_to_method = {
            'checkerboard': sample_checkerboard,
            '2moons': sample_2moons,
            'swissroll': sample_swissroll,
            'circles': sample_circles
        }
        if dataset in dataset_to_method:
            self.sample_target = dataset_to_method[dataset]
        else:
            raise NotImplementedError(f'Dataset {dataset} not recognized')
                
    def sample_source(self, n_samples, noise='gaussian'):
        # Assumed to be in (u, y) format
        samples = self.sample_target(n_samples)

        if noise == 'gaussian':
            # p(u) = N(0, I)
            samples[:, 0] = torch.randn((n_samples,))
        elif noise == 'independent':
            # p(u) = q(u) is the same u-marginal as the target
            shuffle = torch.randperm(n_samples)
            samples[:, 0] = samples[shuffle, 0]
        else:
            raise NotImplementedError(f'Noise {noise} not recognized')

        return samples
    

def sample_checkerboard(n_samples):
    # Sample x ~ Unif[-1, 1]
    x = 2 * torch.rand(n_samples) - 1
    
    # Determine which of the two patterns x is in
    pattern1 = (x <= -0.5) + (x >= 0) * (x<= 0.5)  # [-1, -0.5] OR [0, 0.5]
    pattern2 = torch.logical_not(pattern1)         # [-0.5, 0]  OR [0.5, 1.0]
    
    # Inside of each pattern, determine which sub-square to sample from
    p1_mask = torch.randint(2, [pattern1.sum()])
    p2_mask = torch.randint(2, [pattern2.sum()])
    
    # Sample y values
    y_p1_1 = 0.5 * torch.rand(p1_mask.sum())
    y_p1_2 = 0.5 * torch.rand((1 - p1_mask).sum()) - 1.
    y_p1 = torch.cat([y_p1_1, y_p1_2])
    
    y_p2_1 = 0.5 * torch.rand(p2_mask.sum()) + 0.5
    y_p2_2 = 0.5 * torch.rand((1 - p2_mask).sum()) - 0.5
    y_p2 = torch.cat([y_p2_1, y_p2_2])
    
    
    out = torch.empty([n_samples, 2])
    out[:, 0] = x
    out[pattern1, 1] = y_p1
    out[pattern2, 1] = y_p2
    
    return out

def sample_2moons(n_samples):
    out = make_moons(n_samples=n_samples, shuffle=True, noise=0.05)[0]
    out = torch.tensor(out, dtype=torch.float32)
    out = (out - torch.tensor([0.5, .25])) / torch.sqrt(torch.tensor([0.75, 0.25]))
    return out

def sample_circles(n_samples):
    out = make_circles(n_samples=n_samples, shuffle=True, factor=0.5, noise=0.05)[0]
    out = torch.tensor(out, dtype=torch.float32)
    return out

def sample_swissroll(n_samples):
    out = make_swiss_roll(n_samples=n_samples, noise=0.75)[0]
    out = out[:, [0, 2]]
    out = torch.tensor(out, dtype=torch.float32)
    scale = 12
    out = out / scale
    return out