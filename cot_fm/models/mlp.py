import torch

class TriangularMLP(torch.nn.Module):
    def __init__(self, condition_coords, u_dim, y_dim, w=64, layers=4, out_dim=None, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.y_coords = condition_coords
        self.u_coords = [i for i in range(u_dim + y_dim) if i not in self.y_coords]
        out_dim = u_dim if out_dim is None else out_dim

        modules = [
            torch.nn.Linear(u_dim + y_dim + (1 if time_varying else 0), w),
            torch.nn.SELU()
        ]
        for _ in range(layers):
            modules.append(torch.nn.Linear(w, w))
            modules.append(torch.nn.SELU())
        modules.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        # Assumes x is the concatenation of the input and the time t
        model_out = self.net(x)

        # Embed as a triangular vector field, i.e. of the form (0, v_t)
        out = torch.zeros(x.shape[0], self.u_dim + self.y_dim, device=x.device)
        out[:, self.u_coords] = model_out

        return out