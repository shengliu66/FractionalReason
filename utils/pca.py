import torch
import torch.nn as nn
import torch.nn.functional as F


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components, set_mean_to_zero=False):
        super().__init__()
        self.n_components = n_components
        self.set_mean_to_zero = set_mean_to_zero

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        if self.set_mean_to_zero:
            self.register_buffer("mean_", torch.zeros_like(X.mean(0, keepdim=True)))
        else:
            self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        self.register_buffer("singular_values_", S)
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

    def explained_variance_ratio(self):
        assert hasattr(self, "singular_values_"), "PCA must be fit before use."
        # Compute explained variance ratio
        total_var = (self.singular_values_ ** 2).sum(dim=-1, keepdim=True)
        explained_var_ratio = (self.singular_values_ ** 2) / total_var
        return explained_var_ratio