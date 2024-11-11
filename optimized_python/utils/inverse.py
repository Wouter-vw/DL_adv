import torch

def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L

def get_inverse(Hessian):
    """
    I am trying to copy the way in which Laplace is computing the posterior
    covariance. Because they are able to get symmetric inverse, while I am
    failing in it.
    """
    posterior_scale = _precision_to_scale_tril(Hessian)
    posterior_cov = posterior_scale @ posterior_scale.T

    return posterior_cov
