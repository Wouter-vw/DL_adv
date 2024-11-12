import jax
import jax.numpy as jnp
import jax.scipy.linalg

def _precision_to_scale_tril(P):
    """
    Converts a precision matrix P to the lower-triangular Cholesky factor L
    of its inverse covariance matrix, such that L @ L.T = P^{-1}.
    """
    # Perform Cholesky decomposition of P
    L_P = jnp.linalg.cholesky(P)
    # Solve L_P @ X = I for X, where L_P is lower-triangular
    L_inv = jax.scipy.linalg.solve_triangular(L_P, jnp.eye(P.shape[-1]), lower=True)
    # The lower-triangular Cholesky factor of P^{-1} is the transpose of L_inv
    L = L_inv.T
    return L

def get_inverse(Hessian):
    """
    Computes the posterior covariance matrix from the Hessian (precision matrix)
    using the method from the Laplace approximation. This ensures a symmetric
    inverse without directly inverting the Hessian.
    """
    posterior_scale = _precision_to_scale_tril(Hessian)
    posterior_cov = posterior_scale @ posterior_scale.T
    return posterior_cov