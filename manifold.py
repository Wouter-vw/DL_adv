"""
File containing all the manifold we are going to use for the experiments:
- Regression manifold
- Linearized regression manifold
- Cross entropy manifold
- Linearized cross entropy manifold
"""

import jax
import jax.numpy as jnp
from jax import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap
from functools import partial
import flax.linen as nn
import numpy as np
import math
import time
import copy
import optax

class linearized_cross_entropy_manifold:
    def __init__(self, model, X, y, f_MAP, theta_MAP, batching=False, lambda_reg=None, N=None, B1=None, B2=None):
        self.model = model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.lambda_reg = lambda_reg
        self.batching = batching
        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP
        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None
        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor *= (B2 / B1)

        # Initialize model parameters
        self.params = theta_MAP
        self.n_params, self.unravel_fn = self.get_num_params(self.params)

    def get_num_params(self, params):
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        n_params = flat_params.shape[0]
        return n_params, unravel_fn

    @staticmethod
    def is_diagonal():
        return False

    def CE_loss(self, params, data, f_MAP):
        x, y = data
        # Compute difference between params and theta_MAP
        diff_params = jax.tree_util.tree_map(lambda a, b: a - b, params, self.theta_MAP)
        # Compute jvp
        _, jvp_value = jvp(lambda p: self.model.apply({'params': p}, x), (self.theta_MAP,), (diff_params,))
        logits = f_MAP + jvp_value
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        loss = jnp.sum(loss)
        if self.factor is not None:
            loss *= self.factor
        return loss

    def L2_norm(self, params):
        if self.lambda_reg is None:
            return 0.0
        w_norm = sum([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)])
        return self.lambda_reg * w_norm

    def compute_grad_data_fitting_term(self, params, data, f_MAP):
        loss_fn = lambda p: self.CE_loss(p, data, f_MAP)
        grad_fn = grad(loss_fn)
        grad_params = grad_fn(params)
        return grad_params

    def compute_grad_L2_reg(self, params):
        if self.lambda_reg is None:
            return None
        reg_fn = lambda p: self.L2_norm(p)
        grad_fn = grad(reg_fn)
        grad_params = grad_fn(params)
        return grad_params

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # Convert current_point and velocity to parameter structures
        params = self.unravel_fn(current_point)
        velocity_params = self.unravel_fn(velocity)

        # Compute gradient of data fitting term
        if self.batching:
            grad_data_fitting_term = None
            for batch_x, batch_y, batch_f_MAP in self.X:
                data = (batch_x, batch_y)
                grad_per_batch = self.compute_grad_data_fitting_term(params, data, batch_f_MAP)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data, self.f_MAP)

        # Compute gradient of L2 regularization
        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            total_grad = jax.tree_util.tree_map(
                lambda x, y: x + y, grad_data_fitting_term, grad_reg
            )
        else:
            total_grad = grad_data_fitting_term

        # Flatten total gradient
        flat_total_grad, _ = jax.flatten_util.ravel_pytree(total_grad)

        # Define total loss function
        def total_loss_fn(p):
            return self.CE_loss(p, data, self.f_MAP) + self.L2_norm(p)

        # Compute Hessian-vector product
        hvp_fn = lambda v: jax.jvp(grad(total_loss_fn), (params,), (v,))[1]
        hvp_params = hvp_fn(velocity_params)
        flat_hvp, _ = jax.flatten_util.ravel_pytree(hvp_params)

        # Compute second derivative
        denom = 1.0 + jnp.dot(flat_total_grad, flat_total_grad)
        numerator = jnp.dot(velocity, flat_hvp)
        second_derivative = - (flat_total_grad / denom) * numerator

        if return_hvp:
            return second_derivative, flat_hvp
        else:
            return second_derivative

    def get_gradient_value_in_specific_point(self, current_point):
        # Convert current_point to parameter structure
        params = self.unravel_fn(current_point)

        if self.batching:
            grad_data_fitting_term = None
            for batch_x, batch_y, batch_f_MAP in self.X:
                data = (batch_x, batch_y)
                grad_per_batch = self.compute_grad_data_fitting_term(params, data, batch_f_MAP)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data, self.f_MAP)

        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            total_grad = jax.tree_util.tree_map(
                lambda x, y: x + y, grad_data_fitting_term, grad_reg
            )
        else:
            total_grad = grad_data_fitting_term

        flat_total_grad, _ = jax.flatten_util.ravel_pytree(total_grad)
        return flat_total_grad