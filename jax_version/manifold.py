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
import flax
import numpy as np
import math
import time
import copy
import optax
from dataclasses import dataclass

@dataclass(frozen=True)
class LayerInfo:
    layer_name: str
    kernel_shape: tuple
    bias_shape: tuple
    kernel_idx: int
    bias_idx: int
    kernel_size: int
    bias_size: int

def precompute_param_info(state):
    param_info = []
    idx = 0
    for layer_name, layer_params in state.params['params'].items():
        kernel_shape = layer_params['kernel'].shape  # Should be a tuple
        bias_shape = layer_params['bias'].shape      # Should be a tuple
        kernel_size = int(kernel_shape[0] * kernel_shape[1])
        bias_size = int(bias_shape[0])
        info = LayerInfo(
            layer_name=layer_name,
            kernel_shape=kernel_shape,
            bias_shape=bias_shape,
            kernel_idx=idx,
            bias_idx=idx + kernel_size,
            kernel_size=kernel_size,
            bias_size=bias_size
        )
        param_info.append(info)
        idx += kernel_size + bias_size
    return tuple(param_info)  # Convert list to tuple to make it hashable


@partial(jax.jit, static_argnums=(1,))
def params_from_map_info(solution, param_info):
    params = {'params': {}}
    for info in param_info:
        # Extract and reshape kernel using dynamic slicing
        kernel_flat = jax.lax.dynamic_slice(
            solution,
            [info.kernel_idx],
            [info.kernel_size]
        )
        kernel = kernel_flat.reshape(info.kernel_shape)

        # Extract and reshape bias using dynamic slicing
        bias_flat = jax.lax.dynamic_slice(
            solution,
            [info.bias_idx],
            [info.bias_size]
        )
        bias = bias_flat.reshape(info.bias_shape)

        # Assign to params
        params['params'][info.layer_name] = {
            'kernel': kernel,
            'bias': bias
        }

    return flax.core.FrozenDict(params)


def params_from_map(solution, state):
        params = {'params': {}}
        idx = 0

        # Iterate over the layers to reconstruct the parameters dynamically
        for layer_name, layer_params in state.params['params'].items():
            # Get the shape of the kernel and bias
            kernel_shape = layer_params['kernel'].shape
            bias_shape = layer_params['bias'].shape
            
            # Calculate the number of elements in the kernel and bias
            kernel_size = jnp.prod(jnp.array(kernel_shape))
            bias_size = jnp.prod(jnp.array(bias_shape))


            # Extract and reshape kernel from solution
            kernel_flat = solution[idx:idx + kernel_size]
            if kernel_flat.size == 0:
                raise ValueError(f"Not enough elements in solution for layer {layer_name} kernel.")
            kernel = kernel_flat.reshape(kernel_shape)
            idx += kernel_size
            
            # Extract and reshape bias from map_solution
            bias_flat = solution[idx:idx + bias_size]
            if bias_flat.size == 0:
                raise ValueError(f"Not enough elements in solution for layer {layer_name} bias.")
            bias = bias_flat.reshape(bias_shape)
            idx += bias_size
            
            # Assign the kernel and bias to the params dictionary
            params['params'][layer_name] = {'kernel': kernel, 'bias': bias}
        
        # Replace the state with the new params
        new_state = state.replace(params=params)

        return new_state

def custom_hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]

def make_zero_tangent(x):
    if jnp.issubdtype(x.dtype, jnp.integer) or jnp.issubdtype(x.dtype, jnp.bool_):
        return jax.ShapeDtypeStruct(x.shape, jax.dtypes.float0)
    else:
        return jnp.zeros_like(x)

class regression_manifold:
    def __init__(self, model, X, y, batching=False, lambda_reg=None, noise_var=1):
        self.model = model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.batching = batching

        assert y is None if batching else True, "If batching is True, y should be None"

        # Initialize model parameters
        rng = jax.random.PRNGKey(0)
        dummy_input = self.X[0]
        variables = self.model.init(rng, dummy_input)
        self.params = variables['params']
        self.n_params, self.unravel_fn = self.get_num_params(self.params)

    def get_num_params(self, params):
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        n_params = flat_params.shape[0]
        return n_params, unravel_fn

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, params):
        if self.lambda_reg is None:
            return 0.0
        w_norm = sum([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)])
        return self.lambda_reg * w_norm

    def mse_loss(self, params, data):
        x, y = data
        y_pred = self.model.apply({'params': params}, x)
        loss = 0.5 * (1.0 / self.noise_var) * jnp.sum((y_pred - y) ** 2)
        return loss

    def compute_grad_data_fitting_term(self, params, data):
        loss_fn = lambda p: self.mse_loss(p, data)
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
            # Compute gradient over batches
            grad_data_fitting_term = None
            for batch_x, batch_y in self.X:
                data = (batch_x, batch_y)
                grad_per_batch = self.compute_grad_data_fitting_term(params, data)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data)

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
            return self.mse_loss(p, data) + self.L2_norm(p)

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
            for batch_x, batch_y in self.X:
                data = (batch_x, batch_y)
                grad_per_batch = self.compute_grad_data_fitting_term(params, data)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data)

        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            total_grad = jax.tree_util.tree_map(
                lambda x, y: x + y, grad_data_fitting_term, grad_reg
            )
        else:
            total_grad = grad_data_fitting_term

        flat_total_grad, _ = jax.flatten_util.ravel_pytree(total_grad)
        return flat_total_grad


# Linearized regression manifold
class linearized_regression_manifold:
    def __init__(self, model, X, y, f_MAP, theta_MAP, batching=False, lambda_reg=None, noise_var=1):
        self.model = model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.batching = batching
        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP

        assert y is None if batching else True, "If batching is True, y should be None"

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

    def L2_norm(self, params):
        if self.lambda_reg is None:
            return 0.0
        w_norm = sum([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)])
        return self.lambda_reg * w_norm

    def mse_loss(self, params, data, f_MAP):
        x, y = data
        # Compute difference between params and theta_MAP
        diff_params = jax.tree_util.tree_map(lambda a, b: a - b, params, self.theta_MAP)
        # Compute jvp
        _, jvp_value = jvp(lambda p: self.model.apply({'params': p}, x), (self.theta_MAP,), (diff_params,))
        y_pred = f_MAP + jvp_value
        loss = 0.5 * (1.0 / self.noise_var) * jnp.sum((y_pred - y) ** 2)
        return loss

    def compute_grad(self, params, data, f_MAP):
        loss_fn = lambda p: self.mse_loss(p, data, f_MAP)
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
                grad_per_batch = self.compute_grad(params, data, batch_f_MAP)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad(params, data, self.f_MAP)

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
            return self.mse_loss(p, data, self.f_MAP) + self.L2_norm(p)

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
                grad_per_batch = self.compute_grad(params, data, batch_f_MAP)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad(params, data, self.f_MAP)

        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            total_grad = jax.tree_util.tree_map(
                lambda x, y: x + y, grad_data_fitting_term, grad_reg
            )
        else:
            total_grad = grad_data_fitting_term

        flat_total_grad, _ = jax.flatten_util.ravel_pytree(total_grad)
        return flat_total_grad


class cross_entropy_manifold:
    def __init__(self, model, X, y, batching=False, lambda_reg=None, N=None, B1=None, B2=None):
        self.model = model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.lambda_reg = lambda_reg
        self.batching = batching
        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None
        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor *= (B2 / B1)

        # Initialize model parameters
        rng = jax.random.PRNGKey(0)
        dummy_input = self.X[0]
        # variables = self.model.init(rng, dummy_input)
        # self.params = variables['params']
        self.params = self.model.params
        self.n_params, self.unravel_fn = self.get_num_params(self.params)

    def get_num_params(self, params):
        flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        n_params = flat_params.shape[0]
        return n_params, unravel_fn

    @staticmethod
    def is_diagonal():
        return False

    def CE_loss(self, params, data):
        x, y = data
        logits = self.model.apply_fn(params, x) #{'params': params}
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

    def compute_grad_data_fitting_term(self, params, data):
        loss_fn = lambda p: self.CE_loss(p, data)
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

        n_params = len(current_point)
    
        # Initialize accumulated gradient as a zero vector of shape (n_params, 1)
        grad_data_fitting_term = jnp.zeros((n_params, 1))
        
        if self.batching:
            for batch_x, batch_y in self.X:
                data = (batch_x, batch_y)
                # Compute gradient per batch
                grad_per_batch = self.compute_grad_data_fitting_term(params, data)
                # Flatten the gradient per batch
                flat_grad_per_batch, _ = jax.flatten_util.ravel_pytree(grad_per_batch)
                # Reshape to (-1, 1)
                flat_grad_per_batch = flat_grad_per_batch.reshape(-1, 1)
                # Accumulate gradients
                grad_data_fitting_term += flat_grad_per_batch
        else:
            data = (self.X, self.y)
            grad_per_batch = self.compute_grad_data_fitting_term(params, data)
            flat_grad_per_batch, _ = jax.flatten_util.ravel_pytree(grad_per_batch)
            flat_grad_per_batch = flat_grad_per_batch.reshape(-1, 1)
            grad_data_fitting_term = flat_grad_per_batch

        # Compute gradient of L2 regularization
        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            # Flatten and reshape grad_reg
            flat_grad_reg, _ = jax.flatten_util.ravel_pytree(grad_reg)
            flat_grad_reg = flat_grad_reg.reshape(-1, 1)
            # Total gradient is the sum
            total_grad = grad_data_fitting_term + flat_grad_reg
        else:
            total_grad = grad_data_fitting_term

        # Define total loss function
        def total_loss_fn(p):
            return self.CE_loss(p, data) + self.L2_norm(p)

        # Compute HVP of CE_loss including data in primals and zero tangents for data
        if self.batching:
            # Initialize accumulated HVP
            total_hvp = jnp.zeros(n_params)
            for batch_x, batch_y in self.X:
                data = (batch_x, batch_y)
                def ce_loss_fn(p, d):
                    return self.CE_loss(p, d)
                # Create zero tangents for data
                zero_data_tangent = jax.tree_util.tree_map(make_zero_tangent, data)
                # Compute HVP per batch
                hvp_ce_batch = custom_hvp(ce_loss_fn, (params, data), (velocity_params, zero_data_tangent))
                # Flatten hvp_ce_batch
                flat_hvp_ce_batch, _ = jax.flatten_util.ravel_pytree(hvp_ce_batch)
                # Accumulate HVP
                total_hvp += flat_hvp_ce_batch
        else:
            data = (self.X, self.y)
            def ce_loss_fn(p, d):
                return self.CE_loss(p, d)
            zero_data_tangent = jax.tree_util.tree_map(make_zero_tangent, data)
            hvp_ce_batch = custom_hvp(ce_loss_fn, (params, data), (velocity_params, zero_data_tangent))
            flat_hvp_ce_batch, _ = jax.flatten_util.ravel_pytree(hvp_ce_batch)
            total_hvp = flat_hvp_ce_batch

        # Compute HVP of L2 regularization separately
        if self.lambda_reg is not None:
            # HVP of L2 regularization is 2 * lambda_reg * velocity
            flat_velocity, _ = jax.flatten_util.ravel_pytree(velocity_params)
            hvp_reg = 2 * self.lambda_reg * flat_velocity
            # Total HVP is sum of HVPs
            total_hvp += hvp_reg
                
        # Compute second derivative
        flat_velocity, _ = jax.flatten_util.ravel_pytree(velocity_params)
        numerator = jnp.dot(flat_velocity, total_hvp)
        denom = 1.0 + jnp.dot(total_grad.T, total_grad).item()  # total_grad is (n_params, 1)
        second_derivative = - (total_grad.flatten() / denom) * numerator

        if return_hvp:
            return second_derivative, flat_hvp
        else:
            return second_derivative

    def get_gradient_value_in_specific_point(self, current_point):
        # Convert current_point to parameter structure
        params = self.unravel_fn(current_point)

        if self.batching:
            grad_data_fitting_term = None
            for batch_x, batch_y in self.X:
                data = (batch_x, batch_y)
                grad_per_batch = self.compute_grad_data_fitting_term(params, data)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data)

        grad_reg = self.compute_grad_L2_reg(params)
        if grad_reg is not None:
            total_grad = jax.tree_util.tree_map(
                lambda x, y: x + y, grad_data_fitting_term, grad_reg
            )
        else:
            total_grad = grad_data_fitting_term

        flat_total_grad, _ = jax.flatten_util.ravel_pytree(total_grad)
        return flat_total_grad


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

    @partial(jax.jit, static_argnums=(0, 5))
    def CE_loss(self, params, data, f_MAP, theta_MAP, jvp_fn, factor=None):
        x, y = data
        # Compute difference between params and theta_MAP
        diff_params = jax.tree_util.tree_map(lambda a, b: jnp.asarray(a - b, dtype=jnp.float32), params, self.theta_MAP)
        # Compute jvp
        _, jvp_value = jvp(jvp_fn, (theta_MAP,), (diff_params,))
        logits = f_MAP + jvp_value
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        loss = jnp.sum(loss)
        if factor is not None:
            loss *= factor
        return loss

    def L2_norm(self, params):
        if self.lambda_reg is None:
            return 0.0
        w_norm = sum([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)])
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0, 5))
    def compute_grad_data_fitting_term(self, params, data, f_MAP, theta_MAP, jvp_fn, factor=None):
        loss_fn = lambda p: self.CE_loss(p, data, f_MAP, theta_MAP, jvp_fn, factor)
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

    # @profile
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # Convert current_point and velocity to parameter structures
        params = self.unravel_fn(current_point)
        velocity_params = self.unravel_fn(velocity)

        param_info = precompute_param_info(self.model)


        # Compute gradient of data fitting term
        if self.batching:
            grad_data_fitting_term = None
            for batch_x, batch_y, batch_f_MAP in self.X:
                data = (batch_x, batch_y)
                def jvp_fn(p):
                    state = params_from_map_info(p, param_info)
                    return self.model.apply_fn(state.params, batch_x)

                grad_per_batch = self.compute_grad_data_fitting_term(params, data, batch_f_MAP, self.theta_MAP, jvp_fn, self.factor)
                if grad_data_fitting_term is None:
                    grad_data_fitting_term = grad_per_batch
                else:
                    grad_data_fitting_term = jax.tree_util.tree_map(
                        lambda x, y: x + y, grad_data_fitting_term, grad_per_batch
                    )
        else:
            data = (self.X, self.y)
            def jvp_fn(p):
                state = params_from_map_info(p, param_info)
                return self.model.apply_fn(state["params"], self.X)

            grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data, self.f_MAP, self.theta_MAP, jvp_fn, self.factor)

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
        # hvp_fn = lambda v: jax.jvp(grad(total_loss_fn), (params,), (v,))[1]
        jit_grad_total_loss_fn = jax.jit(grad(total_loss_fn))

        @jax.jit
        def hvp_fn(v):
            _, hvp = jax.jvp(jit_grad_total_loss_fn, (params,), (v,))
            return hvp
        
        hvp_params = hvp_fn(velocity_params)
        flat_hvp, _ = jax.flatten_util.ravel_pytree(hvp_params)

        # Compute second derivative
        denom = 1.0 + jnp.dot(flat_total_grad, flat_total_grad)
        numerator = jnp.dot(velocity.flatten(), flat_hvp)
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