"""
File containing all the manifold we are going to use for the experiments:
- Regression manifold
- Linearized regression manifold
- Cross entropy manifold
- Linearized cross entropy manifold
"""

import jax
import jax.numpy as jnp
from jax import grad, jvp
from functools import partial
from dataclasses import dataclass
import time


class LinearizedCrossEntropyManifold:
    """
    Also in this case I have to separate data fitting term and regularization term for gradient and
    hessian computation in case of batches.
    """

    def __init__(
        self,
        model_state,
        input_data,
        target_labels,
        f_MAP,
        theta_MAP,
        unravel_fn,
        batching=False,
        lambda_reg=None,
    ):
        self.model_state = model_state
        self.input_data = input_data
        self.target_labels = target_labels
        self.N = len(self.input_data)
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP
        # Initialize neural_network parameters
        self.params = theta_MAP
        self.unravel_fn = unravel_fn

    @partial(jax.jit, static_argnums=(0))
    def cross_entropy_loss(self, parameters, data, f_MAP):
        """
        Data fitting term of the loss
        """

        def predict(params, datas):
            y_preds = self.model_state.apply_fn(params, datas)
            return y_preds

        x, y = data

        params_map = self.unravel_fn(self.theta_MAP)
        diff_weights = jax.tree_util.tree_map(lambda p, m: (p - m).astype(m.dtype), parameters, self.params_map)
        _, jvp_value = jvp(predict, (params_map, x), (diff_weights, jnp.zeros_like(x)))

        y_pred = f_MAP + jvp_value

        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)

            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)

            # Use targets as indices (not one-hot encoded)
            return jnp.sum(-jnp.take_along_axis(log_probs, targets[:, None], axis=-1))  # Use targets as indices

        return criterion(y_pred, y)

    def L2_norm(self, parameters):
        w_norm = sum(jnp.sum(w**2) for w in jax.tree_util.tree_leaves(parameters["params"]))
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_data_term_gradient(self, params, data, f_MAP):
        gradient_fun = grad(self.cross_entropy_loss)
        gradient_tree = gradient_fun(params, data, f_MAP)
        return jax.flatten_util.ravel_pytree(gradient_tree)[0]

    @partial(jax.jit, static_argnums=(0))
    def compute_regularization_gradient(self, params):
        gradient_fun = grad(self.L2_norm)
        gradient_tree = gradient_fun(params)
        return jax.flatten_util.ravel_pytree(gradient_tree)[0]

    @partial(jax.jit, static_argnums=(0, 1))
    def custom_hvp(self, function, primals, tangents):
        grad_function = grad(function)
        return jax.jvp(grad_function, primals, tangents)

    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        data = (self.input_data, self.target_labels)
        self.model_state = self.model_state.replace(params=self.unravel_fn(current_point))

        data_term_gradient = 0
        params = self.unravel_fn(current_point)
        self.params_map = self.unravel_fn(self.theta_MAP)

        data_term_gradient = self.compute_data_term_gradient(params, data, self.f_MAP).reshape(-1, 1)

        if self.lambda_reg is not None:
            grad_reg = self.compute_regularization_gradient(params)
            grad_reg = grad_reg.reshape(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = data_term_gradient + grad_reg

        velocity_tree = self.unravel_fn(velocity)

        hvp_data_fitting = 0
        _, result = self.custom_hvp(
            self.cross_entropy_loss, (params, data, self.f_MAP), (velocity_tree, (jnp.zeros_like(data[0]), jnp.zeros(data[1].shape, dtype=jax.dtypes.float0)), jnp.zeros_like(self.f_MAP))
        )

        hvp_data_fitting += jax.flatten_util.ravel_pytree(result)[0]

        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.reshape(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)


class CrossEntropyManifold:
    def __init__(
        self,
        model_state,
        input_data,
        target_labels,
        unravel_fn,
        batching=False,
        lambda_reg=None,
    ):
        self.model_state = model_state
        self.input_data = input_data
        self.target_labels = target_labels
        self.N = len(self.input_data)
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.unravel_fn = unravel_fn

    @partial(jax.jit, static_argnums=(0))
    def cross_entropy_loss(self, parameters, data):
        x, y = data
        y_pred = self.model_state.apply_fn(parameters, x)

        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)

            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)

            # Use targets as indices (not one-hot encoded)
            return jnp.sum(-jnp.take_along_axis(log_probs, targets[:, None], axis=-1))  # Use targets as indices

        return criterion(y_pred, y)

    def L2_norm(self, parameters):
        w_norm = sum(jnp.sum(w**2) for w in jax.tree_util.tree_leaves(parameters["params"]))
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_data_term_gradient(self, params, data):
        gradient_fun = grad(self.cross_entropy_loss)
        gradient_tree = gradient_fun(params, data)
        return jax.flatten_util.ravel_pytree(gradient_tree)[0]

    @partial(jax.jit, static_argnums=(0))
    def compute_regularization_gradient(self, params):
        gradient_fun = grad(self.L2_norm)
        gradient_tree = gradient_fun(params)
        return jax.flatten_util.ravel_pytree(gradient_tree)[0]

    @partial(jax.jit, static_argnums=(0, 1))
    def custom_hvp(self, function, primals, tangents):
        grad_function = grad(function)
        return jax.jvp(grad_function, primals, tangents)

    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        data = (self.input_data, self.target_labels)
        self.model_state = self.model_state.replace(params=self.unravel_fn(current_point))

        grad_data_fitting_term = 0
        params = self.unravel_fn(current_point)
        grad_data_fitting_term = self.compute_data_term_gradient(params, data).reshape(-1, 1)

        if self.lambda_reg is not None:
            grad_reg = self.compute_regularization_gradient(params)
            grad_reg = grad_reg.reshape(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        velocity_tree = self.unravel_fn(velocity)

        hvp_data_fitting = 0
        _, result = self.custom_hvp(self.cross_entropy_loss, (params, data), (velocity_tree, (jnp.zeros_like(data[0]), jnp.zeros(data[1].shape, dtype=jax.dtypes.float0))))

        hvp_data_fitting += jax.flatten_util.ravel_pytree(result)[0]

        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity
            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.view(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)
