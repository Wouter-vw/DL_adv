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
import flax
import optax
from dataclasses import dataclass
from jax.scipy.special import logsumexp
import tensorflow as tf
import time


class linearized_cross_entropy_manifold:
    """
    Also in this case I have to separate data fitting term and regularization term for gradient and
    hessian computation in case of batches.
    """

    def __init__(
        self,
        model,
        state_model,
        X,
        y,
        f_MAP,
        theta_MAP,
        unravel_fn,
        batching=False,
        lambda_reg=None,
        N=None,
        B1=None,
        B2=None,
    
    ):
        self.model = model
        self.state = state_model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.batching = batching
        self.type = type
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP
        # Initialize model parameters
        self.params = theta_MAP
        self.unravel_fn = unravel_fn

        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = 1.0

        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)
    @staticmethod
    def is_diagonal():
        return False

    @partial(jax.jit, static_argnums=(0))
    def CE_loss(self, param, data, f_MAP):
        """
        Data fitting term of the loss
        """

        def predict(params, datas):
            y_preds = self.state.apply_fn(params, datas)
            return y_preds

        x, y = data

        params_map = self.unravel_fn(self.theta_MAP)
        diff_weights = jax.tree_util.tree_map(lambda p, m: (p - m).astype(m.dtype), param, self.params_map)
        _, jvp_value = jvp(predict, (params_map, x), (diff_weights, jnp.zeros_like(x)))

        y_pred = f_MAP + jvp_value
        
        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)
            
            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)
            
            # Use targets as indices (not one-hot encoded)
            return jnp.sum(-jnp.take_along_axis(log_probs, targets[:, None], axis=-1))  # Use targets as indices
        
        return self.factor * criterion(y_pred, y)


    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation.
        """
        # Sum squared values of the parameters
        w_norm = sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(param['params']))
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_grad_data_fitting_term(self, params, data, f_MAP):
        ft_compute_grad = grad(self.CE_loss)

        ft_per_sample_grads = ft_compute_grad(params, data, f_MAP)
        return ft_per_sample_grads

    @partial(jax.jit, static_argnums=(0))
    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        ft_per_sample_grads = ft_compute_grad(params)
        return jax.flatten_util.ravel_pytree(ft_per_sample_grads)[0]
    
    def compute_kfac_hvp(self, intermediates, ft_compute_grad, vel_as_params, num_layers=3):
        """
        Compute the Hessian-vector product (HVP) using K-FAC approximation.
        
        Args:
            intermediates (dict): Intermediate activations from the forward pass.
            ft_compute_grad (dict): Gradients of the parameters.
            vel_as_params (dict): Velocity vector in parameter space.
            num_layers (int): Number of layers to process.

        Returns:
            jnp.ndarray: Hessian-vector product.
        """
        kfac_factors = {}

        # Compute K-FAC factors
        for i in range(num_layers):
            # Weight factors
            A = jnp.matmul(intermediates['intermediates'][f'in_{i}'][0].T, intermediates['intermediates'][f'in_{i}'][0].T) /intermediates['intermediates'][f'in_{i}'][0].shape[0]
            #A = intermediates['intermediates'][f'in_{i}'][0].T @ intermediates['intermediates'][f'in_{i}'][0]
            G = jnp.matmul(ft_compute_grad['params'][f'Dense_{i}']['kernel'].T, ft_compute_grad['params'][f'Dense_{i}']['kernel']) /intermediates['intermediates'][f'in_{i}'][0].shape[0]
            #G = ft_compute_grad['params'][f'Dense_{i}']['kernel'] @ ft_compute_grad['params'][f'Dense_{i}']['kernel'].T
            kfac_factors[f'layer_{i}'] = (G, A)

            # Bias factors
            G_bias = ft_compute_grad['params'][f'Dense_{i}']['bias']
            A_bias = jnp.ones(G_bias.shape[0])
            kfac_factors[f'bias_{i}'] = (G_bias, A_bias)

        # Compute HVP
        hvp_list = []
        for i in range(num_layers):
            # Bias contribution
            bias = kfac_factors[f'bias_{i}'][0] * vel_as_params['params'][f'Dense_{i}']['bias']
            # Weight contribution
            weight = jnp.matmul(
                kfac_factors[f'layer_{i}'][1],
                jnp.matmul(kfac_factors[f'layer_{i}'][0], vel_as_params['params'][f'Dense_{i}']['kernel'])
            )
            hvp_list.extend([bias, weight.flatten()])

        return jnp.concatenate(hvp_list)

    
    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):

        if isinstance(self.X, tf.data.Dataset):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        state = self.state.replace(params = self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            params = self.unravel_fn(current_point)
            self.params_map = self.unravel_fn(self.theta_MAP)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = jax.flatten_util.ravel_pytree(self.compute_grad_data_fitting_term(params, (batch_img, batch_label), batch_MAP))[0]
                grad_data_fitting_term += grad_per_example.reshape(-1, 1)
        else:
            params = self.unravel_fn(current_point)
            self.params_map = self.unravel_fn(self.theta_MAP)
            
            grad_per_example = jax.flatten_util.ravel_pytree(self.compute_grad_data_fitting_term(params, data, self.f_MAP))[0]
            grad_data_fitting_term = grad_per_example.reshape(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = grad_reg.reshape(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        vel_as_params = self.unravel_fn(velocity)

        start = time.time()
        hvp_data_fitting = 0
        if batchify:
            for batch_img, batch_label, batch_f_MAP in self.X:
                _, intermediates = self.model.apply(state.params, batch_img, mutable=['intermediates'])
                hvp_data_fitting = self.compute_kfac_hvp(intermediates, grad_data_fitting_term, vel_as_params, num_layers=3)
        else:
            _, intermediates = self.model.apply(state.params, data[0], mutable=['intermediates'])
            hvp_data_fitting = self.compute_kfac_hvp(intermediates, grad_data_fitting_term, vel_as_params, num_layers=3)

        # I have to add the hvp of the regularization term
        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.reshape(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)
        


class cross_entropy_manifold:
    """
    Also in this case I have to split the gradient loss computation and the gradient of the regularization
    term.
    This is needed to get the correct gradient and hessian computation when using batches.
    """

    def __init__(
        self, 
        state_model, 
        X, 
        y,
        unravel_fn, 
        batching=False, 
        lambda_reg=None, 
        N=None, 
        B1=None, 
        B2=None
    ):
        self.state = state_model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.batching = batching
        self.type = type
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        self.unravel_fn = unravel_fn


        ## stuff we need when using batches
        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = 1.0

        # here I can already compute the factor_loss
        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)

    @staticmethod
    def is_diagonal():
        return False

    @partial(jax.jit, static_argnums=(0))
    def CE_loss(self, param, data):
        """
        Data fitting term of the loss
        """
        x, y = data
        y_pred = self.state.apply_fn(param, x)

        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)
            
            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)
            
            # Use targets as indices (not one-hot encoded)
            return jnp.sum(-jnp.take_along_axis(log_probs, targets[:, None], axis=-1))  # Use targets as indices

        return self.factor * criterion(y_pred, y)



    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation.
        """
        # Sum squared values of the parameters
        w_norm = sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(param['params']))
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_grad_data_fitting_term(self, params, data):
        ft_compute_grad = grad(self.CE_loss)

        ft_per_sample_grads = ft_compute_grad(params, data)
        return jax.flatten_util.ravel_pytree(ft_per_sample_grads)[0]

    @partial(jax.jit, static_argnums=(0))
    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        ft_per_sample_grads = ft_compute_grad(params)
        return jax.flatten_util.ravel_pytree(ft_per_sample_grads)[0]
    
    @partial(jax.jit, static_argnums=(0, 1))
    def custom_hvp(self, f, primals, tangents):
        grad_f = grad(f)
        return jax.jvp(grad_f, primals, tangents)
    
    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        if isinstance(self.X, tf.data.Dataset):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        state = self.state.replace(params = self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            params = self.unravel_fn(current_point)
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += grad_per_example.reshape(-1, 1)
        else:
            params = self.unravel_fn(current_point)
            grad_per_example = self.compute_grad_data_fitting_term(params, data)
            grad_data_fitting_term = grad_per_example.reshape(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = grad_reg.reshape(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        # and I have to reshape the velocity into being the same structure as the params
        vel_as_params = self.unravel_fn(velocity)

        # now I have also to compute the Hvp between hessian and velocity
        start = time.time()
        hvp_data_fitting = 0
        if batchify:
            for batch_img, batch_label in self.X:
                _, result = self.custom_hvp(
                    self.CE_loss,
                    (params, (batch_img, batch_label)),
                    (vel_as_params, (jnp.zeros_like(batch_img), jnp.zeros_like(batch_label))),
                )
            
            hvp_data_fitting += jax.flatten_util.ravel_pytree(result)[0]

        else:
            _, result = self.custom_hvp(
                self.CE_loss, (params, data), (vel_as_params, (jnp.zeros_like(data[0]), jnp.zeros(data[1].shape, dtype=jax.dtypes.float0)))
            )
        
            hvp_data_fitting += jax.flatten_util.ravel_pytree(result)[0]


        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity
            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.view(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)

