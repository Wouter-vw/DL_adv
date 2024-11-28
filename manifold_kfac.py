import jax
import jax.numpy as jnp
from jax import grad, jvp
from functools import partial
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
    ):
        self.model = model
        self.state = state_model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.batching = batching
        self.type = type
        self.lambda_reg = lambda_reg
        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP
        # Initialize model parameters
        self.params = theta_MAP
        self.unravel_fn = unravel_fn
        self.factor = 1.0

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
        return ft_per_sample_grads
    
    @partial(jax.jit, static_argnums=(0,4))
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
        hvp_list = []

        for i in range(num_layers):
            layer_name = f'Dense_{i}'
            interm_key = f'in_{i}'

            # Extract activations and gradients
            activations = intermediates['intermediates'][interm_key][0]  # Shape: (batch_size, n_in)
            grads = ft_compute_grad['params'][layer_name]['kernel']      # Shape: (n_in, n_out)
            vel = vel_as_params['params'][layer_name]['kernel']          # Shape: (n_in, n_out)

            # Compute activation covariance A
            A = jnp.matmul(activations.T, activations) / activations.shape[0]  # Shape: (n_in, n_in)

            # Compute gradient covariance G
            G = jnp.matmul(grads.T, grads) / activations.shape[0]              # Shape: (n_out, n_out)

            # Bias terms
            grads_bias = ft_compute_grad['params'][layer_name]['bias']   # Shape: (n_out,)
            vel_bias = vel_as_params['params'][layer_name]['bias']       # Shape: (n_out,)
            G_bias = (grads_bias ** 2) / activations.shape[0]            # Shape: (n_out,)
            bias_hvp = G_bias * vel_bias                                 # Element-wise multiplication
            hvp_list.append(bias_hvp.flatten())

            # Compute HVP for weights
            weight_hvp = jnp.matmul(A, jnp.matmul(vel, G))  # Shape: (n_in, n_out)
            hvp_list.append(weight_hvp.flatten())


        return jnp.concatenate(hvp_list)
    
    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):

        data = (self.X, self.y)

        # let's start by putting the current points into the model
        state = self.state.replace(params = self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        grad_data_fitting_term = 0
        params = self.unravel_fn(current_point)
        self.params_map = self.unravel_fn(self.theta_MAP)            
        grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data, self.f_MAP)

        # here now I have to compute also the gradient of the regularization term
        # Compute gradient of the regularization term
        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(params)
        else:
            grad_reg = None

        # Combine gradients
        tot_gradient = grad_data_fitting_term
        if grad_reg is not None:
            tot_gradient = jax.tree_util.tree_map(lambda x, y: x + y, tot_gradient, grad_reg)

        tot_gradient = jax.flatten_util.ravel_pytree(tot_gradient)[0]

        vel_as_params = self.unravel_fn(velocity)

        start = time.time()
        hvp_data_fitting = 0
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
        model, 
        state_model, 
        X, 
        y,
        unravel_fn, 
        batching=False, 
        lambda_reg=None
    ):
        self.model = model
        self.state = state_model
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.batching = batching
        self.type = type
        self.lambda_reg = lambda_reg
        self.unravel_fn = unravel_fn
        self.factor = 1.0

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
        return ft_per_sample_grads

    @partial(jax.jit, static_argnums=(0))
    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads
    
    @partial(jax.jit, static_argnums=(0,4))
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
        hvp_list = []

        for i in range(num_layers):
            layer_name = f'Dense_{i}'
            interm_key = f'in_{i}'

            # Extract activations and gradients
            activations = intermediates['intermediates'][interm_key][0]  # Shape: (batch_size, n_in)
            grads = ft_compute_grad['params'][layer_name]['kernel']      # Shape: (n_in, n_out)
            vel = vel_as_params['params'][layer_name]['kernel']          # Shape: (n_in, n_out)

            # Compute activation covariance A
            A = jnp.matmul(activations.T, activations) / activations.shape[0]  # Shape: (n_in, n_in)

            # Compute gradient covariance G
            G = jnp.matmul(grads.T, grads) / activations.shape[0]              # Shape: (n_out, n_out)

            # Bias terms
            grads_bias = ft_compute_grad['params'][layer_name]['bias']   # Shape: (n_out,)
            vel_bias = vel_as_params['params'][layer_name]['bias']       # Shape: (n_out,)
            G_bias = (grads_bias ** 2) / activations.shape[0]            # Shape: (n_out,)
            bias_hvp = G_bias * vel_bias                                 # Element-wise multiplication
            hvp_list.append(bias_hvp.flatten())

            # Compute HVP for weights
            weight_hvp = jnp.matmul(A, jnp.matmul(vel, G))  # Shape: (n_in, n_out)
            hvp_list.append(weight_hvp.flatten())


        return jnp.concatenate(hvp_list)
  
    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):

        data = (self.X, self.y)

        # let's start by putting the current points into the model
        state = self.state.replace(params = self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        grad_data_fitting_term = 0
        params = self.unravel_fn(current_point)
        grad_data_fitting_term = self.compute_grad_data_fitting_term(params, data)

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
        else:
            grad_reg = None

        tot_gradient = grad_data_fitting_term
        if grad_reg is not None:
            tot_gradient = jax.tree_util.tree_map(lambda x, y: x + y, tot_gradient, grad_reg)

        tot_gradient = jax.flatten_util.ravel_pytree(tot_gradient)[0]

        # and I have to reshape the velocity into being the same structure as the params
        vel_as_params = self.unravel_fn(velocity)

        # now I have also to compute the Hvp between hessian and velocity
        start = time.time()
        hvp_data_fitting = 0
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
            return second_derivative.reshape(-1, 1), tot_hvp.view(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)
