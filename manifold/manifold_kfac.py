import jax
import jax.numpy as jnp
from jax import grad, jvp
from functools import partial
import time

jax.config.update("jax_enable_x64", True)


class LinearizedCrossEntropyManifold_kfac:
    """
    Also in this case I have to separate data fitting term and regularization term for gradient and
    hessian computation in case of batches.
    """

    def __init__(
        self,
        neural_network,
        model_state,
        input_data,
        target_labels,
        f_MAP,
        theta_MAP,
        unravel_fn,
        batching=False,
        lambda_reg=None,
    ):
        self.neural_network = neural_network
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
        diff_weights = jax.tree_util.tree_map(
            lambda p, m: (p - m).astype(m.dtype), parameters, self.params_map
        )
        _, jvp_value = jvp(predict, (params_map, x), (diff_weights, jnp.zeros_like(x)))

        y_pred = f_MAP + jvp_value

        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)

            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)

            # Use targets as indices (not one-hot encoded)
            return jnp.sum(
                -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
            )  # Use targets as indices

        return criterion(y_pred, y)

    def L2_norm(self, parameters):
        """
        L2 regularization. I need this separate from the loss for the gradient computation.
        """
        # Sum squared values of the parameters
        w_norm = sum(
            jnp.sum(w**2) for w in jax.tree_util.tree_leaves(parameters["params"])
        )
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_data_term_gradient(self, params, data, f_MAP):
        gradient_fun = grad(self.cross_entropy_loss)
        gradient_tree = gradient_fun(params, data, f_MAP)
        return gradient_tree

    @partial(jax.jit, static_argnums=(0))
    def compute_regularization_gradient(self, params):
        gradient_fun = grad(self.L2_norm)
        gradient_tree = gradient_fun(params)
        return gradient_tree

    @partial(jax.jit, static_argnums=(0, 4))
    def compute_kfac_hvp(
        self, intermediates, gradient_tree, velocity_tree, num_layers=3
    ):
        """
        Compute the Hessian-vector product (HVP) using K-FAC approximation.

        Args:
            intermediates (dict): Intermediate activations from the forward pass.
            gradient_tree (dict): Gradients of the parameters.
            velocity_tree (dict): Velocity vector in parameter space.
            num_layers (int): Number of layers to process.

        Returns:
            jnp.ndarray: Hessian-vector product.
        """
        hvp_list = []

        for i in range(num_layers):
            layer_name = f"Dense_{i}"
            interm_key = f"in_{i}"

            # Extract activations and gradients
            activations = intermediates["intermediates"][interm_key][
                0
            ]  # Shape: (batch_size, n_in)
            grads = gradient_tree["params"][layer_name][
                "kernel"
            ]  # Shape: (n_in, n_out)
            vel = velocity_tree["params"][layer_name]["kernel"]  # Shape: (n_in, n_out)

            # Compute activation covariance A
            A = (
                jnp.matmul(activations.T, activations) / activations.shape[0]
            )  # Shape: (n_in, n_in)

            # Compute gradient covariance G
            G = (
                jnp.matmul(grads.T, grads) / activations.shape[0]
            )  # Shape: (n_out, n_out)

            # Bias terms
            grads_bias = gradient_tree["params"][layer_name]["bias"]  # Shape: (n_out,)
            vel_bias = velocity_tree["params"][layer_name]["bias"]  # Shape: (n_out,)
            G_bias = (grads_bias**2) / activations.shape[0]  # Shape: (n_out,)
            bias_hvp = G_bias * vel_bias  # Element-wise multiplication
            hvp_list.append(bias_hvp.flatten())

            # Compute HVP for weights
            weight_hvp = jnp.matmul(A, jnp.matmul(vel, G))  # Shape: (n_in, n_out)
            hvp_list.append(weight_hvp.flatten())

        return jnp.concatenate(hvp_list)

    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        data = (self.input_data, self.target_labels)

        # let's start by putting the current points into the neural_network
        self.model_state = self.model_state.replace(params=self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        data_term_gradient = 0
        params = self.unravel_fn(current_point)
        self.params_map = self.unravel_fn(self.theta_MAP)
        data_term_gradient = self.compute_data_term_gradient(params, data, self.f_MAP)

        # here now I have to compute also the gradient of the regularization term
        # Compute gradient of the regularization term
        if self.lambda_reg is not None:
            regularization_gradient = self.compute_regularization_gradient(params)
        else:
            regularization_gradient = None

        # Combine gradients
        tot_gradient = data_term_gradient
        if regularization_gradient is not None:
            tot_gradient = jax.tree_util.tree_map(
                lambda x, y: x + y, tot_gradient, regularization_gradient
            )

        tot_gradient = jax.flatten_util.ravel_pytree(tot_gradient)[0]

        velocity_tree = self.unravel_fn(velocity)

        start = time.time()
        hvp_data_fitting = 0
        _, intermediates = self.neural_network.apply(
            self.model_state.params, data[0], mutable=["intermediates"]
        )
        hvp_data_fitting = self.compute_kfac_hvp(
            intermediates, data_term_gradient, velocity_tree, num_layers=3
        )

        # I have to add the hvp of the regularization term
        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting
        end = time.time()

        second_derivative = -(
            (tot_gradient / (1 + tot_gradient.T @ tot_gradient))
            * (velocity.T @ tot_hvp)
        ).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.reshape(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)


class CrossEntropyManifold_kfac:
    """
    Also in this case I have to split the gradient loss computation and the gradient of the regularization
    term.
    This is needed to get the correct gradient and hessian computation when using batches.
    """

    def __init__(
        self,
        neural_network,
        model_state,
        input_data,
        target_labels,
        unravel_fn,
        batching=False,
        lambda_reg=None,
    ):
        self.neural_network = neural_network
        self.model_state = model_state
        self.input_data = input_data
        self.target_labels = target_labels
        self.N = len(self.input_data)
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.unravel_fn = unravel_fn

    @partial(jax.jit, static_argnums=(0))
    def cross_entropy_loss(self, parameters, data):
        """
        Data fitting term of the loss
        """
        x, y = data
        y_pred = self.model_state.apply_fn(parameters, x)

        def criterion(predictions, targets):
            # Apply softmax to predictions to get probabilities
            probs = jax.nn.softmax(predictions, axis=-1)

            # Use log of probabilities to get log-probabilities
            log_probs = jnp.log(probs)

            # Use targets as indices (not one-hot encoded)
            return jnp.sum(
                -jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
            )  # Use targets as indices

        return criterion(y_pred, y)

    def L2_norm(self, parameters):
        """
        L2 regularization. I need this separate from the loss for the gradient computation.
        """
        # Sum squared values of the parameters
        w_norm = sum(
            jnp.sum(w**2) for w in jax.tree_util.tree_leaves(parameters["params"])
        )
        return self.lambda_reg * w_norm

    @partial(jax.jit, static_argnums=(0))
    def compute_data_term_gradient(self, params, data):
        gradient_fun = grad(self.cross_entropy_loss)

        gradient_tree = gradient_fun(params, data)
        return gradient_tree

    @partial(jax.jit, static_argnums=(0))
    def compute_regularization_gradient(self, params):
        gradient_fun = grad(self.L2_norm)
        gradient_tree = gradient_fun(params)
        return gradient_tree

    @partial(jax.jit, static_argnums=(0, 4))
    def compute_kfac_hvp(
        self, intermediates, gradient_tree, velocity_tree, num_layers=3
    ):
        """
        Compute the Hessian-vector product (HVP) using K-FAC approximation.

        Args:
            intermediates (dict): Intermediate activations from the forward pass.
            gradient_tree (dict): Gradients of the parameters.
            velocity_tree (dict): Velocity vector in parameter space.
            num_layers (int): Number of layers to process.

        Returns:
            jnp.ndarray: Hessian-vector product.
        """
        hvp_list = []

        for i in range(num_layers):
            layer_name = f"Dense_{i}"
            interm_key = f"in_{i}"

            # Extract activations and gradients
            activations = intermediates["intermediates"][interm_key][
                0
            ]  # Shape: (batch_size, n_in)
            grads = gradient_tree["params"][layer_name][
                "kernel"
            ]  # Shape: (n_in, n_out)
            vel = velocity_tree["params"][layer_name]["kernel"]  # Shape: (n_in, n_out)

            # Compute activation covariance A
            A = (
                jnp.matmul(activations.T, activations) / activations.shape[0]
            )  # Shape: (n_in, n_in)

            # Compute gradient covariance G
            G = (
                jnp.matmul(grads.T, grads) / activations.shape[0]
            )  # Shape: (n_out, n_out)

            # Bias terms
            grads_bias = gradient_tree["params"][layer_name]["bias"]  # Shape: (n_out,)
            vel_bias = velocity_tree["params"][layer_name]["bias"]  # Shape: (n_out,)
            G_bias = (grads_bias**2) / activations.shape[0]  # Shape: (n_out,)
            bias_hvp = G_bias * vel_bias  # Element-wise multiplication
            hvp_list.append(bias_hvp.flatten())

            # Compute HVP for weights
            weight_hvp = jnp.matmul(A, jnp.matmul(vel, G))  # Shape: (n_in, n_out)
            hvp_list.append(weight_hvp.flatten())

        return jnp.concatenate(hvp_list)

    @partial(jax.jit, static_argnums=(0,))
    def geodesic_system(self, current_point, velocity, return_hvp=False):
        data = (self.input_data, self.target_labels)

        # let's start by putting the current points into the neural_network
        self.model_state = self.model_state.replace(params=self.unravel_fn(current_point))

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        data_term_gradient = 0
        params = self.unravel_fn(current_point)
        data_term_gradient = self.compute_data_term_gradient(params, data)

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            regularization_gradient = self.compute_regularization_gradient(params)
        else:
            regularization_gradient = None

        tot_gradient = data_term_gradient
        if regularization_gradient is not None:
            tot_gradient = jax.tree_util.tree_map(
                lambda x, y: x + y, tot_gradient, regularization_gradient
            )

        tot_gradient = jax.flatten_util.ravel_pytree(tot_gradient)[0]

        # and I have to reshape the velocity into being the same structure as the params
        velocity_tree = self.unravel_fn(velocity)

        # now I have also to compute the Hvp between hessian and velocity
        start = time.time()
        hvp_data_fitting = 0
        _, intermediates = self.neural_network.apply(
            self.model_state.params, data[0], mutable=["intermediates"]
        )
        hvp_data_fitting = self.compute_kfac_hvp(
            intermediates, data_term_gradient, velocity_tree, num_layers=3
        )

        # I have to add the hvp of the regularization term
        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_data_fitting + hvp_reg.reshape(-1)
        else:
            tot_hvp = hvp_data_fitting
        end = time.time()

        second_derivative = -(
            (tot_gradient / (1 + tot_gradient.T @ tot_gradient))
            * (velocity.T @ tot_hvp)
        ).flatten()

        if return_hvp:
            return second_derivative.reshape(-1, 1), tot_hvp.view(-1, 1)
        else:
            return second_derivative.reshape(-1, 1)
