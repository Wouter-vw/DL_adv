import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap, hvp
from torch.utils.data import DataLoader


class RegressionManifold:
    def __init__(self, model, X, y=None, batching=False, lambda_reg=None, noise_var=1, device="cpu"):
        self.model = model.to(device)
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.device = device

        if batching:
            assert y is None, "If batching is True, y should be None"

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, params):
        """
        L2 regularization. Separate from the loss for gradient computation.
        """
        w_norm = sum((p ** 2).sum() for p in params.values())
        return self.lambda_reg * w_norm

    def mse_loss(self, params, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = functional_call(self.model, params, x)
        loss = nn.MSELoss(reduction="sum")(y_pred, y)
        return (0.5 / self.noise_var) * loss

    def compute_grad_data_fitting_term(self, params, x, y):
        loss_fn = lambda params: self.mse_loss(params, x, y)
        grad_fn = grad(loss_fn)
        grads = grad_fn(params)
        return grads

    def compute_grad_L2_reg(self, params):
        reg_fn = lambda params: self.L2_norm(params)
        grad_fn = grad(reg_fn)
        grads = grad_fn(params)
        return grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # Convert inputs to appropriate types
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)
        if not isinstance(velocity, dict):
            velocity = self._vector_to_parameters(velocity)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grad_data_fitting_term = self.compute_grad_data_fitting_term(current_point, x, y)

        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}

        # Compute Hessian-vector product (HVP)
        def loss_fn(params):
            return self.mse_loss(params, x, y) + self.L2_norm(params)

        hvp_fn = lambda params, vec: hvp(loss_fn, params, vec)[1]
        hvp_result = hvp_fn(current_point, velocity)

        # Compute second derivative
        grad_flat = self._parameters_to_vector(total_grad)
        hvp_flat = self._parameters_to_vector(hvp_result)
        velocity_flat = self._parameters_to_vector(velocity)
        denom = 1 + grad_flat.dot(grad_flat)
        numerator = grad_flat.dot(hvp_flat)
        second_derivative = - (grad_flat / denom) * numerator

        if return_hvp:
            return second_derivative, hvp_flat
        else:
            return second_derivative

    def get_gradient_value_in_specific_point(self, current_point):
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)

        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grad_data_fitting_term = self.compute_grad_data_fitting_term(current_point, x, y)

        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}
        total_grad_flat = self._parameters_to_vector(total_grad)
        return total_grad_flat

    def _vector_to_parameters(self, vector):
        """
        Converts a flat vector into a parameter dict matching the model's parameters.
        """
        params_dict = {}
        pointer = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            params_dict[name] = vector[pointer:pointer + num_params].view_as(param).to(self.device)
            pointer += num_params
        return params_dict

    def _parameters_to_vector(self, params):
        """
        Converts a parameter dict into a flat vector.
        """
        return torch.cat([p.reshape(-1) for p in params.values()])


class LinearizedRegressionManifold:
    def __init__(self, model, X, y, f_MAP, J_f_MAP, theta_MAP, batching=False, lambda_reg=None, noise_var=1, device="cpu"):
        self.model = model.to(device)
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.device = device
        self.f_MAP = f_MAP.to(device)
        self.J_f_MAP = J_f_MAP.to(device)
        self.theta_MAP = theta_MAP.to(device)

        assert y is None if batching else True, "If batching is True, y should be None"

        # Convert theta_MAP to parameter dict
        self.theta_MAP_params = self._vector_to_parameters(self.theta_MAP)

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, params):
        """
        L2 regularization. Separate from the loss for gradient computation.
        """
        w_norm = sum((p ** 2).sum() for p in params.values())
        return self.lambda_reg * w_norm

    def mse_loss(self, params, x, y, f_MAP):
        x = x.to(self.device)
        y = y.to(self.device)

        def predict(params):
            return functional_call(self.model, params, x)

        # Compute difference between params and theta_MAP_params
        delta_params = {k: params[k] - self.theta_MAP_params[k] for k in params.keys()}

        # Compute jvp_value
        _, jvp_value = jvp(predict, (self.theta_MAP_params,), (delta_params,))

        y_pred = f_MAP + jvp_value

        loss = (0.5 / self.noise_var) * nn.MSELoss(reduction='sum')(y_pred, y)

        return loss

    def compute_grad(self, params, x, y, f_MAP):
        # Define a loss function that only depends on params
        def loss_fn(params):
            return self.mse_loss(params, x, y, f_MAP)

        grad_fn = grad(loss_fn)
        grads = grad_fn(params)
        return grads

    def compute_grad_L2_reg(self, params):
        reg_fn = lambda params: self.L2_norm(params)
        grad_fn = grad(reg_fn)
        grads = grad_fn(params)
        return grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # Convert inputs to parameter dicts if they are vectors
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)
        if not isinstance(velocity, dict):
            velocity = self._vector_to_parameters(velocity)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)
            f_MAP = self.f_MAP.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y, batch_f_MAP in self.X:
                grads = self.compute_grad(current_point, batch_x, batch_y, batch_f_MAP)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad(current_point, x, y, f_MAP)
            grad_data_fitting_term = grads

        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}

        # Compute Hessian-vector product (HVP)
        def loss_fn(params):
            return self.mse_loss(params, x, y, f_MAP) + self.L2_norm(params)

        hvp_fn = lambda params, vec: hvp(loss_fn, params, vec)[1]
        hvp_result = hvp_fn(current_point, velocity)

        # Convert to flat vectors
        grad_flat = self._parameters_to_vector(total_grad)
        hvp_flat = self._parameters_to_vector(hvp_result)
        velocity_flat = self._parameters_to_vector(velocity)

        # Compute second derivative
        denom = 1 + grad_flat.dot(grad_flat)
        numerator = grad_flat.dot(hvp_flat)
        second_derivative = - (grad_flat / denom) * numerator

        if return_hvp:
            return second_derivative.detach(), hvp_flat.detach()
        else:
            return second_derivative.detach()

    def get_gradient_value_in_specific_point(self, current_point):
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)

        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)
            f_MAP = self.f_MAP.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y, batch_f_MAP in self.X:
                grads = self.compute_grad(current_point, batch_x, batch_y, batch_f_MAP)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad(current_point, x, y, f_MAP)
            grad_data_fitting_term = grads

        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}
        total_grad_flat = self._parameters_to_vector(total_grad)

        return total_grad_flat

    def _vector_to_parameters(self, vector):
        """
        Converts a flat vector into a parameter dict matching the model's parameters.
        """
        params_dict = {}
        pointer = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            params_dict[name] = vector[pointer:pointer + num_params].view_as(param).to(self.device)
            pointer += num_params
        return params_dict

    def _parameters_to_vector(self, params):
        """
        Converts a parameter dict into a flat vector.
        """
        return torch.cat([p.reshape(-1) for p in params.values()])


class CrossEntropyManifold:
    """
    Manifold for models trained with cross-entropy loss, supporting geodesic computations.
    """

    def __init__(self, model, X, y=None, batching=False, device="cpu", lambda_reg=None, model_type="fc", N=None, B1=None, B2=None):
        self.model = model.to(device)
        self.X = X
        self.y = y
        self.N = N if N is not None else len(self.X)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.batching = batching
        self.device = device
        self.model_type = model_type
        self.lambda_reg = lambda_reg

        assert y is None if batching else True, "If batching is True, y should be None"

        # Scaling factor for loss when using batches
        self.B1 = B1
        self.B2 = B2
        self.factor = None
        if self.B1 is not None:
            self.factor = self.N / self.B1
            if self.B2 is not None:
                self.factor *= (self.B2 / self.B1)

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, params):
        """
        L2 regularization term.
        """
        w_norm = sum((p ** 2).sum() for p in params.values())
        return self.lambda_reg * w_norm

    def CE_loss(self, params, x, y):
        """
        Cross-entropy loss function.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if self.model_type != "fc":
            x = x.unsqueeze(1)
        y_pred = functional_call(self.model, params, x)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(y_pred, y)
        if self.factor is not None:
            loss *= self.factor
        return loss

    def compute_grad_data_fitting_term(self, params, x, y):
        """
        Compute gradient of the data fitting term.
        """
        loss_fn = lambda p: self.CE_loss(p, x, y)
        grad_fn = grad(loss_fn)
        grads = grad_fn(params)
        return grads

    def compute_grad_L2_reg(self, params):
        """
        Compute gradient of the L2 regularization term.
        """
        reg_fn = lambda p: self.L2_norm(p)
        grad_fn = grad(reg_fn)
        grads = grad_fn(params)
        return grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        """
        Compute the geodesic system for the manifold.
        """
        # Convert inputs to parameter dictionaries
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)
        if not isinstance(velocity, dict):
            velocity = self._vector_to_parameters(velocity)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad_data_fitting_term(current_point, x, y)
            grad_data_fitting_term = grads

        # Compute gradient of L2 regularization term
        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}

        # Compute Hessian-vector product
        if batchify:
            hvp_data_fitting = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y in self.X:
                def loss_fn(p):
                    return self.CE_loss(p, batch_x, batch_y)
                hvp_fn = lambda p, v: hvp(loss_fn, p, v)[1]
                hvp_batch = hvp_fn(current_point, velocity)
                for k in hvp_data_fitting.keys():
                    hvp_data_fitting[k] += hvp_batch[k]
        else:
            def loss_fn(p):
                return self.CE_loss(p, x, y)
            hvp_fn = lambda p, v: hvp(loss_fn, p, v)[1]
            hvp_data_fitting = hvp_fn(current_point, velocity)

        # Add Hessian-vector product of regularization term
        if self.lambda_reg is not None:
            hvp_reg = {k: 2 * self.lambda_reg * velocity[k] for k in velocity.keys()}
            tot_hvp = {k: hvp_data_fitting[k] + hvp_reg[k] for k in hvp_data_fitting.keys()}
        else:
            tot_hvp = hvp_data_fitting

        # Convert gradients and HVPs to flat vectors
        grad_flat = self._parameters_to_vector(total_grad)
        hvp_flat = self._parameters_to_vector(tot_hvp)
        velocity_flat = self._parameters_to_vector(velocity)

        # Compute second derivative
        denom = 1 + grad_flat.dot(grad_flat)
        numerator = grad_flat.dot(hvp_flat)
        second_derivative = - (grad_flat / denom) * numerator

        if return_hvp:
            return second_derivative.detach(), hvp_flat.detach()
        else:
            return second_derivative.detach()

    def get_gradient_value_in_specific_point(self, current_point):
        """
        Compute the gradient at a specific point.
        """
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad_data_fitting_term(current_point, x, y)
            grad_data_fitting_term = grads

        # Compute gradient of L2 regularization term
        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}
        total_grad_flat = self._parameters_to_vector(total_grad)

        return total_grad_flat

    def _vector_to_parameters(self, vector):
        """
        Converts a flat vector into a parameter dict matching the model's parameters.
        """
        params_dict = {}
        pointer = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            params_dict[name] = vector[pointer:pointer + num_params].view_as(param).to(self.device)
            pointer += num_params
        return params_dict

    def _parameters_to_vector(self, params):
        """
        Converts a parameter dict into a flat vector.
        """
        return torch.cat([p.reshape(-1) for p in params.values()])


class LinearizedCrossEntropyManifold:
    """
    Manifold for linearized models trained with cross-entropy loss, supporting geodesic computations.
    """

    def __init__(self, model, X, y, f_MAP, theta_MAP, batching=False, device="cpu", lambda_reg=None, model_type="fc", N=None, B1=None, B2=None):
        self.model = model.to(device)
        self.X = X
        self.y = y
        self.N = N if N is not None else len(self.X)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.batching = batching
        self.device = device
        self.model_type = model_type
        self.lambda_reg = lambda_reg

        assert y is None if batching else True, "If batching is True, y should be None"

        self.theta_MAP = theta_MAP.to(self.device)
        self.f_MAP = f_MAP.to(self.device)
        self.theta_MAP_params = self._vector_to_parameters(self.theta_MAP)

        # Scaling factor for loss when using batches
        self.B1 = B1
        self.B2 = B2
        self.factor = None
        if self.B1 is not None:
            self.factor = self.N / self.B1
            if self.B2 is not None:
                self.factor *= (self.B2 / self.B1)

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, params):
        """
        L2 regularization term.
        """
        w_norm = sum((p ** 2).sum() for p in params.values())
        return self.lambda_reg * w_norm

    def CE_loss(self, params, x, y, f_MAP):
        """
        Cross-entropy loss function for linearized models.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if self.model_type != "fc":
            x = x.unsqueeze(1)

        def predict(p):
            return functional_call(self.model, p, x)

        # Compute difference between params and theta_MAP_params
        delta_params = {k: params[k] - self.theta_MAP_params[k] for k in params.keys()}

        # Compute jvp_value
        _, jvp_value = jvp(predict, (self.theta_MAP_params,), (delta_params,))

        y_pred = f_MAP + jvp_value
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss = criterion(y_pred, y)
        if self.factor is not None:
            loss *= self.factor
        return loss

    def compute_grad_data_fitting_term(self, params, x, y, f_MAP):
        """
        Compute gradient of the data fitting term.
        """
        loss_fn = lambda p: self.CE_loss(p, x, y, f_MAP)
        grad_fn = grad(loss_fn)
        grads = grad_fn(params)
        return grads

    def compute_grad_L2_reg(self, params):
        """
        Compute gradient of the L2 regularization term.
        """
        reg_fn = lambda p: self.L2_norm(p)
        grad_fn = grad(reg_fn)
        grads = grad_fn(params)
        return grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        """
        Compute the geodesic system for the manifold.
        """
        # Convert inputs to parameter dictionaries
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)
        if not isinstance(velocity, dict):
            velocity = self._vector_to_parameters(velocity)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)
            f_MAP = self.f_MAP.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y, batch_f_MAP in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y, batch_f_MAP)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad_data_fitting_term(current_point, x, y, f_MAP)
            grad_data_fitting_term = grads

        # Compute gradient of L2 regularization term
        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}

        # Compute Hessian-vector product
        if batchify:
            hvp_data_fitting = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y, batch_f_MAP in self.X:
                def loss_fn(p):
                    return self.CE_loss(p, batch_x, batch_y, batch_f_MAP)
                hvp_fn = lambda p, v: hvp(loss_fn, p, v)[1]
                hvp_batch = hvp_fn(current_point, velocity)
                for k in hvp_data_fitting.keys():
                    hvp_data_fitting[k] += hvp_batch[k]
        else:
            def loss_fn(p):
                return self.CE_loss(p, x, y, f_MAP)
            hvp_fn = lambda p, v: hvp(loss_fn, p, v)[1]
            hvp_data_fitting = hvp_fn(current_point, velocity)

        # Add Hessian-vector product of regularization term
        if self.lambda_reg is not None:
            hvp_reg = {k: 2 * self.lambda_reg * velocity[k] for k in velocity.keys()}
            tot_hvp = {k: hvp_data_fitting[k] + hvp_reg[k] for k in hvp_data_fitting.keys()}
        else:
            tot_hvp = hvp_data_fitting

        # Convert gradients and HVPs to flat vectors
        grad_flat = self._parameters_to_vector(total_grad)
        hvp_flat = self._parameters_to_vector(tot_hvp)
        velocity_flat = self._parameters_to_vector(velocity)

        # Compute second derivative
        denom = 1 + grad_flat.dot(grad_flat)
        numerator = grad_flat.dot(hvp_flat)
        second_derivative = - (grad_flat / denom) * numerator

        if return_hvp:
            return second_derivative.detach(), hvp_flat.detach()
        else:
            return second_derivative.detach()

    def get_gradient_value_in_specific_point(self, current_point):
        """
        Compute the gradient at a specific point.
        """
        if not isinstance(current_point, dict):
            current_point = self._vector_to_parameters(current_point)

        # Prepare data
        if isinstance(self.X, DataLoader):
            batchify = True
        else:
            batchify = False
            x, y = self.X.to(self.device), self.y.to(self.device)
            f_MAP = self.f_MAP.to(self.device)

        # Compute gradients
        if batchify:
            grad_data_fitting_term = {k: torch.zeros_like(v) for k, v in current_point.items()}
            for batch_x, batch_y, batch_f_MAP in self.X:
                grads = self.compute_grad_data_fitting_term(current_point, batch_x, batch_y, batch_f_MAP)
                for k in grad_data_fitting_term.keys():
                    grad_data_fitting_term[k] += grads[k]
        else:
            grads = self.compute_grad_data_fitting_term(current_point, x, y, f_MAP)
            grad_data_fitting_term = grads

        # Compute gradient of L2 regularization term
        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(current_point)
        else:
            grad_reg = {k: torch.zeros_like(v) for k, v in current_point.items()}

        # Total gradient
        total_grad = {k: grad_data_fitting_term[k] + grad_reg[k] for k in grad_data_fitting_term.keys()}
        total_grad_flat = self._parameters_to_vector(total_grad)

        return total_grad_flat

    def _vector_to_parameters(self, vector):
        """
        Converts a flat vector into a parameter dict matching the model's parameters.
        """
        params_dict = {}
        pointer = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            params_dict[name] = vector[pointer:pointer + num_params].view_as(param).to(self.device)
            pointer += num_params
        return params_dict

    def _parameters_to_vector(self, params):
        """
        Converts a parameter dict into a flat vector.
        """
        return torch.cat([p.reshape(-1) for p in params.values()])