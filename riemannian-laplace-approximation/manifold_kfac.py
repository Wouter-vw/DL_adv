"""
File containing all the manifold we are going to use for the experiments:
- Regression manifold
- Linearized regression manifold
- cross entropy manifold
- linearized cross entropy manifold
"""

import torch
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functools import partial
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2, stack_gradient3
from torch.distributions import Normal
from torch import nn
import numpy as np
import math
import time
import copy

class cross_entropy_manifold:
    """
    Also in this case I have to split the gradient loss computation and the gradient of the regularization
    term.
    This is needed to get the correct gradient and hessian computation when using batches.
    """

    def __init__(
        self, model, X, y, batching=False, device="cpu", lambda_reg=None, type="fc", N=None, B1=None, B2=None
    ):
        self.model = model

        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.device = device
        self.type = type
        # self.prior_precision = prior_precision
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

        ## stuff we need when using barches
        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None

        # here I can already compute the factor_loss
        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)

    @staticmethod
    def is_diagonal():
        return False
    
    def register_hooks(self, model):
        activations = {}

        def save_activation(name):
            def hook(module, input, output):
                activations[name] = input[0]
            return hook

        # Register hooks only for Linear layers
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):  # Only register hooks for Linear layers
                layer.register_forward_hook(save_activation(name))
    
        return activations

    def compute_gradients(self, model, input, target, criterion):
        model.zero_grad()
        output = model(input)
        loss = self.CE_loss(list(model.parameters()), (input, target), self.f_MAP) 
        loss.backward()
        
        # Extract only gradients for weights (exclude biases)
        gradients = {name: param.grad for name, param in model.named_parameters()}
        
        return gradients, loss
    
    def compute_kfac_approximation(self, gradients, activations):
        """Compute Kronecker factors for each layer, including biases."""
        kfac_factors = {}
        
        for layer_name, grad in gradients.items():
            act_name = layer_name  # Bias or weight layer should have the same name
            
            # Check if the layer is a weight or a bias
            if "weight" in layer_name:
                activation = activations[layer_name.split('.')[0]]  # Fetch the corresponding activation for weights
                # Compute Kronecker factors for weights
                G = torch.matmul(grad.t(), grad) / activation.shape[0]  # Gradient outer product (approximation)
                A = torch.matmul(activation.t(), activation) / activation.shape[0]  # Activation outer product (approximation)

                kfac_factors[layer_name] = (G, A)
                
                # Now add the corresponding bias
                bias_name = layer_name.replace('weight', 'bias')
                bias_grad = gradients[bias_name]
                
                # Reshape bias gradient to be a column vector of shape [16, 1]
                G_bias = bias_grad
                
                # Bias activation is just a vector of ones (shape [N, 1])
                A_bias = torch.ones(bias_grad.shape[0])  # Bias activation is a vector of ones
                kfac_factors[bias_name] = (G_bias, A_bias)
        
        return kfac_factors
    
    def compute_kfac_vector_product(self, kfac_factors, velocity):
        """Compute the KFAC approximation of the Hessian-vector product with L2 regularization and biases."""
        hvp_list = []
        
        for layer_name, (G, A) in kfac_factors.items():
            if "bias" in layer_name:
                # Handle bias separately: We assume bias gradients are scalar, so A is a scalar.
                v = velocity[layer_name].view(-1)  # Bias velocity is a vector (since it's scalar)
                v_out = G * v  # For bias, no matrix multiplication, just scaling by A (which is 1)
            else:
                # Reshape velocity to match layer dimensions for weight layers
                v = velocity[layer_name].view(A.shape[1], -1)  # Reshape to match the layer dimensions
                
                # Perform KFAC HVP for weights
                v_out = torch.matmul(A, torch.matmul(G, v))
            
            
            # Flatten and store
            hvp_list.append(v_out.view(-1))
        
        # Concatenate all layer results into a single vector
        return torch.cat(hvp_list)


    def CE_loss(self, param, data):
        """
        Data fitting term of the loss
        """
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        if self.type != "fc":
            # assuming input for con
            x = x.unsqueeze(1)

        if self.model is None:
            raise NotImplementedError("Compute usual prediction still have to be implemented")
        else:
            # self.fmodel.eval()
            y_pred = self.fmodel(param, self.buffers, x)

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if self.type == "fc":
            if self.factor is not None:
                return self.factor * criterion(y_pred, y)
            else:
                return criterion(y_pred, y)
        else:
            if self.factor is not None:
                return self.factor * criterion(y_pred.view(-1), y)
            else:
                return criterion(y_pred.view(-1), y)

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def compute_grad_data_fitting_term(self, params, data):
        # TODO: understand how to make vmap work without passing the data
        ft_compute_grad = grad(self.CE_loss)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params, data)
        return ft_per_sample_grads

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)

        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        # I would expect both current point and velocity to be
        # two vectors of shape n_params
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        # and I have to reshape the velocity into being the same structure as the params
        vel_as_params = get_params_structure(velocity, params)

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            grad_per_example = self.compute_grad_data_fitting_term(params, data)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # now I have to compute the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        # Register hooks to capture activations from linear layers
        activations = self.register_hooks(self.model)


        vel_as_params = get_params_structure(velocity, params)
        pointer = 0
        velocity_reconstructed = {}

        for name, param in self.model.named_parameters():
            num_param = param.numel()  # Number of elements in the parameter tensor
            velocity_reconstructed[name] = velocity[pointer:pointer + num_param].view(param.shape)
            pointer += num_param


        # now I have also to compute the Hvp between hessian and velocity
        start = time.time()
        if batchify:
            hvp_data_fitting = 0
            for batch_img, batch_label in self.X:
                if self.type == "fc":
                    # Compute Gradients (and loss)
                    gradients, _ = self.compute_gradients(self.model, batch_img, batch_label, nn.CrossEntropyLoss(reduction = 'sum'))
                    kfac_factors = self.compute_kfac_approximation(gradients, activations)
                    # Compute HVP with KFAC
                    hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)
                else:
                    print("If you are getting an error, before here I was using self.CE_loss2, so double check that")
                    # Compute Gradients (and loss)
                    gradients, _ = self.compute_gradients(self.model, batch_img, batch_label, nn.CrossEntropyLoss(reduction = 'sum'))
                    kfac_factors = self.compute_kfac_approximation(gradients, activations)
                    # Compute HVP with KFAC
                    hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)
        else:
            # Compute Gradients (and loss)
            gradients, _ = self.compute_gradients(self.model, data[0], data[1], nn.CrossEntropyLoss(reduction = 'sum'))
            kfac_factors = self.compute_kfac_approximation(gradients, activations)
            # Compute HVP with KFAC
            hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)


        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity
            tot_hvp = hvp_kfac + hvp_reg.view(-1)
        else:
            tot_hvp = hvp_kfac

        tot_hvp = tot_hvp.to(self.device)
        tot_gradient = tot_gradient.to(self.device)
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().cpu().numpy(), tot_hvp.view(-1, 1).detach().cpu().numpy()
        else:
            return second_derivative.view(-1, 1).detach().cpu().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # method to return the gradient of the loss in a specific point
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"
        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if self.batching:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            data = (self.X, self.y)
            grad_per_example = self.compute_grad_data_fitting_term(params, data)

            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)

        # now I have to compute the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient

### Check what parameters to pass to the new hvp!!
class linearized_cross_entropy_manifold:
    """
    Also in this case I have to separate data fitting term and regularization term for gradient and
    hessian computation in case of batches.
    """

    def __init__(
        self,
        model,
        X,
        y,
        f_MAP,
        theta_MAP,
        batching=False,
        device="cpu",
        lambda_reg=None,
        type="fc",
        N=None,
        B1=None,
        B2=None,
    ):
        self.model = model
        # TODO: decide if it is better to pass X and Y or
        # pass data that is either data = (X,y) or a dataloader
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.device = device
        self.type = type
        # self.prior_precision = prior_precision
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

        self.fmodel_map = None
        self.params_map = None
        self.buffers_map = None

        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None

        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)

    @staticmethod
    def is_diagonal():
        return False
     
    def register_hooks(self, model):
        activations = {}

        def save_activation(name):
            def hook(module, input, output):
                activations[name] = input[0]
            return hook

        # Register hooks only for Linear layers
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):  # Only register hooks for Linear layers
                layer.register_forward_hook(save_activation(name))
    
        return activations

    # Function to compute gradients (weights only)
    ## Adjust to more closely reflect CE LOSS (is it necessary)
    def compute_gradients(self, model, input, target, criterion):
        model.zero_grad()
        output = model(input)
        loss = self.CE_loss(list(model.parameters()), (input, target), self.f_MAP) 
        loss.backward()
        
        # Extract only gradients for weights (exclude biases)
        gradients = {name: param.grad for name, param in model.named_parameters()}
        
        return gradients, loss
    
    def compute_kfac_approximation(self, gradients, activations):
        """Compute Kronecker factors for each layer, including biases."""
        kfac_factors = {}
        
        for layer_name, grad in gradients.items():
            act_name = layer_name  # Bias or weight layer should have the same name
            
            # Check if the layer is a weight or a bias
            if "weight" in layer_name:
                activation = activations[layer_name.split('.')[0]]  # Fetch the corresponding activation for weights
                # Compute Kronecker factors for weights
                G = torch.matmul(grad.t(), grad) / activation.shape[0]  # Gradient outer product (approximation)
                A = torch.matmul(activation.t(), activation) / activation.shape[0]  # Activation outer product (approximation)

                kfac_factors[layer_name] = (G, A)
                
                # Now add the corresponding bias
                bias_name = layer_name.replace('weight', 'bias')
                bias_grad = gradients[bias_name]
                
                # Reshape bias gradient to be a column vector of shape [16, 1]
                G_bias = bias_grad
                
                # Bias activation is just a vector of ones (shape [N, 1])
                A_bias = torch.ones(bias_grad.shape[0])  # Bias activation is a vector of ones
                kfac_factors[bias_name] = (G_bias, A_bias)
        
        return kfac_factors
    
    def compute_kfac_vector_product(self, kfac_factors, velocity):
        """Compute the KFAC approximation of the Hessian-vector product with L2 regularization and biases."""
        hvp_list = []
        
        for layer_name, (G, A) in kfac_factors.items():
            if "bias" in layer_name:
                # Handle bias separately: We assume bias gradients are scalar, so A is a scalar.
                v = velocity[layer_name].view(-1)  # Bias velocity is a vector (since it's scalar)
                v_out = G * v  # For bias, no matrix multiplication, just scaling by A (which is 1)
            else:
                # Reshape velocity to match layer dimensions for weight layers
                v = velocity[layer_name].view(A.shape[1], -1)  # Reshape to match the layer dimensions
                
                # Perform KFAC HVP for weights
                v_out = torch.matmul(A, torch.matmul(G, v))
            
            
            # Flatten and store
            hvp_list.append(v_out.view(-1))
        
        # Concatenate all layer results into a single vector
        return torch.cat(hvp_list)

    def CE_loss(self, param, data, f_MAP):
        """
        Data fitting term of the loss
        """

        def predict(params, datas):
            y_preds = self.fmodel_map(params, self.buffers_map, datas)
            return y_preds

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        if self.type != "fc":
            x = x.unsqueeze(1)

        params_map = get_params_structure(self.theta_MAP, param)
        diff_weights = []
        for i in range(len(param)):
            diff_weights.append(param[i] - self.params_map[i])
        diff_weights = tuple(diff_weights)
        _, jvp_value = jvp(predict, (params_map, x), (diff_weights, torch.zeros_like(x)), strict=False)

        y_pred = f_MAP + jvp_value

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if self.type == "fc":
            if self.factor is not None:
                return self.factor * criterion(y_pred, y)
            else:
                return criterion(y_pred, y)
        else:
            if self.factor is not None:
                return self.factor * criterion(y_pred.view(-1), y)
            else:
                return criterion(y_pred.view(-1), y)

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def compute_grad_data_fitting_term(self, params, data, f_MAP):
        # TODO: understand how to make vmap work without passing the data
        ft_compute_grad = grad(self.CE_loss)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params, data, f_MAP)
        return ft_per_sample_grads

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            grad_per_example = self.compute_grad_data_fitting_term(params, data, self.f_MAP)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        # Register hooks to capture activations from linear layers
        activations = self.register_hooks(self.model)


        vel_as_params = get_params_structure(velocity, params)
        pointer = 0
        velocity_reconstructed = {}

        for name, param in self.model.named_parameters():
            num_param = param.numel()  # Number of elements in the parameter tensor
            velocity_reconstructed[name] = velocity[pointer:pointer + num_param].view(param.shape)
            pointer += num_param

        start = time.time()
        hvp_data_fitting = 0
        if batchify:
            for batch_img, batch_label, batch_f_MAP in self.X:
                if self.type == "fc":
                    # Compute Gradients (and loss)
                    gradients, _ = self.compute_gradients(self.model, batch_img, batch_label, nn.CrossEntropyLoss(reduction = 'sum'))
                    kfac_factors = self.compute_kfac_approximation(gradients, activations)
                    # Compute HVP with KFAC
                    hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)
                else:
                    # Compute Gradients (and loss)
                    gradients, _ = self.compute_gradients(self.model, batch_img, batch_label, nn.CrossEntropyLoss(reduction = 'sum'))
                    kfac_factors = self.compute_kfac_approximation(gradients, activations)
                    # Compute HVP with KFAC
                    hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)
        else:
                    # Compute Gradients (and loss)
            gradients, _ = self.compute_gradients(self.model, data[0], data[1], nn.CrossEntropyLoss(reduction = 'sum'))
            kfac_factors = self.compute_kfac_approximation(gradients, activations)
            # Compute HVP with KFAC
            hvp_kfac = self.compute_kfac_vector_product(kfac_factors, velocity_reconstructed)

        # I have to add the hvp of the regularization term
        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_kfac + hvp_reg.view(-1)
        else:
            tot_hvp = hvp_kfac

        tot_hvp = tot_hvp.to(self.device)
        tot_gradient = tot_gradient.to(self.device)
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().cpu().numpy(), tot_hvp.view(-1, 1).detach().cpu().numpy()
        else:
            return second_derivative.view(-1, 1).detach().cpu().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # method to return the gradient of the loss in a specific point
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"
        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if batchify:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            grad_per_example = self.compute_grad_data_fitting_term(params, data, self.f_MAP)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient
