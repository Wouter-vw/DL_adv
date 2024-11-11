"""
Helper functions for our geodesic systems using PyTorch 2.0.
"""

import torch
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap


def get_params_structure(vector, true_params):
    """
    Reshapes a flat vector into a structure matching the original parameters.

    Args:
        vector (torch.Tensor): The flat vector containing parameter values.
        true_params (Iterable[torch.Tensor]): The original parameters.

    Returns:
        tuple: A tuple of tensors matching the structure of true_params.
    """
    list_new_weights = []
    pointer = 0
    for sub_weights in true_params:
        num_param = sub_weights.numel()
        my_param = vector[pointer : pointer + num_param].view_as(sub_weights)
        list_new_weights.append(my_param)
        pointer += num_param

    return tuple(list_new_weights)


def stack_gradient(gradients, n_params, n_examples):
    """
    Stacks gradients per example and sums them.

    Args:
        gradients (Iterable[torch.Tensor]): Gradients for each parameter.
        n_params (int): Total number of parameters.
        n_examples (int): Number of examples.

    Returns:
        torch.Tensor: Summed gradients.
    """
    flatten_grad_per_example = torch.zeros((n_examples, n_params), device=gradients[0].device)
    idx = 0
    for g in gradients:
        _g_flat = g.view(n_examples, -1)
        grad_size = _g_flat.shape[1]
        flatten_grad_per_example[:, idx : idx + grad_size] = _g_flat
        idx += grad_size
    return flatten_grad_per_example.sum(0)


def stack_gradient2(gradients, n_params):
    """
    Sums gradients over examples and flattens them.

    Args:
        gradients (Iterable[torch.Tensor]): Gradients for each parameter.
        n_params (int): Total number of parameters.

    Returns:
        torch.Tensor: Flattened and summed gradients.
    """
    grad_flat = torch.zeros(n_params, device=gradients[0].device)
    idx = 0
    for g in gradients:
        b = g.sum(dim=0)
        b_flat = b.flatten()
        grad_size = b_flat.numel()
        grad_flat[idx : idx + grad_size] = b_flat
        idx += grad_size
    return grad_flat


def custom_hvp(f, primals, tangents, strict=False):
    """
    Computes the Hessian-vector product using forward-over-reverse mode.

    Args:
        f (Callable): Function whose Hessian is to be computed.
        primals (Iterable[torch.Tensor]): Input tensors.
        tangents (Iterable[torch.Tensor]): Tangent vectors.
        strict (bool, optional): Whether to use strict mode. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Function output and Hessian-vector product.
    """
    return jvp(grad(f), primals, tangents, strict=strict)


def stack_gradient3(gradients, n_params):
    """
    Flattens and stacks gradients.

    Args:
        gradients (Iterable[torch.Tensor]): Gradients for each parameter.
        n_params (int): Total number of parameters.

    Returns:
        torch.Tensor: Flattened gradients.
    """
    grad_flat = torch.zeros(n_params, device=gradients[0].device)
    idx = 0
    for g in gradients:
        g_flat = g.flatten()
        grad_size = g_flat.numel()
        grad_flat[idx : idx + grad_size] = g_flat
        idx += grad_size
    return grad_flat