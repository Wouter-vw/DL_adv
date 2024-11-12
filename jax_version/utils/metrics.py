"""
Metrics computation adapted to JAX and Flax.
"""

import numpy as np
from sklearn import metrics
import jax
import jax.numpy as jnp
import math
from functools import partial

# Note: For calibration metrics (ECE and MCE), we'll implement them directly
# since we might not have an equivalent library in JAX.

def accuracy(y_pred, y_true):
    """
    Computes the accuracy of predictions.
    """
    return jnp.mean(jnp.argmax(y_pred, axis=1) == y_true) * 100


def nll(y_pred, y_true):
    """
    Mean Categorical Negative Log-Likelihood.
    `y_pred` is a probability vector.
    """
    return metrics.log_loss(y_true, np.array(y_pred))


def brier(y_pred, y_true):
    """
    Computes the Brier score.
    """
    def one_hot(targets, nb_classes):
        targets = targets.astype(int)
        res = jnp.eye(nb_classes)[targets.reshape(-1)]
        return res.reshape(targets.shape + (nb_classes,))
    y_one_hot = one_hot(y_true, y_pred.shape[-1])
    return metrics.mean_squared_error(np.array(y_pred), np.array(y_one_hot))


def calibration(pys, y_true, M=15):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    # For binary classification, extract the positive class probability
    if jnp.max(y_true) == 1:
        pys = pys[:, 1]

    probs = pys
    labels = y_true

    # If probs are 2D, get the max prob and predicted labels
    if probs.ndim == 2:
        confidences = jnp.max(probs, axis=1)
        predictions = jnp.argmax(probs, axis=1)
    else:
        # For binary case
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    n = labels.shape[0]
    ece = 0.0
    mce = 0.0

    bin_boundaries = jnp.linspace(0.0, 1.0, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = jnp.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = jnp.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = jnp.mean(confidences[in_bin])
            bin_error = jnp.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += bin_error * prop_in_bin
            mce = jnp.maximum(mce, bin_error)
    return ece * 100, mce * 100


def nlpd_using_predictions(mu_star, var_star, true_target):
    """
    Computes the Negative Log Predictive Density (NLPD) for regression.
    """
    nlpd = jnp.abs(
        0.5 * jnp.log(2 * jnp.pi) + 0.5 * jnp.mean(jnp.log(var_star) + (true_target - mu_star) ** 2 / var_star)
    )
    return nlpd


def mae(mu_star, true_target):
    """
    Computes the Mean Absolute Error (MAE) for regression.
    """
    return jnp.mean(jnp.abs(true_target - mu_star))


def rmse(mu_star, true_target):
    """
    Computes the Root Mean Square Error (RMSE) for regression.
    """
    return jnp.sqrt(jnp.mean((true_target - mu_star) ** 2))


def error_metrics(mu_star, var_star, true_target):
    """
    Computes RMSE, MAE, and NLPD for regression.
    """
    _rmse = rmse(mu_star, true_target)
    _mae = mae(mu_star, true_target)
    _nlpd = nlpd_using_predictions(mu_star, var_star, true_target)
    return _rmse, _mae, _nlpd


def compute_metrics(model, weights_list, test_data, n_posterior_samples, calibration_bins=15, verbose=True, save=None):
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Get the unravel function to reconstruct the parameters
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, X_test[0])
    initial_params = variables['params']
    _, unravel_fn = jax.flatten_util.ravel_pytree(initial_params)

    metrics_dict = {"accuracy": [], "nll": [], "brier": [], "ece": [], "mce": []}

    for weights in weights_list:
        p_y_test = 0
        for s in range(n_posterior_samples):
            flat_weights = weights[s]
            params = unravel_fn(flat_weights)
            # Compute the predictions
            logits = model.apply({'params': params}, X_test)
            p_y_test += jax.nn.softmax(logits, axis=-1)

        p_y_test /= n_posterior_samples

        _accuracy = accuracy(p_y_test, y_test)
        _nll = nll(p_y_test, y_test)
        _brier = brier(p_y_test, y_test)
        _ece, _mce = calibration(p_y_test, y_test, M=calibration_bins)

        metrics_dict["accuracy"].append(_accuracy)
        metrics_dict["nll"].append(_nll)
        metrics_dict["brier"].append(_brier)
        metrics_dict["ece"].append(_ece)
        metrics_dict["mce"].append(_mce)

    if verbose:
        print_metrics(metrics_dict)

    if save is not None:
        np.save(save + "_metrics.npy", metrics_dict)
    else:
        print_metrics(metrics_dict)


def compute_metrics_per_sample(model, weights_list, test_data, n_posterior_samples, calibration_bins=15, verbose=True):
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Get the unravel function
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, X_test[0])
    initial_params = variables['params']
    _, unravel_fn = jax.flatten_util.ravel_pytree(initial_params)

    metrics_list = []
    for weights in weights_list:
        metrics_dict = {"accuracy": [], "nll": [], "brier": [], "ece": [], "mce": []}

        for s in range(n_posterior_samples):
            flat_weights = weights[s]
            params = unravel_fn(flat_weights)
            # Compute the predictions
            logits = model.apply({'params': params}, X_test)
            p_y_test = jax.nn.softmax(logits, axis=-1)

            _accuracy = accuracy(p_y_test, y_test)
            _nll = nll(p_y_test, y_test)
            _brier = brier(p_y_test, y_test)
            _ece, _mce = calibration(p_y_test, y_test, M=calibration_bins)

            metrics_dict["accuracy"].append(_accuracy)
            metrics_dict["nll"].append(_nll)
            metrics_dict["brier"].append(_brier)
            metrics_dict["ece"].append(_ece)
            metrics_dict["mce"].append(_mce)

        metrics_list.append(metrics_dict)

    if verbose:
        print_metrics_per_sample(metrics_list)

    return metrics_list


def print_metrics(metrics_dict):
    for metric, m_list in metrics_dict.items():
        print(f"> {metric}: {m_list}")


def print_metrics_per_sample(metrics_list):
    for metrics_dict in metrics_list:
        for metric, m_list in metrics_dict.items():
            print(f"> {metric}: {m_list}")


### Test
# N = 20  # number of samples
# C = 3   # number of classes

# y_true = jnp.array([2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 0, 1, 2, 2, 0, 1, 0, 2, 1, 0])

# # Fixed p_y_test values as a tensor of probabilities, each row summing to 1
# y_pred = jnp.array([
#     [0.607, 0.175, 0.218],
#     [0.079, 0.849, 0.072],
#     [0.199, 0.434, 0.367],
#     [0.288, 0.292, 0.420],
#     [0.210, 0.512, 0.278],
#     [0.467, 0.324, 0.209],
#     [0.335, 0.275, 0.390],
#     [0.215, 0.118, 0.667],
#     [0.754, 0.169, 0.077],
#     [0.162, 0.491, 0.347],
#     [0.303, 0.609, 0.088],
#     [0.135, 0.488, 0.377],
#     [0.145, 0.279, 0.576],
#     [0.266, 0.116, 0.618],
#     [0.554, 0.343, 0.103],
#     [0.224, 0.574, 0.202],
#     [0.690, 0.245, 0.065],
#     [0.121, 0.212, 0.667],
#     [0.327, 0.411, 0.262],
#     [0.515, 0.269, 0.216]
# ])


# print("Accuracy: ", accuracy(y_pred, y_true))
# print("NLL: ", nll(y_pred, y_true))
# print("Brier: ", brier(y_pred, y_true))
# _ece, _mce = calibration(y_pred, y_true)
# print("ECE: ", _ece)
# print("MCE: ", _mce)

### Results
# Accuracy:  80.0
# NLL:  0.8151565860712264
# Brier:  0.15734966
# ECE:  44.265003
# MCE:  84.899994