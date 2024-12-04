import numpy as np
from sklearn import metrics
import jax.numpy as jnp

# Note: For calibration metrics (ECE and MCE), we'll implement them directly
# since we might not have an equivalent library in JAX.


def accuracy(y_pred, y_true):
    return jnp.mean(jnp.argmax(y_pred, axis=1) == y_true) * 100


def nll(y_pred, y_true):
    return metrics.log_loss(y_true, np.array(y_pred))


def brier(y_pred, y_true):
    def one_hot(targets, nb_classes):
        targets = targets.astype(int)
        res = jnp.eye(nb_classes)[targets.reshape(-1)]
        return res.reshape(targets.shape + (nb_classes,))

    y_one_hot = one_hot(y_true, y_pred.shape[-1])
    return metrics.mean_squared_error(np.array(y_pred), np.array(y_one_hot))
