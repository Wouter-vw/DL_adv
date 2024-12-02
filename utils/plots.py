import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


def plot_map_confidence(x_train, y_train, XX1, XX2, conf, title="Confidence MAP", save_path=None):
    """Plots MAP confidence with training data."""
    plt.contourf(
        XX1,
        XX2,
        conf.reshape(XX1.shape),
        alpha=0.8,
        antialiased=True,
        cmap="Blues",
        levels=jnp.arange(0.0, 1.01, 0.1),
    )
    plt.colorbar()
    plt.scatter(
        x_train[:, 0][y_train == 0],
        x_train[:, 1][y_train == 0],
        c="orange",
        edgecolors="black",
        s=45,
        alpha=1,
    )
    plt.scatter(
        x_train[:, 0][y_train == 1],
        x_train[:, 1][y_train == 1],
        c="violet",
        edgecolors="black",
        s=45,
        alpha=1,
    )
    plt.title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_confidence(x_train, y_train, XX1, XX2, conf, contours, title="Confidence", save_path=None):
    """Plots confidence with training data."""
    plt.contourf(
        XX1,
        XX2,
        conf.reshape(XX1.shape),
        alpha=0.8,
        antialiased=True,
        cmap="Blues",
        levels=np.arange(0.0, 1.01, 0.1),
        zorder=-10,
    )
    plt.scatter(
        x_train[:, 0][y_train == 0],
        x_train[:, 1][y_train == 0],
        c="orange",
        edgecolors="black",
        s=45,
        alpha=1,
        zorder=10,
    )
    plt.scatter(
        x_train[:, 0][y_train == 1],
        x_train[:, 1][y_train == 1],
        c="violet",
        edgecolors="black",
        s=45,
        alpha=1,
        zorder=10,
    )
    plt.contour(
        XX1, XX2, contours.reshape(XX1.shape), levels=[0.5], colors="k", alpha=0.5, zorder=0
    )
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
