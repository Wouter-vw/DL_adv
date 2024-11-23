"""
File containing the banana experiments
"""


import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import sklearn.model_selection
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax2torch import jax2torch
from torch2jax import t2j
from jax.scipy.stats import multivariate_normal
import tensorflow as tf

import flax.linen as nn
from flax import struct
from flax.training import train_state
import optax

from laplace import Laplace
import matplotlib.colors as colors
import seaborn as sns
#####################################
import geomai.utils.geometry as geometry
#import geomai.utils.geometry_diffrax as geometry
####################################################
from torch import nn as nn_torch
########################################
#from manifold import linearized_cross_entropy_manifold
#from manifoldwouter import linearized_cross_entropy_manifold
from manifold_kfac import linearized_cross_entropy_manifold
#########################################
from tqdm import tqdm
import sklearn.datasets
from datautils import make_pinwheel_data
from utils.metrics import accuracy, nll, brier, calibration
from sklearn.metrics import brier_score_loss
import argparse
from torchmetrics.functional.classification import calibration_error
import os

jax.config.update('jax_enable_x64', True)

def main(args):
    palette = sns.color_palette("colorblind")
    print("Linearization?")
    print(args.linearized_pred)
    subset_of_weights = args.subset  # must be 'all'
    hessian_structure = args.structure  #'full' # other possibility is 'diag' or 'full'
    n_posterior_samples = args.samples
    security_check = True
    optimize_prior = args.optimize_prior
    print("Are we optimizing the prior? ", optimize_prior)

    batch_data = args.batch_data

    # run with several seeds
    seed = args.seed
    jrandom.PRNGKey(seed)
    rng = jrandom.PRNGKey(seed)
    torch.manual_seed(seed)
    print("Seed: ", seed)

    shuffle = True
    # now I have to laod the banana dataset
    filen = os.path.join("data", "banana", "banana.csv")
    Xy = np.loadtxt(filen, delimiter=",")
    Xy = jnp.asarray(Xy)
    x_train, y_train = Xy[:, :-1], Xy[:, -1]
    x_test, y_test = Xy[:0, :-1], Xy[:0, -1]
    y_train, y_test = y_train - 1, y_test - 1

    split_train_size = 0.7
    strat = None
    x_full, y_full = jnp.concatenate((x_train, x_test)), jnp.concatenate((y_train, y_test))
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_full, y_full, train_size=split_train_size, random_state=230, shuffle=shuffle, stratify=strat
    )
    x_test, x_valid, y_test, y_valid = sklearn.model_selection.train_test_split(
        x_test, y_test, train_size=0.5, random_state=230, shuffle=shuffle, stratify=strat
    )

    x_train = x_train[:265, :]
    y_train = y_train[:265]

    # define train , valid, test, create data loader for training
    x_train = x_train.astype(jnp.float32)
    x_valid = x_valid.astype(jnp.float32)
    x_test = x_test.astype(jnp.float32)
    # labels are integers
    y_train = y_train.astype(jnp.int32)
    y_valid = y_valid.astype(jnp.int32)
    y_test = y_test.astype(jnp.int32)

    # Review if there is a more efficient method for this!
    # Shuffle and batch manually
    def create_data_loader(x_data, y_data, batch_size, shuffle=True):
        dataset_size = len(x_data)
        indices = jnp.arange(dataset_size)
        
        if shuffle:
            indices = jrandom.permutation(rng, dataset_size, independent=True)
        
        # Create batches
        batches = []
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            batches.append((x_data[batch_idx], y_data[batch_idx]))
        
        return batches
    
    # Create data loaders
    batch_size_train = 1024
    batch_size_valid = 50
    batch_size_test = 50
    
    train_loader = create_data_loader(x_train, y_train, batch_size=batch_size_train, shuffle=True)
    valid_loader = create_data_loader(x_valid, y_valid, batch_size=batch_size_valid, shuffle=False)
    test_loader = create_data_loader(x_test, y_test, batch_size=batch_size_test, shuffle=False)
    
    # Define your model using Flax
    class MLP(nn.Module):
        num_features: int
        hidden_size: int
        num_output: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.hidden_size)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.num_output)(x)
            return x


    # Create training state
    def create_train_state(rng, model, optimizer):
        params = model.init(rng, jnp.ones([1, num_features]))  # Dummy input for parameter initialization
        if optimizer == "sgd":
            learning_rate = 1e-3
            weight_decay = 1e-2
            optimizer = optax.sgd(learning_rate)
        else:
            learning_rate = 1e-3
            weight_decay = 1e-3
            optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

        apply_fn = jax.jit(model.apply)

        return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=optimizer)

        # Loss function
    @jax.jit
    def compute_loss(logits, labels):
        labels = jnp.asarray(labels, dtype=jnp.int32)  # Ensure labels are int32
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))

    # Training function
    @jax.jit
    def train_step(state, batch_img, batch_label):
        def loss_fn(params):
            logits = state.apply_fn(params, batch_img)
            return compute_loss(logits, batch_label)

        # Compute gradients
        grad = jax.grad(loss_fn)(state.params)
        # Update the state with the new parameters
        new_state = state.apply_gradients(grads=grad)
        return new_state

    num_features = x_train.shape[-1]
    num_output = 2
    H = 16
    model = MLP(num_features=num_features, hidden_size=H, num_output=num_output)
    state = create_train_state(rng, model, optimizer=args.optimizer)


    if args.optimizer == "sgd":
        learning_rate = 1e-3
        weight_decay = 1e-2
        optimizer = optax.sgd(learning_rate)
        max_epoch = 2500
    else:
        learning_rate = 1e-3
        weight_decay = 1e-3
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        max_epoch = 1500  
    for epoch in range(max_epoch):
        train_loss = 0.0
    
        # Training step
        for batch_img, batch_label in train_loader:
            state = train_step(state, batch_img, batch_label)
            train_loss += compute_loss(state.apply_fn(state.params, batch_img), batch_label)
        train_loss /= len(x_train)
    
        # Validation step
        val_loss = 0
        val_accuracy = 0
    
        for val_img, val_label in valid_loader:
            logits = state.apply_fn(state.params, val_img)
            val_pred = jnp.argmax(logits, axis=1)
            val_accuracy += jnp.sum(val_pred == val_label)  # Sum correct predictions
    
        val_accuracy /= len(x_valid)  # Calculate accuracy as the fraction of correct predictions
    
        # Print every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val accuracy: {val_accuracy:.4f}")
    
    def get_map_solution(state):
        map_solution, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
        return map_solution, unravel_fn
    
    map_solution, unravel_fn = get_map_solution(state)
    test = unravel_fn(map_solution)
    state = state.replace(params = unravel_fn(map_solution))

    N_grid = 100
    offset = 2
    x1min = x_train[:, 0].min() - offset
    x1max = x_train[:, 0].max() + offset
    x2min = x_train[:, 1].min() - offset
    x2max = x_train[:, 1].max() + offset

    # Create grid using jnp.linspace and jnp.meshgrid
    x_grid = jnp.linspace(x1min, x1max, N_grid)
    y_grid = jnp.linspace(x2min, x2max, N_grid)
    XX1, XX2 = jnp.meshgrid(x_grid, y_grid, indexing='ij')  # Use 'ij' for matrix indexing
    X_grid = jnp.column_stack((XX1.ravel(), XX2.ravel()))

    # Computing and plotting the MAP confidence
    logits_map = state.apply_fn(state.params, X_grid)  # Compute logits
    probs_map = jax.nn.softmax(logits_map)  # Apply softmax

    conf = probs_map.max(1)

    # Plotting
    plt.contourf(
        XX1,
        XX2,
        conf.reshape(N_grid, N_grid),
        alpha=0.8,
        antialiased=True,
        cmap="Blues",
        levels=jnp.arange(0.0, 1.01, 0.1),
    )
    plt.colorbar()
    plt.scatter(
        x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c="orange", edgecolors="black", s=45, alpha=1
    )
    plt.scatter(
        x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c="violet", edgecolors="black", s=45, alpha=1
    )
    plt.title("Confidence MAP")
    plt.xticks([], [])
    plt.yticks([], [])
    # plt.show()


    ## Quick import of pytorch model for the laplace package!
    model_torch= nn_torch.Sequential(
        nn_torch.Linear(num_features, H), torch.nn.Tanh(), nn_torch.Linear(H, H), torch.nn.Tanh(), nn_torch.Linear(H, num_output)
        )

    layer_mapping = {
        'Dense_0': model_torch[0],  # First Linear layer
        'Dense_1': model_torch[2],  # Second Linear layer
        'Dense_2': model_torch[4]   # Third Linear layer
    }

    # Transfer weights and biases
    for flax_layer, torch_layer in layer_mapping.items():
        # Convert and load Flax weights (transpose to match PyTorch)
        weight = torch.tensor(np.array(state.params['params'][flax_layer]['kernel'])).T
        # Ensure the weight tensor is contiguous
        torch_layer.weight.data = weight.contiguous()

        # Convert and load Flax bias (no transpose needed)
        bias = torch.tensor(np.array(state.params['params'][flax_layer]['bias']))
        # Ensure the bias tensor is contiguous
        torch_layer.bias.data = bias.contiguous()

    # We need to define a torch dataloader quickly
    x_torch_train = torch.from_numpy(np.array(x_train))
    y_torch_train = torch.from_numpy(np.array(y_train)).long()
    train_torch_dataset = torch.utils.data.TensorDataset(x_torch_train, y_torch_train)
    train_torch_loader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=265, shuffle=True)

    print("Fitting Laplace")
    la = Laplace(
        model_torch,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        prior_precision=2 * weight_decay,
    )
    la.fit(train_torch_loader)

    if optimize_prior:
        la.optimize_prior_precision(method="marglik")

    print("Prior precision we are using")
    print(la.prior_precision)

    ## Rewritten to use jax where possible, from now on
    ## we seek to only use jax if we don't need external packages
    # and get some samples from it, our initial velocities
    # now I can get some samples for the Laplace approx
        
    if hessian_structure == "diag":
        samples = jax.random.normal(rng, shape=(n_posterior_samples, la.n_params))
        samples = samples * t2j(la.posterior_scale.reshape(1, la.n_params))
        V_LA = samples

    else:
        scale_tril = scale_tril = jnp.array(la.posterior_scale)
        V_LA = jax.random.multivariate_normal(rng, mean=jnp.zeros_like(map_solution), cov=scale_tril @ scale_tril.T, shape=(n_posterior_samples,))
        print(V_LA.shape)

    #  ok now I have the initial velocities. I can therefore consider my manifold
    if args.linearized_pred:
        # here I have to first compute the f_MAP in both cases
        state = state.replace(params = unravel_fn(map_solution))
        f_MAP = state.apply_fn(state.params, x_train)

        state_model_2 = create_train_state(rng, model, optimizer=args.optimizer)
        
        # here depending if I am using a diagonal approx, I have to redefine the model
        if batch_data:
            # Create a TensorFlow dataset
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, f_MAP))
            
            # Shuffle, batch and prefetch for performance optimization
            new_train_loader = dataset.shuffle(buffer_size=len(x_train))  # Shuffling the entire dataset
            new_train_loader = new_train_loader.batch(50)  # Batch size of 50
            new_train_loader = new_train_loader.prefetch(tf.data.AUTOTUNE)  # Automatically tune the prefetch buffer size

        ########### All the lines below will not work until we have converted linearized_cross_entropy_manifold ####
        if optimize_prior:
            if batch_data:
                manifold = linearized_cross_entropy_manifold(
                    state_model_2,
                    new_train_loader,
                    y=None,
                    f_MAP=f_MAP,
                    theta_MAP=map_solution,
                    unravel_fn = unravel_fn,
                    batching=True,
                    lambda_reg=la.prior_precision.item() / 2,
                )

            else:
                manifold = linearized_cross_entropy_manifold(
                    state_model_2,
                    x_train,
                    y_train,
                    f_MAP=f_MAP,
                    theta_MAP=map_solution,
                    unravel_fn = unravel_fn,
                    batching=False,
                    lambda_reg=la.prior_precision.item() / 2,
                )
        else:
            if batch_data:
                manifold = linearized_cross_entropy_manifold(
                    state_model_2,
                    new_train_loader,
                    y=None,
                    f_MAP=f_MAP,
                    theta_MAP=map_solution,
                    unravel_fn = unravel_fn,
                    batching=True,
                    lambda_reg=weight_decay,
                )

            else:
                manifold = linearized_cross_entropy_manifold(
                    state_model_2,
                    x_train,
                    y_train,
                    f_MAP=f_MAP,
                    theta_MAP=map_solution,
                    unravel_fn = unravel_fn,
                    batching=False,
                    lambda_reg=weight_decay,
                )
    else:
        # here we have the usual manifold
        state_model_2 = create_train_state(rng, model, optimizer=args.optimizer)

        # here depending if I am using a diagonal approx, I have to redefine the model
        if optimize_prior:
            if batch_data:
                manifold = cross_entropy_manifold(
                    state_model_2, train_loader, y=None, batching=True, lambda_reg=la.prior_precision.item() / 2
                )

            else:
                manifold = cross_entropy_manifold(
                    state_model_2, x_train, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
                )
        else:
            if batch_data:
                manifold = cross_entropy_manifold(
                    state_model_2, train_loader, y=None, batching=True, lambda_reg=weight_decay
                )

            else:
                manifold = cross_entropy_manifold(
                    state_model_2, x_train, y_train, batching=False, lambda_reg=weight_decay
                )
    # now i have my manifold and so I can solve the expmap
    weights_ours = jnp.zeros((n_posterior_samples, len(map_solution)))
    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = V_LA[n, :].reshape(-1, 1)
            # here I can try to sample a subset of datapoints, create a new manifold and solve expmap
        if args.expmap_different_batches:
            n_sub_data = 150

            idx_sub = jax.random.choice(rng, jnp.arange(len(x_train)), shape=(n_sub_data,), replace=False)
            sub_x_train = x_train[idx_sub, :]
            sub_y_train = y_train[idx_sub]
            if args.linearized_pred:
                sub_f_MAP = f_MAP[idx_sub]
                manifold = linearized_cross_entropy_manifold(
                    state_model_2,
                    sub_x_train,
                    sub_y_train,
                    f_MAP=sub_f_MAP,
                    theta_MAP=map_solution,
                    unravel_fn=unravel_fn,
                    batching=False,
                    lambda_reg=la.prior_precision.item() / 2,
                    N=len(x_train),
                    B1=n_sub_data,
                )
            else:
                manifold = cross_entropy_manifold(
                    state_model_2,
                    sub_x_train,
                    sub_y_train,
                    batching=False,
                    lambda_reg=la.prior_precision.item() / 2,
                    N=len(x_train),
                    B1=n_sub_data,
                )

            curve, failed = geometry.expmap(manifold, map_solution.clone(), v)
        else:
            curve, failed = geometry.expmap(manifold, map_solution.clone(), v)
        _new_weights = curve(1)[0]
        weights_ours = weights_ours.at[n, :].set(jnp.array(_new_weights.reshape(-1)))
    # now I can use my weights for prediction. Deoending if I am using linearization or not the prediction looks differently
    if args.linearized_pred:
        state_model_2 = state_model_2.replace(params = unravel_fn(map_solution))
        f_MAP_grid = state_model_2.apply_fn(state_model_2.params, X_grid)
        f_MAP_test = state_model_2.apply_fn(state_model_2.params, x_test)

        def predict(params, datas):
            y_pred = state_model_2.apply_fn(params, datas)
            return y_pred

        P_grid_OURS_lin = 0
        P_test_OURS = 0


        # now I can do the same for our method
        for n in range(n_posterior_samples):
            # get the theta weights we are interested in #
            w_OUR = weights_ours[n, :]
            params = unravel_fn(map_solution)

            diff_weights = (w_OUR - map_solution).astype(jnp.float32)

            diff_as_params = unravel_fn(diff_weights)

            _, jvp_value_grid = jax.jvp(
                predict,
                (params, X_grid),
                (diff_as_params, jnp.zeros_like(X_grid)))

            f_OUR_grid = f_MAP_grid + jvp_value_grid

            probs_grid = jax.nn.softmax(f_OUR_grid, axis=1)
            P_grid_OURS_lin += probs_grid

        P_grid_OURS_lin /= n_posterior_samples
        P_grid_OUR_conf = P_grid_OURS_lin.max(1)

        plt.contourf(
            XX1,
            XX2,
            P_grid_OUR_conf.reshape(N_grid, N_grid),
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
            XX1, XX2, P_grid_OURS_lin[:, 0].reshape(N_grid, N_grid), levels=[0.5], colors="k", alpha=0.5, zorder=0
        )
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("All weights, full Hessian approx - Confidence OUR linearized")
        plt.show()

        # now I can do the same for our method
        for n in range(n_posterior_samples):
            # get the theta weights we are interested in #
            w_OUR = weights_ours[n, :]
            params = unravel_fn(map_solution)

            diff_weights = (w_OUR - map_solution).astype(jnp.float32)

            diff_as_params = unravel_fn(diff_weights)

            _, jvp_value_test = jax.jvp(
                predict,
                (params, x_test),
                (diff_as_params, jnp.zeros_like(x_test)))

            f_OUR_test = f_MAP_test + jvp_value_test

            probs_test = jax.nn.softmax(f_OUR_test, axis=1)
            P_test_OURS += probs_test

    else:
        # and then our stuff
        P_grid_OUR = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            state = state.replace(params = unravel_fn(weights_ours[n, :]))
            # compute the predictions
            P_grid_OUR += jax.nn.softmax(state.apply_fn(state.params, X_grid), axis=1)

        P_grid_OUR /= n_posterior_samples
        P_grid_OUR_conf = P_grid_OUR.max(1)

        plt.contourf(
            XX1,
            XX2,
            P_grid_OUR_conf.reshape(N_grid, N_grid),
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
        plt.contour(XX1, XX2, P_grid_OUR[:, 0].reshape(N_grid, N_grid), levels=[0.5], colors="k", alpha=0.5, zorder=0)
        plt.title("All weights, full Hessian approx - Confidence OURS")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

        P_test_OURS = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            state = state.replace(params = unravel_fn(weights_ours[n, :]))
            # compute the predictions
            P_test_OURS += jax.nn.softmax(state.apply_fn(state.params, x_test), axis=1)
        
    # I can compute and plot the results

    P_test_OURS /= n_posterior_samples

    accuracy_OURS = accuracy(P_test_OURS, y_test)
    nll_OUR = nll(P_test_OURS, y_test)
    brier_OURS = brier(P_test_OURS, y_test)
    #ece_our = calibration_error(P_test_OURS, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    #mce_our = calibration_error(P_test_OURS, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    #print(f"Results OURS: accuracy {accuracy_OURS}, nll {nll_OUR}, brier {brier_OURS}, ECE {ece_our}, MCE {mce_our}")
    # now I can create my dictionary
    #dict_OUR = {"Accuracy": accuracy_OURS, "NLL": nll_OUR, "Brier": brier_OURS, "ECE": ece_our, "MCE": mce_our}
    print(f"Results OURS: accuracy {accuracy_OURS}, nll {nll_OUR}, brier {brier_OURS}")
    # now I can create my dictionary
    dict_OUR = {"Accuracy": accuracy_OURS, "NLL": nll_OUR, "Brier": brier_OURS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geomeatric Approximate Inference (GEOMAI)")
    parser.add_argument("--seed", "-s", type=int, default=230, help="seed")
    parser.add_argument("--optimizer", "-optim", type=str, default="sgd", help="otpimizer used to train the model")

    parser.add_argument("--optimize_prior", "-opt_prior", type=bool, default=False, help="optimize prior")
    parser.add_argument("--batch_data", "-batch", type=bool, default=False, help="batch data")

    parser.add_argument("--structure", "-str", type=str, default="full", help="Hessian struct for Laplace")
    parser.add_argument("--subset", "-sub", type=str, default="all", help="subset of weights for Laplace")
    parser.add_argument("--samples", "-samp", type=int, default=50, help="number of posterior samples")
    parser.add_argument("--linearized_pred", "-lin", type=bool, default=False, help="Linearization for prediction")
    parser.add_argument(
        "--expmap_different_batches",
        "-batches",
        type=bool,
        default=False,
        help="Solve exponential map using only a batch of the data and not the full dataset",
    )
    parser.add_argument(
        "--test_all",
        "-test_all",
        type=bool,
        default=False,
        help="Use also the validation set that we are not using for evaluation",
    )

    args = parser.parse_args()
    main(args)
