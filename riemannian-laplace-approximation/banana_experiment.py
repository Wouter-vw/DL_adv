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
import geomai.utils.geometry as geometry
from torch import nn as nn_torch
from manifold import cross_entropy_manifold, linearized_cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import sklearn.datasets
from datautils import make_pinwheel_data
from utils.metrics import accuracy, nll, brier, calibration
from sklearn.metrics import brier_score_loss
import argparse
from torchmetrics.functional.classification import calibration_error
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2
import os

jax.config.update('jax_enable_x64', True)

def main(args):
    palette = sns.color_palette("colorblind")
    print("Linearization?")
    print(args.linearized_pred)
    subset_of_weights = args.subset  #'last_layer' # either 'last_layer' or 'all'
    hessian_structure = args.structure  #'full' # other possibility is 'diag' or 'full'
    n_posterior_samples = args.samples
    security_check = True
    optimize_prior = args.optimize_prior
    print("Are we optimizing the prior? ", optimize_prior)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(matplotlib.rcParams["lines.markersize"] ** 2)
    plt.scatter(
        x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c="orange", edgecolors="black", s=45, alpha=1
    )
    plt.scatter(
        x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c="violet", edgecolors="black", s=45, alpha=1
    )
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Train")
    plt.show()

    print("Some info about the dataset:")
    print(f"Train: {x_train.shape, y_train.shape}")
    print(f"Valid: {x_valid.shape, y_valid.shape}")
    print(f"Test: {x_test.shape, y_test.shape}")

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
    def create_train_state(rng, model, learning_rate, weight_decay, optimizer_type):
        params = model.init(rng, jnp.ones([1, num_features]))  # Dummy input for parameter initialization
        if optimizer_type == 'sgd':
            #weight_decay = 1e-2
            #optimizer = optax.chain(optax.sgd(learning_rate), optax.add_decayed_weights(weight_decay))
            optimizer = optax.sgd(learning_rate)
        else:
            weight_decay = 1e-3
            optimizer = optax.chain(optax.adam(learning_rate=learning_rate), optax.add_decayed_weights(weight_decay))
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

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
    state = create_train_state(rng, model, learning_rate=1e-3, weight_decay=1e-2, optimizer_type=args.optimizer)

    
    max_epoch = 100
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
    
    def params_from_map(map_solution, state):
        params = {'params': {}}
        idx = 0
        
        # Iterate over the layers to reconstruct the parameters dynamically
        for layer_name, layer_params in state.params['params'].items():
            # Get the shape of the kernel and bias
            kernel_shape = layer_params['kernel'].shape
            bias_shape = layer_params['bias'].shape
            
            # Calculate the number of elements in the kernel and bias
            kernel_size = jnp.prod(jnp.array(kernel_shape))
            bias_size = jnp.prod(jnp.array(bias_shape))

            # Extract and reshape kernel from map_solution
            kernel_flat = map_solution[idx:idx + kernel_size]
            kernel = kernel_flat.reshape(kernel_shape)
            idx += kernel_size
            
            # Extract and reshape bias from map_solution
            bias_flat = map_solution[idx:idx + bias_size]
            bias = bias_flat.reshape(bias_shape)
            idx += bias_size
            
            # Assign the kernel and bias to the params dictionary
            params['params'][layer_name] = {'kernel': kernel, 'bias': bias}
        
        # Replace the state with the new params
        new_state = state.replace(params=params)

        return new_state

    def get_map_solution(state):
        """Function to flatten the kernels and biases, then concatenate them."""
        params_flattened = []
        
        # Iterate over the layers dynamically
        for layer_name, layer_params in state.params['params'].items():
            kernel = layer_params['kernel']
            bias = layer_params['bias']
            
            # Transpose the kernel and flatten both kernel and bias
            params_flattened.extend([kernel.flatten(), bias.flatten()])
        
        # Concatenate the flattened kernel and bias arrays
        map_solution = jnp.concatenate(params_flattened)
        
        return map_solution
    
    map_solution = get_map_solution(state)
    state = params_from_map(map_solution, state)

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
    plt.show()

    ## Still need to add weight decay to optimizers
    if args.optimizer == 'sgd':
        weight_decay = 1e-2
    else:
        weight_decay = 1e-3

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

    ## We can verify that the transfer went well
    # computing and plotting the MAP confidence
    # with torch.no_grad():
    #     probs_map = torch.softmax(model_torch(torch.from_numpy(X_grid).float()), dim=1).numpy()

    # conf = probs_map.max(1)

    # # Plotting
    # plt.contourf(
    #     XX1,
    #     XX2,
    #     conf.reshape(N_grid, N_grid),
    #     alpha=0.8,
    #     antialiased=True,
    #     cmap="Blues",
    #     levels=jnp.arange(0.0, 1.01, 0.1),
    # )
    # plt.colorbar()
    # plt.scatter(
    #     x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c="orange", edgecolors="black", s=45, alpha=1
    # )
    # plt.scatter(
    #     x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c="violet", edgecolors="black", s=45, alpha=1
    # )
    # plt.title("Confidence MAP")
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()

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
    if subset_of_weights == "last_layer":
        if hessian_structure == "diag":
            n_last_layer_weights = num_output * H + num_output ## CHECK!! Why is the original not using this number?
            samples = jax.random.normal(rng, shape=(n_posterior_samples, la.n_params))
            samples = samples * t2j(la.posterior_scale.reshape(1, la.n_params))
            V_LA = samples
        else:
            n_last_layer_weights = num_output * H + num_output
            scale_tril = scale_tril = jnp.array(la.posterior_scale)
            V_LA = jax.random.multivariate_normal(rng, mean=jnp.zeros(n_last_layer_weights), cov=scale_tril @ scale_tril.T, shape=(n_posterior_samples,))
    else:
        if hessian_structure == "diag":
            samples = jax.random.normal(rng, shape=(n_posterior_samples, la.n_params))
            samples = samples * t2j(la.posterior_scale.reshape(1, la.n_params))
            V_LA = samples

        else:
            scale_tril = scale_tril = jnp.array(la.posterior_scale)
            V_LA = jax.random.multivariate_normal(rng, mean=jnp.zeros_like(map_solution), cov=scale_tril @ scale_tril.T, shape=(n_posterior_samples,))
            print(V_LA.shape)


    class FeatureExtractor(nn.Module):
        num_features: int
        H: int
        num_output: int

        def setup(self):
            # Define the layers
            self.dense1 = nn.Dense(self.H)
            self.dense2 = nn.Dense(self.H)
            self.output = nn.Dense(self.num_output)

        def __call__(self, x):
            # Pass through the layers with Tanh activations
            x = nn.tanh(self.dense1(x))
            x = nn.tanh(self.dense2(x))
            return self.output(x)

    # To instantiate the model ( Reminder to change weight decay, learning rate etc.. )
    feature_extractor_model = FeatureExtractor(num_features=num_features, H=H, num_output=num_output)
    f_state = create_train_state(rng, feature_extractor_model, learning_rate=1e-3, weight_decay=1e-2, optimizer_type="sgd")

    class LinearModel(nn.Module):
        H: int
        num_output: int

        def setup(self):
            # Define the linear layer
            self.output = nn.Dense(self.num_output)

        def __call__(self, x):
            # Pass through the linear layer
            return self.output(x)

    # To instantiate the model ( Reminder to change weight decay, learning rate etc.. )
    ll = LinearModel(H=H, num_output=num_output)
    l_state = create_train_state(rng, ll, learning_rate=1e-3, weight_decay=1e-2, optimizer_type="sgd")

    # ok now I have the initial velocities. I can therefore consider my manifold
    if args.linearized_pred:
        # here I have to first compute the f_MAP in both cases
        state = params_from_map(map_solution, state)
        f_MAP = state.apply_fn(state.params, x_train)

        if subset_of_weights == "last_layer":
            weights_ours = jnp.zeros(n_posterior_samples, len(map_solution))

            MAP = map_solution.clone()
            feature_extractor_map = MAP[0:-n_last_layer_weights]
            ll_map = MAP[-n_last_layer_weights:]
            print(feature_extractor_map.shape)
            print(ll_map.shape)

            state_f_map = params_from_map(map_solution, f_state)
            state_ll = params_from_map(map_solution, l_state)
            
            # I have to precompute some stuff
            # i.e. I am treating the hidden activation before the last layer as my input
            # because since the weights are fixed, then this feature vector is fixed
            R = f_state.apply_fn(f_state.params, x_train)
            #################### I cannot work with the manifold yet as its still based on torch. ##########
            if optimize_prior:
                manifold = linearized_cross_entropy_manifold(
                    ll,
                    R,
                    y_train,
                    f_MAP=f_MAP,
                    theta_MAP=ll_map,
                    batching=False,
                    lambda_reg=la.prior_precision.item() / 2,
                )

            else:
                manifold = linearized_cross_entropy_manifold(
                    ll, R, y_train, f_MAP=f_MAP, theta_MAP=ll_map, batching=False, lambda_reg=weight_decay
                )
        else:
            state_model_2 = create_train_state(rng, model, learning_rate=1e-3, weight_decay=1e-2, optimizer_type="sgd")
            
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
                        batching=False,
                        lambda_reg=weight_decay,
                    )
    else:
        # here we have the usual manifold
        if subset_of_weights == "last_layer":
            weights_ours = jnp.zeros(n_posterior_samples, len(map_solution))

            MAP = map_solution.clone()
            feature_extractor_map = MAP[0:-n_last_layer_weights]
            ll_map = MAP[-n_last_layer_weights:]
            print(feature_extractor_map.shape)
            print(ll_map.shape)

            state_f_map = params_from_map(map_solution, f_state)
            state_ll = params_from_map(map_solution, l_state)
            
            R = f_state.apply_fn(f_state.params, x_train)

            if optimize_prior:
                manifold = cross_entropy_manifold(
                    ll, R, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
                )

            else:
                manifold = cross_entropy_manifold(ll, R, y_train, batching=False, lambda_reg=weight_decay)

        else:
            state_model_2 = create_train_state(rng, model, learning_rate=1e-3, weight_decay=1e-2, optimizer_type="sgd")

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
           ########################################### Below this I still have to rewrite in jax ################         
    # now i have my manifold and so I can solve the expmap
    weights_ours = torch.zeros(n_posterior_samples, len(map_solution))
    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = V_LA[n, :].reshape(-1, 1)

        if subset_of_weights == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            _new_ll_weights = curve(1)[0]
            _new_weights = torch.cat(
                (feature_extractor_map.view(-1), torch.from_numpy(_new_ll_weights).float().view(-1)), dim=0
            )
            weights_ours[n, :] = _new_weights.view(-1)
            torch.nn.utils.vector_to_parameters(_new_weights, model.parameters())

        else:
            # here I can try to sample a subset of datapoints, create a new manifold and solve expmap
            if args.expmap_different_batches:
                n_sub_data = 150

                idx_sub = np.random.choice(np.arange(0, len(x_train), 1), n_sub_data, replace=False)
                sub_x_train = x_train[idx_sub, :]
                sub_y_train = y_train[idx_sub]
                if args.linearized_pred:
                    sub_f_MAP = f_MAP[idx_sub]
                    manifold = linearized_cross_entropy_manifold(
                        model2,
                        sub_x_train,
                        sub_y_train,
                        f_MAP=sub_f_MAP,
                        theta_MAP=map_solution,
                        batching=False,
                        lambda_reg=la.prior_precision.item() / 2,
                        N=len(x_train),
                        B1=n_sub_data,
                    )
                else:
                    manifold = cross_entropy_manifold(
                        model2,
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
            weights_ours[n, :] = torch.from_numpy(_new_weights.reshape(-1))

    # I can get the LA weights
    weights_LA = torch.zeros(n_posterior_samples, len(map_solution))

    for n in range(n_posterior_samples):
        if subset_of_weights == "last_layer":
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float() + ll_map.clone()
            laplace_weigths = torch.cat((feature_extractor_map.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()
        else:
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float() + map_solution
            # laplace_weigths = torch.cat((feature_extractor_MAP.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()

    # now I can use my weights for prediction. Deoending if I am using linearization or not the prediction looks differently
    if args.linearized_pred:
        if subset_of_weights == "last_layer":
            # so I have to put the MAP back to the feature extraction part of the model
            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            # and then I have to create the new dataset
            with torch.no_grad():
                R_MAP_grid = feature_extractor_model(torch.from_numpy(X_grid).float()).clone()
                R_MAP_test = feature_extractor_model(x_test)

            # I have also to compute the f_MAP here
            with torch.no_grad():
                f_MAP_grid = ll(R_MAP_grid).clone()
                f_MAP_test = ll(R_MAP_test)

            # now I have my dataset, and I have also the initial velocities I
            # computed for LA

            # I guess I should not consider the softmax here
            def predict(params, data):
                y_pred = fmodel(params, buffers, data)
                return y_pred

            P_grid_LAPLACE_lin = 0
            P_grid_OURS_lin = 0

            # let's start with the X_grid
            for n in range(n_posterior_samples):
                w_LA = weights_LA[n, :]
                w_ll_LA = w_LA[-n_last_layer_weights:]

                assert len(w_ll_LA) == len(
                    ll_map
                ), "We have a problem in the length of the last layer weights we are considering"
                # put the weights into the model
                torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())
                ll.zero_grad()

                diff_weights = w_ll_LA - ll_map

                fmodel, params, buffers = make_functional_with_buffers(ll)

                diff_as_params = get_params_structure(diff_weights, params)

                # here I have to use the new dataset to predict
                _, jvp_value_grid = jvp(
                    predict, (params, R_MAP_grid), (diff_as_params, torch.zeros_like(R_MAP_grid)), strict=False
                )

                f_LA_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_LA_grid, dim=1)
                P_grid_LAPLACE_lin += probs_grid.detach().numpy()

            # I should also do the same with our model (and use the tangent vector)
            # because if we use the final weights we get something wrong
            for n in range(n_posterior_samples):
                w_OUR = weights_ours[n, :]
                w_ll_OUR = w_OUR[-n_last_layer_weights:]
                assert len(w_ll_OUR) == len(
                    ll_map
                ), "We have a problem in the length of the last layer weights we are considering"
                # put the weights into the model
                torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())
                ll.zero_grad()

                diff_weights = w_ll_OUR - ll_map

                fmodel, params, buffers = make_functional_with_buffers(ll)

                diff_as_params = get_params_structure(diff_weights, params)

                # here I have to use the new dataset to predict
                _, jvp_value_grid = jvp(
                    predict, (params, R_MAP_grid), (diff_as_params, torch.zeros_like(R_MAP_grid)), strict=False
                )

                f_OUR_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_OUR_grid, dim=1)
                P_grid_OURS_lin += probs_grid.detach().numpy()

        else:
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            with torch.no_grad():
                f_MAP_grid = model2(torch.from_numpy(X_grid).float()).clone()
                f_MAP_test = model2(x_test)

            def predict(params, data):
                y_pred = fmodel(params, buffers, data)
                return y_pred

            P_grid_LAPLACE_lin = 0
            P_grid_OURS_lin = 0
            P_test_OURS = 0
            P_test_LAPLACE = 0

            # let's start with the X_grid
            for n in range(n_posterior_samples):
                w_LA = weights_LA[n, :]
                # put the weights into the model
                torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
                model2.zero_grad()

                diff_weights = w_LA - map_solution

                fmodel, params, buffers = make_functional_with_buffers(model2)

                diff_as_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    predict,
                    (params, torch.from_numpy(X_grid).float()),
                    (diff_as_params, torch.zeros_like(torch.from_numpy(X_grid).float())),
                    strict=False,
                )

                f_LA_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_LA_grid, dim=1)
                P_grid_LAPLACE_lin += probs_grid.detach().numpy()

            # now I can do the same for our method
            for n in range(n_posterior_samples):
                # get the theta weights we are interested in #
                w_OUR = weights_ours[n, :]
                torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
                model2.zero_grad()

                diff_weights = w_OUR - map_solution

                fmodel, params, buffers = make_functional_with_buffers(model2)

                # I have to make the diff_weights with the same tree shape as the params
                diff_as_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    predict,
                    (params, torch.from_numpy(X_grid).float()),
                    (diff_as_params, torch.zeros_like(torch.from_numpy(X_grid).float())),
                    strict=False,
                )

                f_OUR_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_OUR_grid, dim=1)
                P_grid_OURS_lin += probs_grid.detach().numpy()

        P_grid_LAPLACE_lin /= n_posterior_samples

        P_grid_LAPLACE_conf = P_grid_LAPLACE_lin.max(1)

        plt.contourf(
            XX1,
            XX2,
            P_grid_LAPLACE_conf.reshape(N_grid, N_grid),
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
            alpha=1.0,
            zorder=10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 1],
            x_train[:, 1][y_train == 1],
            c="violet",
            edgecolors="black",
            s=45,
            alpha=1.0,
            zorder=10,
        )
        plt.contour(
            XX1, XX2, P_grid_LAPLACE_lin[:, 0].reshape(N_grid, N_grid), levels=[0.5], colors="k", alpha=0.5, zorder=0
        )
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("All weights, full Hessian approx - Confidence LA linearized")
        plt.show()

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
        # plt.colorbar()
        # plt.scatter(x_train[:,0], x_train[:,1], s=40, c=y_train, edgecolors='k', cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
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
        # plt.title('All weights, full Hessian approx - Confidence OURS linearized')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("All weights, full Hessian approx - Confidence OUR linearized")
        plt.show()

        # I have also to add the computation on the test set
        for n in range(n_posterior_samples):
            w_LA = weights_LA[n, :]
            # put the weights into the model
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            model2.zero_grad()

            diff_weights = w_LA - map_solution

            fmodel, params, buffers = make_functional_with_buffers(model2)

            diff_as_params = get_params_structure(diff_weights, params)

            _, jvp_value_test = jvp(
                predict, (params, x_test), (diff_as_params, torch.zeros_like(x_test)), strict=False
            )

            f_LA_test = f_MAP_test + jvp_value_test

            probs_test = torch.softmax(f_LA_test, dim=1)
            P_test_LAPLACE += probs_test.detach()

        # now I can do the same for our method
        for n in range(n_posterior_samples):
            # get the theta weights we are interested in #
            w_OUR = weights_ours[n, :]
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            model2.zero_grad()

            diff_weights = w_OUR - map_solution

            fmodel, params, buffers = make_functional_with_buffers(model2)

            # I have to make the diff_weights with the same tree shape as the params
            diff_as_params = get_params_structure(diff_weights, params)

            _, jvp_value_grid = jvp(
                predict, (params, x_test), (diff_as_params, torch.zeros_like(x_test)), strict=False
            )

            f_OUR_test = f_MAP_test + jvp_value_grid

            probs_test = torch.softmax(f_OUR_test, dim=1)
            P_test_OURS += probs_test.detach()

    else:
        P_grid_LAPLACE = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_grid_LAPLACE += torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).numpy()

        P_grid_LAPLACE /= n_posterior_samples

        P_grid_LAPLACE_conf = P_grid_LAPLACE.max(1)

        plt.contourf(
            XX1,
            XX2,
            P_grid_LAPLACE_conf.reshape(N_grid, N_grid),
            alpha=0.8,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.0, 1.01, 0.1),
            zorder=-10,
        )
        # plt.colorbar()
        # plt.scatter(x_train[:,0], x_train[:,1], s=40, c=y_train, edgecolors='k', cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        plt.scatter(
            x_train[:, 0][y_train == 0],
            x_train[:, 1][y_train == 0],
            c="orange",
            edgecolors="black",
            s=45,
            alpha=1.0,
            zorder=10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 1],
            x_train[:, 1][y_train == 1],
            c="violet",
            edgecolors="black",
            s=45,
            alpha=1.0,
            zorder=10,
        )
        plt.contour(
            XX1, XX2, P_grid_LAPLACE[:, 0].reshape(N_grid, N_grid), levels=[0.5], colors="k", alpha=0.5, zorder=0
        )
        plt.title("All weights, full Hessian approx - Confidence LA")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

        # and then our stuff
        P_grid_OUR = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_grid_OUR += torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).numpy()

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
        # plt.colorbar()
        # plt.scatter(x_train[:,0], x_train[:,1], s=40, c=y_train, edgecolors='k', cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
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

        # I have to add some computation in the test set that i was missing here
        P_test_LAPLACE = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace prediction in region"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_test_LAPLACE += torch.softmax(model(x_test), dim=1)

        P_test_OURS = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace prediction in region"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_test_OURS += torch.softmax(model(x_test), dim=1)

    # I can compute and plot the results
    # here I can also compute the MAP results
    torch.nn.utils.vector_to_parameters(map_solution, model.parameters())
    with torch.no_grad():
        P_test_MAP = torch.softmax(model(x_test), dim=1)

    accuracy_MAP = accuracy(P_test_MAP, y_test)

    nll_MAP = nll(P_test_MAP, y_test)

    brier_MAP = brier(P_test_MAP, y_test)

    ece_map = calibration_error(P_test_MAP, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    mce_map = calibration_error(P_test_MAP, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    P_test_OURS /= n_posterior_samples
    P_test_LAPLACE /= n_posterior_samples

    accuracy_LA = accuracy(P_test_LAPLACE, y_test)
    accuracy_OURS = accuracy(P_test_OURS, y_test)

    nll_LA = nll(P_test_LAPLACE, y_test)
    nll_OUR = nll(P_test_OURS, y_test)

    brier_LA = brier(P_test_LAPLACE, y_test)
    brier_OURS = brier(P_test_OURS, y_test)

    ece_la = calibration_error(P_test_LAPLACE, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    mce_la = calibration_error(P_test_LAPLACE, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    ece_our = calibration_error(P_test_OURS, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    mce_our = calibration_error(P_test_OURS, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    print(f"Results MAP: accuracy {accuracy_MAP}, nll {nll_MAP}, brier {brier_MAP}, ECE {ece_map}, MCE {mce_map}")
    print(f"Results LA: accuracy {accuracy_LA}, nll {nll_LA}, brier {brier_LA}, ECE {ece_la}, MCE {mce_la}")
    print(f"Results OURS: accuracy {accuracy_OURS}, nll {nll_OUR}, brier {brier_OURS}, ECE {ece_our}, MCE {mce_our}")

    # now I can create my dictionary
    dict_MAP = {"Accuracy": accuracy_MAP, "NLL": nll_MAP, "Brier": brier_MAP, "ECE": ece_map, "MCE": mce_map}
    dict_LA = {"Accuracy": accuracy_LA, "NLL": nll_LA, "Brier": brier_LA, "ECE": ece_la, "MCE": mce_la}
    dict_OUR = {"Accuracy": accuracy_OURS, "NLL": nll_OUR, "Brier": brier_OURS, "ECE": ece_our, "MCE": mce_our}

    final_dict = {"results_MAP": dict_MAP, "results_LA": dict_LA, "results_OUR": dict_OUR}


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
