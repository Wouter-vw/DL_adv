"""
File containing the banana experiments
"""
import torch
import numpy as np
import matplotlib
import jax
import jax.numpy as jnp
import jax.random as jrandom

from torch2jax import t2j
import tensorflow as tf

import optax

from neural_network import MLP, create_train_state, compute_loss, train_step
from data_loading import load_banana_data, create_data_loader
from plots import plot_map_confidence, plot_ours_confidence

from laplace import Laplace
import seaborn as sns
#####################################
import geomai.utils.geometry as geometry
####################################################
from torch import nn as nn_torch
########################################
from manifold_kfac import linearized_cross_entropy_manifold, cross_entropy_manifold
#########################################
from tqdm import tqdm
from utils.metrics import accuracy, nll, brier
import argparse
from torchmetrics.functional.classification import calibration_error


jax.config.update('jax_enable_x64', True)

def main(args):
    palette = sns.color_palette("colorblind")
    print("Linearization?")
    print(args.linearized_pred)
    subset_of_weights = args.subset  # must be 'all'
    hessian_structure = args.structure  #'full' # other possibility is 'diag' or 'full'
    n_posterior_samples = args.samples
    optimize_prior = args.optimize_prior
    print("Are we optimizing the prior? ", optimize_prior)

    batch_data = args.batch_data

    # run with several seeds
    seed = args.seed
    jrandom.PRNGKey(seed)
    rng = jrandom.PRNGKey(seed)
    torch.manual_seed(seed)
    print("Seed: ", seed)

    # Load the banana dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_banana_data()
    
    # Create data loaders
    batch_size_train = 256
    batch_size_valid = 50
    batch_size_test = 50
    
    train_loader = create_data_loader(x_train, y_train, batch_size_train, rng, shuffle=True)
    valid_loader = create_data_loader(x_valid, y_valid, batch_size_valid, rng, shuffle=False)
    test_loader = create_data_loader(x_test, y_test, batch_size_test, rng, shuffle=False)
    
    if args.optimizer == "sgd":
        learning_rate = 1e-3
        weight_decay = 1e-2
        max_epoch = 2000

        # Define the optimizer with weight decay
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),  # Apply weight decay
            optax.sgd(learning_rate)                  # SGD optimizer
        )
    else:
        learning_rate = 1e-3
        weight_decay = 1e-3
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        max_epoch = 1500  

    num_features = x_train.shape[-1]
    num_output = 2
    H = 16
    model = MLP(num_features=num_features, hidden_size=H, num_output=num_output)
    state = create_train_state(rng, model, optimizer=optimizer)

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

    plot_map_confidence(x_train, y_train, XX1, XX2, conf, title="Confidence MAP")

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

    @jax.jit
    def rearrange_V_LA(test):
        # Initialize the output array
        V_LA_jax = jnp.zeros(354, dtype=test.dtype)

        # Assign the first slice
        V_LA_jax = V_LA_jax.at[0:16].set(test[32:48])

        # Interleave even and odd indices
        even_indices = test[0:32:2]
        odd_indices = test[1:32:2]
        V_LA_jax = V_LA_jax.at[16:48].set(jnp.concatenate([even_indices, odd_indices], axis=0))

        # Assign the next slice
        V_LA_jax = V_LA_jax.at[48:64].set(test[304:320])

        # Rearrange test[48:304]
        num_rows = (304 - 48) // 16
        indices = jnp.arange(num_rows * 16).reshape(16, num_rows).T.flatten()
        V_LA_jax = V_LA_jax.at[64:320].set(test[48:304][indices])

        # Assign test[352:354]
        V_LA_jax = V_LA_jax.at[320:322].set(test[352:354])

        # Rearrange test[320:352]
        test_test = test[320:352]
        indices_2 = jnp.arange(test_test.size).reshape(-1, 16).T.flatten()
        V_LA_jax = V_LA_jax.at[322:354].set(test_test[indices_2])

        return V_LA_jax

    for n in range(n_posterior_samples):
        V_LA = V_LA.at[n, :].set(rearrange_V_LA(V_LA[n, :]))

    #  ok now I have the initial velocities. I can therefore consider my manifold
    if args.linearized_pred:
        # here I have to first compute the f_MAP in both cases
        state = state.replace(params = unravel_fn(map_solution))
        f_MAP = state.apply_fn(state.params, x_train)

        state_model_2 = create_train_state(rng, model, optimizer=optimizer)
        
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
                    model,
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
                    model,
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
                    model,
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
                    model,
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
        state_model_2 = create_train_state(rng, model, optimizer=optimizer)

        # here depending if I am using a diagonal approx, I have to redefine the model
        if optimize_prior:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model, state_model_2, train_loader, y=None, unravel_fn=unravel_fn, batching=True, lambda_reg=la.prior_precision.item() / 2
                )

            else:
                manifold = cross_entropy_manifold(
                    model, state_model_2, x_train, y_train, unravel_fn=unravel_fn, batching=False, lambda_reg=la.prior_precision.item() / 2
                )
        else:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model, state_model_2, train_loader, y=None, unravel_fn=unravel_fn, batching=True, lambda_reg=weight_decay
                )

            else:
                manifold = cross_entropy_manifold(
                    model, state_model_2, x_train, y_train, unravel_fn=unravel_fn, batching=False, lambda_reg=weight_decay
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
                    model,
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
                    model,
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

        plot_ours_confidence(
            x_train, y_train, XX1, XX2, P_grid_OUR_conf, P_grid_OURS_lin[:, 0], title="Confidence OURS linearized"
        )

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

        plot_ours_confidence(
            x_train, y_train, XX1, XX2, P_grid_OUR_conf, P_grid_OUR[:, 0], title="Confidence OURS"
        )

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
    
    ece_our = calibration_error(torch.from_numpy(np.array(P_test_OURS)), torch.from_numpy(np.array(y_test)), norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    mce_our = calibration_error(torch.from_numpy(np.array(P_test_OURS)), torch.from_numpy(np.array(y_test)), norm="max", task="multiclass", num_classes=2, n_bins=10) * 100
    #ece_our = calibration_error(P_test_OURS, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    #mce_our = calibration_error(P_test_OURS, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    print(f"Results OURS: accuracy {accuracy_OURS}, nll {nll_OUR}, brier {brier_OURS}, ECE {ece_our}, MCE {mce_our}")

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