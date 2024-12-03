"""
File containing the banana experiments
"""

import argparse
import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import torch
from laplace import Laplace
from torch import nn as nn_torch
from torchmetrics.functional.classification import calibration_error
from tqdm import tqdm
####################################
import warnings

# Suppresses an irrelevant warning from the laplace package
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*laplace\.baselaplace"
)

#####################################

import manifold.geometry_diffrax as geometry_diffrax
import manifold.geometry as geometry
from manifold.manifold_kfac import CrossEntropyManifold_kfac, LinearizedCrossEntropyManifold_kfac
from manifold.manifold import CrossEntropyManifold, LinearizedCrossEntropyManifold
from utils.data_loading import create_data_loader, load_banana_data
from utils.evaluation import accuracy, brier, nll
from utils.neural_network import MLP, compute_loss, create_train_state, train_step
from utils.plots import plot_map_confidence, plot_confidence

jax.config.update("jax_enable_x64", True)


@jax.jit
def rearrange_velocity_samples_laplace(test):
    """
    Rearranges elements of the input array 'test' into a new array of length 354.
    Returns the rearranged array.
    """
    # Initialize the output array
    velocity_samples_laplace_jax = jnp.zeros(354, dtype=test.dtype)

    # Assign the first slice
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[0:16].set(test[32:48])

    # Interleave even and odd indices
    even_indices = test[0:32:2]
    odd_indices = test[1:32:2]
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[16:48].set(jnp.concatenate([even_indices, odd_indices], axis=0))

    # Assign the next slice
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[48:64].set(test[304:320])

    # Rearrange test[48:304]
    num_rows = (304 - 48) // 16
    indices = jnp.arange(num_rows * 16).reshape(16, num_rows).T.flatten()
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[64:320].set(test[48:304][indices])

    # Assign test[352:354]
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[320:322].set(test[352:354])

    # Rearrange test[320:352]
    test_test = test[320:352]
    indices_2 = jnp.arange(test_test.size).reshape(-1, 16).T.flatten()
    velocity_samples_laplace_jax = velocity_samples_laplace_jax.at[322:354].set(test_test[indices_2])

    return velocity_samples_laplace_jax

# main function to run the experiment and plot the results
def main(args):
    # Print the input flags
    print(f"Input Flags: Seed: {args.seed}, Linearization? {args.linearized_pred}, # Posterior Samples: {args.samples}, Prior Optimization? {args.optimize_prior}, Optimizer: {args.optimizer}, KFAC? {args.kfac}, Diffrax? {args.diffrax}")
    n_posterior_samples = args.samples
    optimize_prior = args.optimize_prior

    # run with several seeds
    seed = args.seed
    jrandom.PRNGKey(seed)
    rng = jrandom.PRNGKey(seed)
    torch.manual_seed(seed)

    # Save the plots if the flag is set
    if args.savefig:
        savepath = f"plots_seed_{args.seed}_linearized_{args.linearized_pred}_samples_{args.samples}_optimize_prior_{args.optimize_prior}_optimizer_{args.optimizer}_diffrax_{args.diffrax}"
        ## Create a folder to save the plots
        if not os.path.exists("plots"):
            os.makedirs(savepath)
    
    # Load the banana dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_banana_data()

    # Create data loaders
    batch_size_train = 256

    train_loader = create_data_loader(x_train, y_train, batch_size_train, rng, shuffle=True)

    # Define the optimizer
    if args.optimizer == "sgd": # SGD optimizer
        learning_rate = 1e-3
        weight_decay = 1e-2
        max_epoch = 2000

        # Define the optimizer with weight decay
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),  # Apply weight decay
            optax.sgd(learning_rate),  # SGD optimizer
        )
    else:   # AdamW optimizer
        learning_rate = 1e-3
        weight_decay = 1e-3
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        max_epoch = 1500

    num_features = x_train.shape[-1]    # Number of features  
    num_output = 2  # Binary classification
    H = 16  # Number of hidden units
    # Define the multi-layer perceptron network
    model = MLP(num_features=num_features, hidden_size=H, num_output=num_output)
    state = create_train_state(rng, model, optimizer=optimizer) 

    # Train the model
    for epoch in range(max_epoch):
        train_loss = 0.0

        # Training step
        for batch_img, batch_label in train_loader:
            state = train_step(state, batch_img, batch_label)
            train_loss += compute_loss(state.apply_fn(state.params, batch_img), batch_label)
        train_loss /= len(x_train)

        # Validation step
        logits = state.apply_fn(state.params, x_valid)  # Compute logits
        val_pred = jnp.argmax(logits, axis=1)   # Predictions
        val_accuracy = jnp.mean(val_pred == y_valid)    # Compute accuracy

        # Print every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Val accuracy: {val_accuracy:.4f}")

    # Compute the MAP solution
    def get_map_solution(state):
        map_solution, unravel_fn = jax.flatten_util.ravel_pytree(state.params) 
        map_solution = jnp.array(map_solution, dtype=jnp.float64) 
        return map_solution, unravel_fn

    map_solution, unravel_fn = get_map_solution(state)
    state = state.replace(params=unravel_fn(map_solution))  # Update the state with the MAP solution

    N_grid = 100    # Number of grid points
    offset = 2  
    # Define the edges of the grid
    x1min = x_train[:, 0].min() - offset    
    x1max = x_train[:, 0].max() + offset
    x2min = x_train[:, 1].min() - offset    
    x2max = x_train[:, 1].max() + offset

    # Create grid using jnp.linspace and jnp.meshgrid
    x_grid = jnp.linspace(x1min, x1max, N_grid)
    y_grid = jnp.linspace(x2min, x2max, N_grid)
    grid_mesh_x, grid_mesh_y = jnp.meshgrid(x_grid, y_grid, indexing="ij")  # Use 'ij' for matrix indexing
    grid_points = jnp.column_stack((grid_mesh_x.ravel(), grid_mesh_y.ravel()))

    # Computing and plotting the MAP confidence
    logits_map = state.apply_fn(state.params, grid_points)  # Compute logits
    probs_map = jax.nn.softmax(logits_map)  # Apply softmax

    conf = probs_map.max(1)  # Compute the confidence

    # Save the plots if the flag is set, else display the plots
    if args.savefig:
        plot_map_confidence(x_train, y_train, grid_mesh_x, grid_mesh_y, conf, title="Confidence MAP", save_path=f"{savepath}/MAP.pdf")
    else:
        plot_map_confidence(x_train, y_train, grid_mesh_x, grid_mesh_y, conf, title="Confidence MAP")

    ## Quick import of pytorch model for the laplace package!
    model_torch = nn_torch.Sequential(
        nn_torch.Linear(num_features, H),
        torch.nn.Tanh(),
        nn_torch.Linear(H, H),
        torch.nn.Tanh(),
        nn_torch.Linear(H, num_output),
    ).to(dtype=torch.float64)

    layer_mapping = {
        "Dense_0": model_torch[0],  # First Linear layer
        "Dense_1": model_torch[2],  # Second Linear layer
        "Dense_2": model_torch[4],  # Third Linear layer
    }

    # Transfer weights and biases
    for flax_layer, torch_layer in layer_mapping.items():
        # Convert and load Flax weights (transpose to match PyTorch)
        weight = torch.tensor(np.array(state.params["params"][flax_layer]["kernel"]), dtype=torch.float64).T
        # Ensure the weight tensor is contiguous
        torch_layer.weight.data = weight.contiguous()

        # Convert and load Flax bias (no transpose needed)
        bias = torch.tensor(np.array(state.params["params"][flax_layer]["bias"]), dtype=torch.float64)
        # Ensure the bias tensor is contiguous
        torch_layer.bias.data = bias.contiguous()

    # We need to define a torch dataloader quickly
    x_torch_train = torch.from_numpy(np.array(x_train)).to(dtype=torch.float64)     
    y_torch_train = torch.from_numpy(np.array(y_train)).long()      
    train_torch_dataset = torch.utils.data.TensorDataset(x_torch_train, y_torch_train)  # Create a dataset
    train_torch_loader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=265, shuffle=True) # Create a dataloader

    # Fit the Laplace approximation
    print("Fitting Laplace")
    la = Laplace(
        model_torch,
        "classification",
        "all",
        hessian_structure="full",  # we can also use "diag" for a possible speedup but its not significantly faster and less accurate
        prior_precision=2 * weight_decay,
    )
    la.fit(train_torch_loader)

    if optimize_prior:
        la.optimize_prior_precision(method="marglik")

    print("Prior precision we are using")
    print(la.prior_precision)

    scale_tril = jnp.array(la.posterior_scale, dtype=jnp.float64) # Posterior scale
    velocity_samples_laplace = jax.random.multivariate_normal(
        rng,
        mean=jnp.zeros_like(map_solution),
        cov=scale_tril @ scale_tril.T,
        shape=(n_posterior_samples,),
    )
    print(velocity_samples_laplace.shape)

    # Rearrange the velocity samples
    for n in range(n_posterior_samples):
        velocity_samples_laplace = velocity_samples_laplace.at[n, :].set(rearrange_velocity_samples_laplace(velocity_samples_laplace[n, :]))

    #  Now we have the initial velocities  and, therefore, we can consider the manifold
    if optimize_prior:
        lambda_reg = la.prior_precision.item() / 2  # Regularization parameter
    else:
        lambda_reg = weight_decay   # Regularization parameter

    # Create the manifold
    state_model_2 = create_train_state(rng, model, optimizer=optimizer)

    if args.kfac:   # KFAC approximation
        if args.linearized_pred:
            # We have to first compute the f_MAP in both cases
            state = state.replace(params=unravel_fn(map_solution))
            f_MAP = state.apply_fn(state.params, x_train)

            # Linearized manifold
            manifold = LinearizedCrossEntropyManifold_kfac(
                model,
                state_model_2,
                x_train,
                y_train,
                f_MAP=f_MAP,
                theta_MAP=map_solution,
                unravel_fn=unravel_fn,
                batching=False,
                lambda_reg=lambda_reg,
            )
        else:
            # Usual manifold (Non-linearized)
            manifold = CrossEntropyManifold_kfac(
                model,
                state_model_2,
                x_train,
                y_train,
                unravel_fn=unravel_fn,
                batching=False,
                lambda_reg=lambda_reg,
            )
    else:  
        if args.linearized_pred:
            # We have to first compute the f_MAP in both cases
            state = state.replace(params=unravel_fn(map_solution))
            f_MAP = state.apply_fn(state.params, x_train)

            # Linearized manifold
            manifold = LinearizedCrossEntropyManifold(
                state_model_2,
                x_train,
                y_train,
                f_MAP=f_MAP,
                theta_MAP=map_solution,
                unravel_fn=unravel_fn,
                batching=False,
                lambda_reg=lambda_reg,
            )
        else:
            # Usual manifold (Non-linearized)
            manifold = CrossEntropyManifold(
                state_model_2,
                x_train,
                y_train,
                unravel_fn=unravel_fn,
                batching=False,
                lambda_reg=lambda_reg,
            )

    # Now we have the manifold and we can solve the exponential map
    weights_ours = jnp.zeros((n_posterior_samples, len(map_solution))) # Initialize the weights
    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = velocity_samples_laplace[n, :].reshape(-1, 1)   # Reshape the velocity samples
        # Solve the expmap using the diffrax solver if the flag is set, else use the standard scipy solver
        if args.diffrax:    
            final_c, _, failed = geometry_diffrax.expmap(manifold, map_solution.clone(), v)
            _new_weights = final_c
        else:
            curve, failed = geometry.expmap(manifold, map_solution.clone(), v)
            _new_weights = curve(1)[0]
        weights_ours = weights_ours.at[n, :].set(jnp.array(_new_weights.reshape(-1)))

    # We can use the weights for prediction. Depending if I am using linearization or not the prediction looks differently
    if args.linearized_pred:
        state_model_2 = state_model_2.replace(params=unravel_fn(map_solution))
        f_MAP_grid = state_model_2.apply_fn(state_model_2.params, grid_points)
        f_MAP_test = state_model_2.apply_fn(state_model_2.params, x_test)

        def predict(params, datas):
            y_pred = state_model_2.apply_fn(params, datas)
            return y_pred

        linearized_grid_posterior_probabilities = 0
        test_posterior_probabilities = 0

        # Now we can do the same for our method
        for n in range(n_posterior_samples):
            # Get the theta weights we are interested in #
            w_OUR = weights_ours[n, :]  # Weights
            params = unravel_fn(map_solution)   

            diff_weights = (w_OUR - map_solution).astype(jnp.float64)   # Difference in weights

            diff_as_params = unravel_fn(diff_weights)   

            # Compute the predictions
            _, jvp_value_grid = jax.jvp(
                predict,
                (params, grid_points),
                (diff_as_params, jnp.zeros_like(grid_points)),
            )

            f_OUR_grid = f_MAP_grid + jvp_value_grid    

            probs_grid = jax.nn.softmax(f_OUR_grid, axis=1)  # Apply softmax to compute probabilities
            linearized_grid_posterior_probabilities += probs_grid   # Compute the posterior probabilities

        linearized_grid_posterior_probabilities /= n_posterior_samples  # Average the probabilities across samples
        grid_posterior_confidence = linearized_grid_posterior_probabilities.max(1) # Compute the confidence

        if args.savefig:    # Save the plots if the flag is set
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence,
                linearized_grid_posterior_probabilities[:, 0],
                title="Confidence RIEM LA linearized",
                save_path=f"{savepath}/RIEM_LA.pdf",
            )
        else:       # Display the plots
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence,
                linearized_grid_posterior_probabilities[:, 0],
                title="Confidence RIEM LA linearized",
            )

        P_grid_laplace_lin = 0
        P_test_laplace = 0

        ## Now we seek to compute the baseline laplace
        for n in range(n_posterior_samples):
            weights_laplace = velocity_samples_laplace[n, :]
            state_model_2 = state_model_2.replace(params=unravel_fn(map_solution))
            diff_as_params = unravel_fn(weights_laplace)
            _, jvp_value_grid = jax.jvp(
                predict,
                (params, grid_points),
                (diff_as_params, jnp.zeros_like(grid_points)),
            )
            f_laplace_grid = f_MAP_grid + jvp_value_grid
            probs_grid_laplace = jax.nn.softmax(f_laplace_grid, axis=1)
            P_grid_laplace_lin += probs_grid_laplace
        P_grid_laplace_lin /= n_posterior_samples
        P_grid_laplace_conf = P_grid_laplace_lin.max(1)

        if args.savefig:    # Save the plots if the flag is set
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                P_grid_laplace_conf,
                P_grid_laplace_lin[:, 0],
                title="Confidence LAPLACE linearized",
                save_path=f"{savepath}/LA.pdf"
            )
        else:    # Display the plots
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                P_grid_laplace_conf,
                P_grid_laplace_lin[:, 0],
                title="Confidence LAPLACE linearized")

        ## Now consider the test set:
        for n in range(n_posterior_samples):
            # Get the theta weights we are interested in #
            w_OUR = weights_ours[n, :]
            params = unravel_fn(map_solution)

            diff_weights = (w_OUR - map_solution).astype(jnp.float64)

            diff_as_params = unravel_fn(diff_weights)

            _, jvp_value_test = jax.jvp(predict, (params, x_test), (diff_as_params, jnp.zeros_like(x_test)))

            f_OUR_test = f_MAP_test + jvp_value_test

            probs_test = jax.nn.softmax(f_OUR_test, axis=1)
            test_posterior_probabilities += probs_test

        for n in range(n_posterior_samples):
            weights_laplace = velocity_samples_laplace[n, :]
            state_model_2 = state_model_2.replace(params=unravel_fn(map_solution))
            params = unravel_fn(map_solution)
            diff_as_params = unravel_fn(weights_laplace)
            _, jvp_value_test = jax.jvp(predict, (params, x_test), (diff_as_params, jnp.zeros_like(x_test)))

            f_laplace_test = f_MAP_test + jvp_value_test
            probs_test_laplace = jax.nn.softmax(f_laplace_test, axis=1)
            P_test_laplace += probs_test_laplace


    else:   # Non-linearized prediction
        # and then our stuff
        grid_posterior_probabilities = 0
        for n in range(n_posterior_samples):
            # Put the weights in the model
            state = state.replace(params=unravel_fn(weights_ours[n, :]))
            # Compute the predictions
            grid_posterior_probabilities += jax.nn.softmax(state.apply_fn(state.params, grid_points), axis=1)

        grid_posterior_probabilities /= n_posterior_samples # Average the probabilities across samples
        grid_posterior_confidence = grid_posterior_probabilities.max(1)     # Compute the confidence

        if args.savefig:    # Save the plots if the flag is set
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence,
                grid_posterior_probabilities[:, 0],
                title="Confidence RIEM LA",
                save_path=f"{savepath}/RIEM LA.pdf"
            )
        else:       # Display the plots
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence,
                grid_posterior_probabilities[:, 0],
                title="Confidence RIEM LA"
            )

        test_posterior_probabilities = 0
        for n in range(n_posterior_samples):
            # put the weights in the model
            state = state.replace(params=unravel_fn(weights_ours[n, :]))
            # compute the predictions
            test_posterior_probabilities += jax.nn.softmax(state.apply_fn(state.params, x_test), axis=1)

   
        # and then laplace stuff
        grid_posterior_probabilities_la = 0
        for n in range(n_posterior_samples):
            laplace_weights = velocity_samples_laplace[n,:] + map_solution
            # put the weights in the model
            state = state.replace(params=unravel_fn(laplace_weights))
            # compute the predictions
            grid_posterior_probabilities_la += jax.nn.softmax(state.apply_fn(state.params, grid_points), axis=1)

        grid_posterior_probabilities_la /= n_posterior_samples
        grid_posterior_confidence_la = grid_posterior_probabilities_la.max(1)
        if args.savefig:        # Save the plots if the flag is set
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence_la,
                grid_posterior_probabilities_la[:, 0],
                title="Confidence LAPLACE",
                save_path=f"{savepath}/LA.pdf"
            )
        else:       # Display the plots
            plot_confidence(
                x_train,
                y_train,
                grid_mesh_x,
                grid_mesh_y,
                grid_posterior_confidence_la,
                grid_posterior_probabilities_la[:, 0],
                title="Confidence LAPLACE")

        P_test_laplace = 0
        for n in range(n_posterior_samples):
            laplace_weights = velocity_samples_laplace[n,:] + map_solution
            # put the weights in the model
            state = state.replace(params=unravel_fn(laplace_weights))
            # compute the predictions
            P_test_laplace += jax.nn.softmax(state.apply_fn(state.params, x_test), axis=1)

    # We can compute and plot the results

    test_posterior_probabilities /= n_posterior_samples 
    P_test_laplace /= n_posterior_samples

    accuracy_posterior = accuracy(test_posterior_probabilities, y_test) # Compute accuracy
    negative_log_likelihood = nll(test_posterior_probabilities, y_test) # Compute negative log likelihood
    brier_score = brier(test_posterior_probabilities, y_test)   # Compute Brier score

    test_posterior_probabilities_torch = torch.from_numpy(np.array(test_posterior_probabilities))   
    y_test_torch = torch.from_numpy(np.array(y_test))   

    # Compute the Expected Calibration Error (ECE)
    ece = (
        calibration_error(
            test_posterior_probabilities_torch,
            y_test_torch,
            norm="l1",
            task="multiclass",
            num_classes=2,
            n_bins=10,
        )
        * 100
    )
    
    # Compute the Maximum Calibration Error (MCE)
    mce = (
        calibration_error(
            test_posterior_probabilities_torch,
            y_test_torch,
            norm="max",
            task="multiclass",
            num_classes=2,
            n_bins=10,
        )
        * 100
    )

    accuracy_laplace = accuracy(P_test_laplace, y_test)     # Compute accuracy for Laplace
    nll_laplace = nll(P_test_laplace, y_test)   # Compute negative log likelihood for Laplace
    brier_score_laplace = brier(P_test_laplace, y_test)     # Compute Brier score for Laplace

    P_test_laplace_torch = torch.from_numpy(np.array(P_test_laplace))
    # Compute the Expected Calibration Error (ECE) for Laplace
    ece_laplace = (
        calibration_error(
            P_test_laplace_torch,
            y_test_torch,
            norm="l1",
            task="multiclass",
            num_classes=2,
            n_bins=10,
        )
        * 100
    )
    # Compute the Maximum Calibration Error (MCE) for Laplace
    mce_laplace = (
        calibration_error(
            P_test_laplace_torch,
            y_test_torch,
            norm="max",
            task="multiclass",
            num_classes=2,
            n_bins=10,
        )
        * 100
    )

    # ece = calibration_error(test_posterior_probabilities, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    # mce = calibration_error(test_posterior_probabilities, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100


    ## Lets compare to the MAP Estimates too!
    state = state.replace(params=unravel_fn(map_solution))  # Update the state with the MAP solution
    P_test_MAP = jax.nn.softmax(state.apply_fn(state.params, x_test), axis=1)   # Compute the MAP probabilities
    accuracy_MAP = accuracy(P_test_MAP, y_test) # Compute accuracy for MAP
    nll_MAP = nll(P_test_MAP, y_test)   # Compute negative log likelihood for MAP
    brier_MAP = brier(P_test_MAP, y_test)   # Compute Brier score for MAP
    MAP_probs_torch = torch.from_numpy(np.array(P_test_MAP))  
    # Compute the Expected Calibration Error (ECE) for MAP  
    ece_map = calibration_error(MAP_probs_torch, y_test_torch, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    # Compute the Maximum Calibration Error (MCE) for MAP
    mce_map = calibration_error(MAP_probs_torch, y_test_torch, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100

    # Print the results
    print(f"Results MAP: accuracy {accuracy_MAP}, nll {nll_MAP}, brier {brier_MAP}, ECE {ece_map}, MCE {mce_map}")
    print(f"Results RIEM LA: accuracy {accuracy_posterior}, nll {negative_log_likelihood}, brier {brier_score}, ECE {ece}, MCE {mce}")
    print(f"Results LA: accuracy {accuracy_laplace}, nll {nll_laplace}, brier {brier_score_laplace}, ECE {ece_laplace}, MCE {mce_laplace}")


if __name__ == "__main__":
    # Parse the input arguments in the different flags
    parser = argparse.ArgumentParser(description="Geomeatric Approximate Inference (GEOMAI)")
    parser.add_argument("--seed", "-s", type=int, default=230, help="seed")
    parser.add_argument(
        "--optimizer",
        "-optim",
        type=str,
        default="sgd",
        help="optimizer used to train the model",
    )
    parser.add_argument(
        "--optimize_prior",
        "-opt_prior",
        type=bool,
        default=False,
        help="optimize prior",
    )
    parser.add_argument("--samples", "-samp", type=int, default=50, help="number of posterior samples")
    parser.add_argument(
        "--linearized_pred",
        "-lin",
        type=bool,
        default=False,
        help="Linearization for prediction",
    )
    parser.add_argument("--kfac", "-kfac", type=bool, default=False, help="Use the KFAC approximation")
    parser.add_argument("--diffrax", "-diffrax", type=bool, default=False, help="Solve with diffrax instead of scipy")
    parser.add_argument("--savefig", "-savefig", type= bool, default=False, help="Whether figures should be saved")

    args = parser.parse_args()  # Arguments
    main(args)  
