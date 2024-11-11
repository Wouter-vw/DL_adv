import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import sklearn.model_selection

from laplace import Laplace
import matplotlib.colors as colors
import seaborn as sns
import geomai.utils.geometry as geometry
from torch import nn
from manifold import CrossEntropyManifold, LinearizedRegressionManifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import sklearn.datasets
from datautils import make_pinwheel_data
from utils.metrics import accuracy, nll, brier, calibration
from sklearn.metrics import brier_score_loss
import argparse
from torchmetrics.functional.classification import calibration_error
from torch.func import grad, jvp, vjp, hessian, jacfwd, jacrev, vmap, functional_call
from functorch_utils import get_params_structure, stack_gradient, custom_hvp, stack_gradient2
import os


def main(args):
    # sns.set_style('darkgrid')
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed: ", seed)

    shuffle = True
    # now I have to load the banana dataset
    filen = os.path.join("data", "banana", "banana.csv")
    Xy = np.loadtxt(filen, delimiter=",")
    x_train, y_train = Xy[:, :-1], Xy[:, -1]
    x_test, y_test = Xy[:0, :-1], Xy[:0, -1]
    y_train, y_test = y_train - 1, y_test - 1

    split_train_size = 0.7
    strat = None
    x_full, y_full = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
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

    # Convert data to tensors and move to device
    x_train = torch.from_numpy(x_train).float().to(device)
    x_valid = torch.from_numpy(x_valid).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    y_train = torch.from_numpy(y_train).long().reshape(-1).to(device)
    y_valid = torch.from_numpy(y_valid).long().reshape(-1).to(device)
    y_test = torch.from_numpy(y_test).long().reshape(-1).to(device)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=265, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)

    num_features = x_train.shape[-1]
    num_output = 2
    H = 16

    model = nn.Sequential(
        nn.Linear(num_features, H),
        torch.nn.Tanh(),
        nn.Linear(H, H),
        torch.nn.Tanh(),
        nn.Linear(H, num_output),
    ).to(device)

    # Compile the model for optimization
    model = torch.compile(model)

    if args.optimizer == "sgd":
        weight_decay = 1e-2

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)

        max_epoch = 2500
    else:
        weight_decay = 1e-3

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

        max_epoch = 1500

    loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(max_epoch):
        train_loss = 0
        for batch_img, batch_label in train_loader:
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)

            y_prob = model(batch_img)

            loss = loss_criterion(y_prob, batch_label)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # now I can evaluate my model on the validation set
        with torch.no_grad():
            valid_accuracy = 0
            for batch_img_valid, batch_label_valid in valid_loader:
                batch_img_valid = batch_img_valid.to(device)
                batch_label_valid = batch_label_valid.to(device)

                y_valid_prob = model(batch_img_valid)
                valid_pred = torch.argmax(y_valid_prob, dim=1)
                valid_accuracy += (valid_pred == batch_label_valid.view(-1)).int().sum()

            valid_accuracy = valid_accuracy / len(x_valid)
            train_loss = train_loss / len(x_train)

        if (epoch + 1) % 100 == 0:
            print("Epoch: {}, Train loss: {}, Valid acc: {}".format(epoch + 1, train_loss, valid_accuracy))

    # at the end of the training I can get the map solution
    map_solution = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    torch.nn.utils.vector_to_parameters(map_solution, model.parameters())

    N_grid = 100
    offset = 2
    x1min = x_train[:, 0].min() - offset
    x1max = x_train[:, 0].max() + offset
    x2min = x_train[:, 1].min() - offset
    x2max = x_train[:, 1].max() + offset

    x_grid = torch.linspace(x1min, x1max, N_grid)
    y_grid = torch.linspace(x2min, x2max, N_grid)
    XX1, XX2 = torch.meshgrid(x_grid, y_grid, indexing='ij')
    X_grid = torch.column_stack((XX1.flatten(), XX2.flatten())).to(device)

    # computing and plotting the MAP confidence
    with torch.no_grad():
        probs_map = torch.softmax(model(X_grid), dim=1).cpu().numpy()

    conf = probs_map.max(1)

    plt.contourf(
        XX1.cpu(),
        XX2.cpu(),
        conf.reshape(N_grid, N_grid),
        alpha=0.8,
        antialiased=True,
        cmap="Blues",
        levels=np.arange(0.0, 1.01, 0.1),
    )
    plt.colorbar()
    plt.scatter(
        x_train[:, 0][y_train == 0].cpu(),
        x_train[:, 1][y_train == 0].cpu(),
        c="orange",
        edgecolors="black",
        s=45,
        alpha=1,
    )
    plt.scatter(
        x_train[:, 0][y_train == 1].cpu(),
        x_train[:, 1][y_train == 1].cpu(),
        c="violet",
        edgecolors="black",
        s=45,
        alpha=1,
    )
    plt.title("Confidence MAP")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

    print("Fitting Laplace")
    la = Laplace(
        model,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        prior_precision=2 * weight_decay,
    )
    la.fit(train_loader)

    if optimize_prior:
        la.optimize_prior_precision(method="marglik")

    print("Prior precision we are using")
    print(la.prior_precision)

    # Get samples from the Laplace approximation
    if subset_of_weights == "last_layer":
        if hessian_structure == "diag":
            n_last_layer_weights = num_output * H + num_output
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().cpu().numpy()
        else:
            n_last_layer_weights = num_output * H + num_output
            dist = MultivariateNormal(loc=torch.zeros(n_last_layer_weights, device=device), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,)).detach().cpu().numpy()
    else:
        if hessian_structure == "diag":
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().cpu().numpy()
        else:
            dist = MultivariateNormal(loc=torch.zeros_like(map_solution, device=device), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,)).detach().cpu().numpy()
            print(V_LA.shape)

    # Prepare the manifold and solve the exponential map
    if args.linearized_pred:
        torch.nn.utils.vector_to_parameters(map_solution, model.parameters())

        with torch.no_grad():
            f_MAP = model(x_train)

        if subset_of_weights == "last_layer":
            weights_ours = torch.zeros(n_posterior_samples, len(map_solution), device=device)

            MAP = map_solution.clone()
            feature_extractor_map = MAP[0:-n_last_layer_weights]
            ll_map = MAP[-n_last_layer_weights:]
            print(feature_extractor_map.shape)
            print(ll_map.shape)

            feature_extractor_model = nn.Sequential(
                nn.Linear(num_features, H),
                torch.nn.Tanh(),
                nn.Linear(H, H),
                torch.nn.Tanh(),
            ).to(device)
            ll = nn.Linear(H, num_output).to(device)

            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            with torch.no_grad():
                R = feature_extractor_model(x_train)

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
            model2 = nn.Sequential(
                nn.Linear(num_features, H),
                torch.nn.Tanh(),
                nn.Linear(H, H),
                torch.nn.Tanh(),
                nn.Linear(H, num_output),
            ).to(device)

            if batch_data:
                new_dataset = torch.utils.data.TensorDataset(x_train, y_train, f_MAP)
                new_train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=50, shuffle=True)

            if optimize_prior:
                if batch_data:
                    manifold = linearized_cross_entropy_manifold(
                        model2,
                        new_train_loader,
                        y=None,
                        f_MAP=f_MAP,
                        theta_MAP=map_solution,
                        batching=True,
                        lambda_reg=la.prior_precision.item() / 2,
                    )
                else:
                    manifold = linearized_cross_entropy_manifold(
                        model2,
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
                        model2,
                        new_train_loader,
                        y=None,
                        f_MAP=f_MAP,
                        theta_MAP=map_solution,
                        batching=True,
                        lambda_reg=weight_decay,
                    )
                else:
                    manifold = linearized_cross_entropy_manifold(
                        model2,
                        x_train,
                        y_train,
                        f_MAP=f_MAP,
                        theta_MAP=map_solution,
                        batching=False,
                        lambda_reg=weight_decay,
                    )
    else:
        if subset_of_weights == "last_layer":
            weights_ours = torch.zeros(n_posterior_samples, len(map_solution), device=device)

            MAP = map_solution.clone()
            feature_extractor_map = MAP[0:-n_last_layer_weights]
            ll_map = MAP[-n_last_layer_weights:]
            print(feature_extractor_map.shape)
            print(ll_map.shape)

            feature_extractor_model = nn.Sequential(
                nn.Linear(num_features, H),
                torch.nn.Tanh(),
                nn.Linear(H, H),
                torch.nn.Tanh(),
            ).to(device)
            ll = nn.Linear(H, num_output).to(device)

            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            with torch.no_grad():
                R = feature_extractor_model(x_train)

            if optimize_prior:
                manifold = cross_entropy_manifold(
                    ll, R, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
                )
            else:
                manifold = cross_entropy_manifold(ll, R, y_train, batching=False, lambda_reg=weight_decay)
        else:
            model2 = nn.Sequential(
                nn.Linear(num_features, H),
                torch.nn.Tanh(),
                nn.Linear(H, H),
                torch.nn.Tanh(),
                nn.Linear(H, num_output),
            ).to(device)

            if optimize_prior:
                if batch_data:
                    manifold = cross_entropy_manifold(
                        model2, train_loader, y=None, batching=True, lambda_reg=la.prior_precision.item() / 2
                    )
                else:
                    manifold = cross_entropy_manifold(
                        model2, x_train, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
                    )
            else:
                if batch_data:
                    manifold = cross_entropy_manifold(
                        model2, train_loader, y=None, batching=True, lambda_reg=weight_decay
                    )
                else:
                    manifold = cross_entropy_manifold(
                        model2, x_train, y_train, batching=False, lambda_reg=weight_decay
                    )

    # Solve the exponential map
    weights_ours = torch.zeros(n_posterior_samples, len(map_solution), device=device)
    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = V_LA[n, :].reshape(-1, 1)

        if subset_of_weights == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            _new_ll_weights = curve(1)[0]
            _new_weights = torch.cat(
                (feature_extractor_map.view(-1), torch.from_numpy(_new_ll_weights).float().view(-1).to(device)), dim=0
            )
            weights_ours[n, :] = _new_weights.view(-1)
            torch.nn.utils.vector_to_parameters(_new_weights, model.parameters())

        else:
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
            weights_ours[n, :] = torch.from_numpy(_new_weights.reshape(-1)).to(device)

    # Get the LA weights
    weights_LA = torch.zeros(n_posterior_samples, len(map_solution), device=device)

    for n in range(n_posterior_samples):
        if subset_of_weights == "last_layer":
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float().to(device) + ll_map.clone()
            laplace_weigths = torch.cat((feature_extractor_map.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths
        else:
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float().to(device) + map_solution
            weights_LA[n, :] = laplace_weigths

    # Prediction
    if args.linearized_pred:
        if subset_of_weights == "last_layer":
            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            with torch.no_grad():
                R_MAP_grid = feature_extractor_model(X_grid).clone()
                R_MAP_test = feature_extractor_model(x_test)

            with torch.no_grad():
                f_MAP_grid = ll(R_MAP_grid).clone()
                f_MAP_test = ll(R_MAP_test)

            def predict(params, data):
                return functional_call(ll, params, data)

            P_grid_LAPLACE_lin = 0
            P_grid_OURS_lin = 0

            for n in range(n_posterior_samples):
                w_LA = weights_LA[n, :]
                w_ll_LA = w_LA[-n_last_layer_weights:]

                params = {name: param for name, param in zip(ll.state_dict().keys(), ll_map)}
                diff_weights = w_ll_LA - ll_map

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    lambda p: predict(p, R_MAP_grid),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_LA_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_LA_grid, dim=1)
                P_grid_LAPLACE_lin += probs_grid.detach().cpu().numpy()

            for n in range(n_posterior_samples):
                w_OUR = weights_ours[n, :]
                w_ll_OUR = w_OUR[-n_last_layer_weights:]

                params = {name: param for name, param in zip(ll.state_dict().keys(), ll_map)}
                diff_weights = w_ll_OUR - ll_map

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    lambda p: predict(p, R_MAP_grid),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_OUR_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_OUR_grid, dim=1)
                P_grid_OURS_lin += probs_grid.detach().cpu().numpy()

        else:
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            with torch.no_grad():
                f_MAP_grid = model2(X_grid).clone()
                f_MAP_test = model2(x_test)

            def predict(params, data):
                return functional_call(model2, params, data)

            P_grid_LAPLACE_lin = 0
            P_grid_OURS_lin = 0
            P_test_OURS = 0
            P_test_LAPLACE = 0

            for n in range(n_posterior_samples):
                w_LA = weights_LA[n, :]
                params = {name: param for name, param in zip(model2.state_dict().keys(), map_solution)}
                diff_weights = w_LA - map_solution

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    lambda p: predict(p, X_grid),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_LA_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_LA_grid, dim=1)
                P_grid_LAPLACE_lin += probs_grid.detach().cpu().numpy()

            for n in range(n_posterior_samples):
                w_OUR = weights_ours[n, :]
                params = {name: param for name, param in zip(model2.state_dict().keys(), map_solution)}
                diff_weights = w_OUR - map_solution

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_grid = jvp(
                    lambda p: predict(p, X_grid),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_OUR_grid = f_MAP_grid + jvp_value_grid

                probs_grid = torch.softmax(f_OUR_grid, dim=1)
                P_grid_OURS_lin += probs_grid.detach().cpu().numpy()

            P_grid_LAPLACE_lin /= n_posterior_samples

            P_grid_LAPLACE_conf = P_grid_LAPLACE_lin.max(1)

            plt.contourf(
                XX1.cpu(),
                XX2.cpu(),
                P_grid_LAPLACE_conf.reshape(N_grid, N_grid),
                alpha=0.8,
                antialiased=True,
                cmap="Blues",
                levels=np.arange(0.0, 1.01, 0.1),
                zorder=-10,
            )

            plt.scatter(
                x_train[:, 0][y_train == 0].cpu(),
                x_train[:, 1][y_train == 0].cpu(),
                c="orange",
                edgecolors="black",
                s=45,
                alpha=1.0,
                zorder=10,
            )
            plt.scatter(
                x_train[:, 0][y_train == 1].cpu(),
                x_train[:, 1][y_train == 1].cpu(),
                c="violet",
                edgecolors="black",
                s=45,
                alpha=1.0,
                zorder=10,
            )
            plt.contour(
                XX1.cpu(),
                XX2.cpu(),
                P_grid_LAPLACE_lin[:, 0].reshape(N_grid, N_grid),
                levels=[0.5],
                colors="k",
                alpha=0.5,
                zorder=0,
            )
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title("All weights, full Hessian approx - Confidence LA linearized")
            plt.show()

            P_grid_OURS_lin /= n_posterior_samples
            P_grid_OUR_conf = P_grid_OURS_lin.max(1)

            plt.contourf(
                XX1.cpu(),
                XX2.cpu(),
                P_grid_OUR_conf.reshape(N_grid, N_grid),
                alpha=0.8,
                antialiased=True,
                cmap="Blues",
                levels=np.arange(0.0, 1.01, 0.1),
                zorder=-10,
            )
            plt.scatter(
                x_train[:, 0][y_train == 0].cpu(),
                x_train[:, 1][y_train == 0].cpu(),
                c="orange",
                edgecolors="black",
                s=45,
                alpha=1,
                zorder=10,
            )
            plt.scatter(
                x_train[:, 0][y_train == 1].cpu(),
                x_train[:, 1][y_train == 1].cpu(),
                c="violet",
                edgecolors="black",
                s=45,
                alpha=1,
                zorder=10,
            )
            plt.contour(
                XX1.cpu(),
                XX2.cpu(),
                P_grid_OURS_lin[:, 0].reshape(N_grid, N_grid),
                levels=[0.5],
                colors="k",
                alpha=0.5,
                zorder=0,
            )
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title("All weights, full Hessian approx - Confidence OUR linearized")
            plt.show()

            # Compute test predictions
            for n in range(n_posterior_samples):
                w_LA = weights_LA[n, :]
                params = {name: param for name, param in zip(model2.state_dict().keys(), map_solution)}
                diff_weights = w_LA - map_solution

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_test = jvp(
                    lambda p: predict(p, x_test),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_LA_test = f_MAP_test + jvp_value_test

                probs_test = torch.softmax(f_LA_test, dim=1)
                P_test_LAPLACE += probs_test.detach()

            for n in range(n_posterior_samples):
                w_OUR = weights_ours[n, :]
                params = {name: param for name, param in zip(model2.state_dict().keys(), map_solution)}
                diff_weights = w_OUR - map_solution

                diff_params = get_params_structure(diff_weights, params)

                _, jvp_value_test = jvp(
                    lambda p: predict(p, x_test),
                    (params,),
                    (diff_params,),
                    strict=False,
                )

                f_OUR_test = f_MAP_test + jvp_value_test

                probs_test = torch.softmax(f_OUR_test, dim=1)
                P_test_OURS += probs_test.detach()

    else:
        P_grid_LAPLACE = 0
        for n in tqdm(range(n_posterior_samples), desc="Computing Laplace samples"):
            torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
            with torch.no_grad():
                P_grid_LAPLACE += torch.softmax(model(X_grid), dim=1).cpu().numpy()

        P_grid_LAPLACE /= n_posterior_samples

        P_grid_LAPLACE_conf = P_grid_LAPLACE.max(1)

        plt.contourf(
            XX1.cpu(),
            XX2.cpu(),
            P_grid_LAPLACE_conf.reshape(N_grid, N_grid),
            alpha=0.8,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.0, 1.01, 0.1),
            zorder=-10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 0].cpu(),
            x_train[:, 1][y_train == 0].cpu(),
            c="orange",
            edgecolors="black",
            s=45,
            alpha=1.0,
            zorder=10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 1].cpu(),
            x_train[:, 1][y_train == 1].cpu(),
            c="violet",
            edgecolors="black",
            s=45,
            alpha=1.0,
            zorder=10,
        )
        plt.contour(
            XX1.cpu(),
            XX2.cpu(),
            P_grid_LAPLACE[:, 0].reshape(N_grid, N_grid),
            levels=[0.5],
            colors="k",
            alpha=0.5,
            zorder=0,
        )
        plt.title("All weights, full Hessian approx - Confidence LA")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

        P_grid_OUR = 0
        for n in tqdm(range(n_posterior_samples), desc="Computing OUR samples"):
            torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
            with torch.no_grad():
                P_grid_OUR += torch.softmax(model(X_grid), dim=1).cpu().numpy()

        P_grid_OUR /= n_posterior_samples
        P_grid_OUR_conf = P_grid_OUR.max(1)

        plt.contourf(
            XX1.cpu(),
            XX2.cpu(),
            P_grid_OUR_conf.reshape(N_grid, N_grid),
            alpha=0.8,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.0, 1.01, 0.1),
            zorder=-10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 0].cpu(),
            x_train[:, 1][y_train == 0].cpu(),
            c="orange",
            edgecolors="black",
            s=45,
            alpha=1,
            zorder=10,
        )
        plt.scatter(
            x_train[:, 0][y_train == 1].cpu(),
            x_train[:, 1][y_train == 1].cpu(),
            c="violet",
            edgecolors="black",
            s=45,
            alpha=1,
            zorder=10,
        )
        plt.contour(
            XX1.cpu(),
            XX2.cpu(),
            P_grid_OUR[:, 0].reshape(N_grid, N_grid),
            levels=[0.5],
            colors="k",
            alpha=0.5,
            zorder=0,
        )
        plt.title("All weights, full Hessian approx - Confidence OURS")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

        P_test_LAPLACE = 0
        for n in tqdm(range(n_posterior_samples), desc="Computing Laplace test predictions"):
            torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
            with torch.no_grad():
                P_test_LAPLACE += torch.softmax(model(x_test), dim=1)

        P_test_OURS = 0
        for n in tqdm(range(n_posterior_samples), desc="Computing OUR test predictions"):
            torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
            with torch.no_grad():
                P_test_OURS += torch.softmax(model(x_test), dim=1)

    # Compute metrics
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

    # Create the results dictionary
    dict_MAP = {"Accuracy": accuracy_MAP, "NLL": nll_MAP, "Brier": brier_MAP, "ECE": ece_map, "MCE": mce_map}
    dict_LA = {"Accuracy": accuracy_LA, "NLL": nll_LA, "Brier": brier_LA, "ECE": ece_la, "MCE": mce_la}
    dict_OUR = {"Accuracy": accuracy_OURS, "NLL": nll_OUR, "Brier": brier_OURS, "ECE": ece_our, "MCE": mce_our}

    final_dict = {"results_MAP": dict_MAP, "results_LA": dict_LA, "results_OUR": dict_OUR}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometric Approximate Inference (GEOMAI)")
    parser.add_argument("--seed", "-s", type=int, default=230, help="seed")
    parser.add_argument("--optimizer", "-optim", type=str, default="sgd", help="Optimizer used to train the model")

    parser.add_argument("--optimize_prior", "-opt_prior", action='store_true', help="Optimize prior")
    parser.add_argument("--batch_data", "-batch", action='store_true', help="Batch data")

    parser.add_argument("--structure", "-str", type=str, default="full", help="Hessian structure for Laplace")
    parser.add_argument("--subset", "-sub", type=str, default="all", help="Subset of weights for Laplace")
    parser.add_argument("--samples", "-samp", type=int, default=50, help="Number of posterior samples")
    parser.add_argument("--linearized_pred", "-lin", action='store_true', help="Linearization for prediction")
    parser.add_argument(
        "--expmap_different_batches",
        "-batches",
        action='store_true',
        help="Solve exponential map using only a batch of the data",
    )
    parser.add_argument(
        "--test_all",
        "-test_all",
        action='store_true',
        help="Use the validation set for evaluation",
    )

    args = parser.parse_args()
    main(args)