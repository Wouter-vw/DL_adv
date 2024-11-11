import torch
import math

def accuracy(y_pred, y_true):
    with torch.no_grad():
        correct = (y_pred.argmax(dim=1) == y_true).float()
        acc = correct.mean() * 100
    return acc.item()

def nll(y_pred, y_true):
    """
    Mean Categorical Negative Log-Likelihood. Assumes `y_pred` is a probability vector.
    """
    with torch.no_grad():
        nll_loss = torch.nn.NLLLoss(reduction='mean')
        loss = nll_loss(torch.log(y_pred), y_true)
    return loss.item()

def brier(y_pred, y_true):
    with torch.no_grad():
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(-1)).float()
        loss = torch.mean((y_pred - y_true_one_hot) ** 2)
    return loss.item()

def calibration(pys, y_true, M=15):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    with torch.no_grad():
        conf, preds = torch.max(pys, dim=1)
        accs = preds.eq(y_true)
        bins = torch.linspace(0, 1, M + 1, device=pys.device)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        ece = torch.zeros(1, device=pys.device)
        mce = torch.zeros(1, device=pys.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf >= bin_lower) & (conf < bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                acc_in_bin = accs[in_bin].float().mean()
                avg_conf_in_bin = conf[in_bin].mean()
                gap = torch.abs(avg_conf_in_bin - acc_in_bin)
                ece += gap * prop_in_bin
                mce = torch.max(mce, gap)

        ece = ece.item() * 100
        mce = mce.item() * 100

    return ece, mce

def nlpd_using_predictions(mu_star, var_star, true_target):
    with torch.no_grad():
        nlpd = torch.abs(
            0.5 * math.log(2 * math.pi) + 0.5 * torch.mean(torch.log(var_star) + (true_target - mu_star) ** 2 / var_star)
        )
    return nlpd.item()

def mae(mu_star, true_target):
    with torch.no_grad():
        mae_value = torch.mean(torch.abs(true_target - mu_star))
    return mae_value.item()

def rmse(mu_star, true_target):
    with torch.no_grad():
        rmse_value = torch.sqrt(torch.mean((true_target - mu_star) ** 2))
    return rmse_value.item()

def error_metrics(mu_star, var_star, true_target):
    _rmse = rmse(mu_star, true_target)
    _mae = mae(mu_star, true_target)
    _nlpd = nlpd_using_predictions(mu_star, var_star, true_target)
    return _rmse, _mae, _nlpd

def compute_metrics(args, model, weights_list, test_data, verbose=True, save=None, device="cpu"):
    X_test = test_data["X"].to(device)
    y_test = test_data["y"].to(device)

    metrics_dict = {"accuracy": [], "nll": [], "brier": [], "ece": [], "mce": []}

    for weights in weights_list:
        p_y_test = 0
        for s in range(args.n_posterior_samples):
            # Update model weights
            torch.nn.utils.vector_to_parameters(weights[s, :].float(), model.parameters())
            # Compute predictions
            with torch.no_grad():
                p_y_test += torch.softmax(model(X_test), dim=1)
        p_y_test /= args.n_posterior_samples

        _accuracy = accuracy(p_y_test, y_test)
        _nll = nll(p_y_test, y_test)
        _brier = brier(p_y_test, y_test)
        _ece, _mce = calibration(p_y_test, y_test, M=args.calibration_bins)

        metrics_dict["accuracy"].append(_accuracy)
        metrics_dict["nll"].append(_nll)
        metrics_dict["brier"].append(_brier)
        metrics_dict["ece"].append(_ece)
        metrics_dict["mce"].append(_mce)

    if verbose:
        print_metrics(metrics_dict)

    if save is not None:
        torch.save(metrics_dict, save + "_metrics.pt")
    else:
        print_metrics(metrics_dict)

def compute_metrics_per_sample(args, model, weights_list, test_data, verbose=True, device="cpu"):
    X_test = test_data["X"].to(device)
    y_test = test_data["y"].to(device)

    metrics_list = []
    for weights in weights_list:
        metrics_dict = {"accuracy": [], "nll": [], "brier": [], "ece": [], "mce": []}

        for s in range(args.n_posterior_samples):
            # Update model weights
            torch.nn.utils.vector_to_parameters(weights[s, :].float(), model.parameters())
            # Compute predictions
            with torch.no_grad():
                p_y_test = torch.softmax(model(X_test), dim=1)

            _accuracy = accuracy(p_y_test, y_test)
            _nll = nll(p_y_test, y_test)
            _brier = brier(p_y_test, y_test)
            _ece, _mce = calibration(p_y_test, y_test, M=args.calibration_bins)

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
    for metric, values in metrics_dict.items():
        print(f"> {metric}: {values}")

def print_metrics_per_sample(metrics_list):
    for metrics_dict in metrics_list:
        for metric, values in metrics_dict.items():
            print(f"> {metric}: {values}")