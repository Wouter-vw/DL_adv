import itertools
import random

# Generate ten random seeds
random.seed(42)

diffrax = [True, False]
kfac = [True, False]
samples = [100]
seeds = random.sample(range(1, 1000), 10)
linearized_pred = [True, False]
optimize_prior = [True, False]
optimizers = ["sgd"]  # Add more optimizers if needed
epochs = [1500, 2000, 2500]
savefig = [False]

# Generate all combinations
combinations = itertools.product(
    seeds,
    optimizers,
    optimize_prior,
    samples,
    linearized_pred,
    kfac,
    diffrax,
    epochs,
    savefig,
)

# Write the shell script
with open("run_banana_experiments.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.write("set -e\n\n")  # Exit on any error

    for combo in combinations:
        seed, optimizer, opt_prior, samp, lin, kfac, diffrax, epochs, savefig = combo
        cmd = (
            f"python banana_experiment.py "
            f"--seed {seed} "
            f"--optimizer {optimizer} "
            f"--optimize_prior {opt_prior} "
            f"--samples {samp} "
            f"--linearized_pred {lin} "
            f"--kfac {kfac} "
            f"--diffrax {diffrax} "
            f"--savefig {savefig}"
            f"--epochs {epochs}"
        )
        f.write(cmd + "\n")

print("Shell script 'run_banana_experiments.sh' generated successfully.")
