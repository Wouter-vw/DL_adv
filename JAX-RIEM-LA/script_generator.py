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
epochs = [1500, 2000, 2500, 3000]
savefig = [True]

# Generate all combinations
combinations = itertools.product(
    seeds,
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
        seed, opt_prior, samp, lin, kfac, diffrax, epochs, savefig = combo
        # Construct the command
        # boolean flags
        opt_prior = "--optimize_prior True" if opt_prior else ""
        lin = "--linearized_pred True" if lin else ""
        kfac = "--kfac True" if kfac else ""
        diffrax = "--diffrax True" if diffrax else ""
        savefig = "--savefig True"
        # integer flags
        seed = f"--seed {seed}"
        samp = f"--samples {samp}"
        epochs = f"--epochs {epochs}"
        # Construct the command
        cmd = f"python banana_experiment.py {seed} {opt_prior} {samp} {lin} {kfac} {diffrax} {epochs} {savefig}"
        f.write(cmd + "\n")

print("Shell script 'run_banana_experiments.sh' generated successfully.")
