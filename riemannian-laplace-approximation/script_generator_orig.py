import itertools
import random

# Generate ten random seeds
random.seed(42)

seeds = random.sample(range(1, 1000), 10)
optimize_prior = [True, False]
samples = [100]
linearized_pred = [True, False]
savefig = [False]

# Generate all combinations
combinations = itertools.product(
    seeds,
    optimize_prior,
    samples,
    linearized_pred,
)

# Write the shell script
with open("run_banana_experiments_orig.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.write("set -e\n\n")  # Exit on any error

    for combo in combinations:
        seed, opt_prior, samp, lin = combo
        # Construct the command
        # boolean flags
        opt_prior = "--optimize_prior True" if opt_prior else ""
        lin = "--linearized_pred True" if lin else ""
        savefig = "--savefig True"
        # integer flags
        seed = f"--seed {seed}"
        samp = f"--samples {samp}"
        # Construct the command
        cmd = f"python banana_experiment.py {seed} {opt_prior} {samp} {lin} {savefig}"
        f.write(cmd + "\n")

print("Shell script 'run_banana_experiments_orig.sh' generated successfully.")
