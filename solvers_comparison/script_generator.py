import itertools
import random
import numpy as np

# Generate ten random seeds
random.seed(42)

diffrax = [True]
# comparison within seeds
init_seed, fin_seed = map(int, input("Enter the range of seeds (separated by a space): ").split())
seeds = np.linspace(init_seed, fin_seed, fin_seed - init_seed + 1, dtype=int)

# solvers method
solver = ['dopri5', 'tsit5', 'bosh3']

print("Tips for the tuning of the step size controllers:")
print(" - pcoeff >= 0.2")
print(" - icoeff >= 0.3")
print(" - pcoeff + icoeff <= 0.7")

init_pcoeff, fin_pcoeff = map(float, input("Enter the range of pcoeffs (separated by a space): ").split())
step_size = 0.1
num_samples = int((fin_pcoeff - init_pcoeff) / step_size) + 1
pcoeff = np.linspace(init_pcoeff, fin_pcoeff, num_samples)
init_icoeff, fin_icoeff = map(float, input("Enter the range of icoeffs (separated by a space): ").split())
num_samples = int((fin_icoeff - init_icoeff) / step_size) + 1
icoeff = np.linspace(init_icoeff, fin_icoeff, num_samples)

kfac = [False]
samples = [100]  
linearized_pred = [False]
optimize_prior = [False]
epochs = [2000]
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
    solver,
    pcoeff,
    icoeff
)

# Write the shell script
with open("run_banana_experiments.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.write("set -e\n\n")  # Exit on any error

    for combo in combinations:
        seed, opt_prior, samp, lin, kfac, diffrax, epochs, savefig, solver, pcoeff, icoeff = combo
        # Construct the command
        # boolean flags
        opt_prior = "--optimize_prior True" if opt_prior else ""
        lin = "--linearized_pred True" if lin else ""
        kfac = "--kfac True" if kfac else ""
        diffrax = "--diffrax True" if diffrax else ""
        savefig = "--savefig True"
        # solvers flags
        solver = f"--solver {solver}"
        pcoeff = f"--pcoeff {pcoeff}"
        icoeff = f"--icoeff {icoeff}"
        # integer flags
        seed = f"--seed {seed}"
        samp = f"--samples {samp}"
        epochs = f"--epochs {epochs}"
        # Construct the command
        cmd = f"python banana_experiment.py {seed} {opt_prior} {samp} {lin} {kfac} {diffrax} {epochs} {savefig} {solver} {pcoeff} {icoeff}"
        f.write(cmd + "\n")

print("Shell script 'run_banana_experiments.sh' generated successfully.")
