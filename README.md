## Reimplementation and Extension of "Riemannian Laplace approximations for Bayesian neural networks"

This is the code for our final project in the course DD2412, "Deep Learning, Advanced". The code seeks to initially reimplement the methods described in the paper "Riemannian Laplace approximations for Bayesian neural networks" accepted at NeurIPS 2023. It then seeks to extend this paper by implementing the KFAC approximation as well as a GPU ODE solver, in order to improve computational efficiency. The original authors' code was provided using Torch, the reimplementation was done using Jax, again with efficiency in mind. 

> Note: The efficiency of a GPU solver was tested by comparing the original authors' ODE solving method with a method based on Diffrax. Therefore, the file manifold/geometry.py is taken directly from the original repo: https://github.com/federicobergamin/riemannian-laplace-approximation.

Our virtual environment with all dependencies is given in `deep_learning.yaml`. To create this virtual environment do the following: 
```
conda env create -f deep_learning.yaml
```
Activate the environment with:
```
conda activate deep_learning
```

## Reproducing Results
Our analysis focused on the banana experiment subset of the original paper, however the methods could be applied to any of the datasets provided. In order to reproduce our analysis, the following commands need to be run:
```
# Creating the shell script
python script_generator.py

# Making the shell script runnable
chmod +x run_banana_experiments.sh

# Running the trials
./run_banana_experiments.sh
```

These trials will compare the method using 1500, 2000 and 2500 epochs to train the original MLP. Also, they will run for both optimized and unoptimized priors, linearized and unlinearized methods, with and without the KFAC approximation and finally with and without the GPU ODE solver. 100 posterior samples are always used. 

> Note: Running the script to reproduce our results will take several hours most likely, and will generate 480 folders containing three different plots each! Make sure you have ample time and disk space to run it!

## Single runs
If you are interested in running single runs with specific flags use the format below, where seed, prior optimization, linearization, KFAC, diffrax, epochs and savefig are all specified. 
```
python banana_experiment.py -s 0 -opt_prior True -samp 100 -lin True -kfac True -diffrax True -savefig False -epochs 2000
```
The first printed line will show you what flags are active. The latest results will always be appended to the banana_results.csv and will also display in your standard output. If savefig is set to True, the figures will not be shown but rather saved. If if it set to false the figures will be shown in your standard output, preventing the remainder of the code from running until they are closed. 

