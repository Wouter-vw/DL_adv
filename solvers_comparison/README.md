## Virtual environment

Our virtual environment with all dependencies is given in `deep_learning.yaml`. To create this virtual environment do the following: 
```
conda env create -f deep_learning.yaml
```
Activate the environment with:
```
conda activate deep_learning
```

## Solvers comparison and step size controller tuning
We tested three main ODE solvers from the diffrax library (https://docs.kidger.site/diffrax/api/solvers/ode_solvers/): Dopri5, Tsist5, Bosh3. The comparison can be done across multiple seeds and by trying different values for the step size controllers pcoeff (proportional part) and icoeff (integral part). In order to reproduce the comparison, the following commands need to be run:
```
# Creating the shell script
python script_generator.py

# Making the shell script runnable
chmod +x run_banana_experiments.sh

# Running the trials
./run_banana_experiments.sh
```
To have a better comparison, the slowest setup has been used, namely with unoptimized priors, unlinearized methods and without KFAC. The epochs fixed at 2000 and 100 posterior samples are always used. 

> Note: Running the script to reproduce the comparison will take on average 5/6 min per command line in run_banana_experiments.sh.
## Single runs
If you are interested in running single runs with specific flags use the format below, where seed, prior optimization, linearization, KFAC, diffrax, epochs and savefig are all specified. 
```
python banana_experiment.py -s 0 -opt_prior False -samp 100 -lin True -kfac False -diffrax True -savefig False -epochs 2000 -solver dopri5 -pcoeff 0.2 -icoeff 0.3
```
The first printed line will show you what flags are active. The latest results will always be appended to the banana_results.csv and will also display in your standard output. If savefig is set to True, the figures will not be shown but rather saved. If it set to false the figures will be shown in your standard output, preventing the remainder of the code from running until they are closed. 

## Analysis reported 
You can find our comparisons between different solvers and step size controllers in the .csv files: comparison_fastest_pcoeff_icoeff, comparison_stepsizecontroller, solvers_comparison(4seeds) and solvers_comparison(10seeds).
