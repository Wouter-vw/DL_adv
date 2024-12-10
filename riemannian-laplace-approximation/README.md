## Original Paper's Code

> Note: The code in this repo is largely from the original paper, which can be found at https://github.com/federicobergamin/riemannian-laplace-approximation. The only additions were functions to time parts of the code, write the results to a csv and to potentially save the plots. The instructions below are just to recreate our tests, for more detailed instructions on runnning single trials please refer to the original repo!

Our virtual environment with all dependencies is given in `geomai.yml`. To create this virtual environment do the following: 
```
conda env create -f geomai.yml
```
Activate the environment with:
```
conda activate geomai
```
Our analysis focused on the banana experiment subset of the original paper. In order to reproduce our analysis, the following commands need to be run:
```
# Creating the shell script
python script_generator_orig.py

# Making the shell script runnable
chmod +x run_banana_experiments_orig.sh

# Running the trials
./run_banana_experiments_orig.sh
```
> Note: Running the script to reproduce our results will take over 12 hours most likely, and will generate many folders containing three different plots each! Make sure you have ample time and disk space to run it!