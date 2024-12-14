#!/bin/bash

set -e

python banana_experiment.py --seed 0  --samples 100   --diffrax True --epochs 2000 --savefig True --solver dopri5 --pcoeff 0.2 --icoeff 0.3
python banana_experiment.py --seed 0  --samples 100   --diffrax True --epochs 2000 --savefig True --solver tsit5 --pcoeff 0.2 --icoeff 0.3
python banana_experiment.py --seed 0  --samples 100   --diffrax True --epochs 2000 --savefig True --solver bosh3 --pcoeff 0.2 --icoeff 0.3
