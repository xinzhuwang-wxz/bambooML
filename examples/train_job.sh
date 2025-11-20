#!/bin/bash
#SBATCH --job-name=bambooml
#SBATCH --output=bambooml.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

python -m bambooML.runner.cli train -c bambooML/examples/data.yaml -n bambooML/examples/model.py -i _:bambooML/examples/data.csv --num-epochs 1 --batch-size 2
