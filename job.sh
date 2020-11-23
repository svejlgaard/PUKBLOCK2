#!/bin/bash
SBATCH --job-name=tsne_all
#SBATCH --ntasks=1
SBATCH --mem=16G

echo
echo "SLURM ENVIRONMENT"
echo "-----------------"
env | grep SLURM_ | sort | while read var; do
    echo " * $var"
done

echo
echo "STENO ENVIRONMENT"
echo "-----------------"
echo " * SCRATCH=$SCRATCH"

# Job "steps"
srun pip3 install numpy
srun Data_Visualisation.py
