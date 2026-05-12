#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --ntasks=1
#SBATCH --output=job-%j.out

# ✅ Initialize Conda

# ✅ Activate environment
source ~/.bashrc
conda init
conda activate test

 
# ✅ Run your script
python run.py
