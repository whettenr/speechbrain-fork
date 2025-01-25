#!/bin/bash

#SBATCH --job-name=prep_lebench   # nom du job
#SBATCH --account=dha@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/prep_ll_md_%j.log  # log file


module load pytorch-gpu/py3/2.0.1
conda activate ft-sb

cd /gpfswork/rech/nkp/uaj64gk/growth/prep_lebench

python prep_sm.py