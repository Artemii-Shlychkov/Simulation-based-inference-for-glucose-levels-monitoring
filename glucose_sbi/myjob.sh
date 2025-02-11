#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=16
#SBATCH --output=/storage/homefs/as25u160/logs/log.txt
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gpu_koch
 
# Your code below this line
module load Anaconda3
module load CUDA/12.2.0
module load Workspace_Home
eval "$(conda shell.bash hook)"
/storage/homefs/as25u160/.conda/envs/myenv/bin/python3.12 /storage/homefs/as25u160/scripts/infer_parameters_2.py --config "default_config_2.yaml" --plot