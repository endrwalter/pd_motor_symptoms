#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --nodelist=khayyam
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --job-name=mf_all
#SBATCH --output=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/batchResults/%j_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/batchResults/%j_stdErr.txt
#SBATCH --container-image=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML/enroot_img/cpu_cont.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/neuroart/pd/motor_symptoms_ML
#SBATCH --container-remap-root

printf "Starting dir $(pwd)\n"
printf "Moving to working dir.\n"
cd /storage/DSH/projects/neuroart/pd/motor_symptoms_ML/code_ensemble
printf "current dir $(pwd)\n\n"
source /storage/DSH/projects/neuroart/pd/motor_symptoms_ML/enroot_img/venv/bin/activate

printf "\nLaunching PD script.\n"

python3 main_calibrate.py --config config_motor_f_all.ini &
python3 main_calibrate.py --config config_motor_f_active.ini &
wait
