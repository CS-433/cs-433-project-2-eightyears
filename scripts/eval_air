#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --account=master

echo START BY $USER AT `date`

# Activate Virtual Environment and Load Modules for GPUs
nvidia-smi
module purge
module load gcc cuda cudnn python/2.7 mvapich2
source /home/$USER/venvs/atloc/bin/activate

# Temporary Directory
TEMP=$TMPDIR
TEMP+="/AtLoc-master"

# Recreate the Hierarchy in Fast Temporary Directory
mkdir $TEMP
cp -r /home/$USER/cs433-atloc4topo/AtLoc-master/* $TEMP

# Skip unzipping and moving datasets since they are already copied to the right places
# AtLoc-master/data/
# └── comballaz
#     ├── air (with data from /work/topo/VNAV/Real_Data/comballaz/dji-air2)
#     └── air_synthetic (with unzipped data from /work/topo/VNAV/Synthetic_Data/comballaz/comballaz-air2.zip/comballaz-air2)
# I am also assuming the train_air job has been run beforehand

# Run Code: Create Means, Test Dataset on Last Epoch and Generate Saliency Maps
srun python $TEMP/run.py --dataset comballaz --scene air --model AtLoc --data_dir $TEMP/data --logdir /home/$USER/cs433-atloc4topo/AtLoc-master/logs
wait
srun python $TEMP/eval.py --dataset comballaz --scene air --model AtLoc --gpus 0 --data_dir $TEMP/data --weights /home/$USER/cs433-atloc4topo/AtLoc-master/logs/comballaz_air_AtLoc_False/models/epoch_085.pth.tar --logdir /home/$USER/cs433-atloc4topo/AtLoc-master/logs 
wait
srun python $TEMP/run.py --dataset comballaz --scene air --model AtLoc --gpus 0 --data_dir $TEMP/data --weights /home/$USER/cs433-atloc4topo/AtLoc-master/logs/comballaz_air_AtLoc_False/models/epoch_000.pth.tar --final_weights /home/$USER/cs433-atloc4topo/AtLoc-master/logs/comballaz_air_AtLoc_False/models/epoch_085.pth.tar --logdir /home/$USER/cs433-atloc4topo/AtLoc-master/logs
wait

echo END OF $SLURM_JOB_ID AT `date`
