#!/bin/bash
#SBATCH --job-name=QBO-SVR
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=zs1542@nyu.edu # put your email here if you want emails
#SBATCH --output=/scratch/zs1542/QBO-1d/output_scripts/QBO_%j.out
#SBATCH --error=/scratch/zs1542/QBO-1d/output_scripts/QBO_%j.err
#SBATCH --array=1-840

echo "$SLURM_ARRAY_TASK_ID"

id=$SLURM_ARRAY_TASK_ID
dir=/scratch/zs1542/QBO-1d/experiments_w_grid

echo "Your NetID is: zs1542"
echo "start working"
echo "Mission: Experiments with SVR annual cycle"


mkdir $dir/model_$id
cd $dir/model_$id

singularity exec $nv \
  --overlay /scratch/$USER/my_env/overlay-10GB-400K.ext3:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
  source /ext3/env.sh;
  conda activate qbo;
  python /scratch/zs1542/QBO-1d/main_annual_w_rigor.py -p $id -d $dir
  "

echo "FINISH"