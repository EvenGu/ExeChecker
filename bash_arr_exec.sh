#!/bin/bash -l
#$ -j y
#$ -l h_rt=24:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=48G
##$ -l gpu_type=A6000
##$ -l buyin

# job a name
#$ -N exP_D
# output file name
##$ -o scc_outputs/exe.qlog
# Submit an array job with 10 tasks 
#$ -t 1-10


module purge
module load miniconda
conda activate ~/myResearch/miniconda3/envs/det2
# python --version
# env|grep -i cuda

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
#export PYTHONPATH=$PWD
#echo $PYTHONPATH

exe_id=$((SGE_TASK_ID-1))
echo "exercise_class: ${exe_id}"
python trainMulti_perExe.py \
    --config config/execheck_Multi_perExe_decouple1.yaml \
    -exercise_class ${exe_id} \
    -model_saved_name "./checkpoints_perExe_D/execheck_stgat/exe${exe_id}" \
    # -last_model "./checkpoints_perExe/execheck_stgat/exe${exe_id}_r.pth" \
    # -val_first True

