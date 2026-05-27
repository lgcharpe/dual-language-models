#!/bin/bash
#SBATCH --account=project_465002893
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# set -euo pipefail

mkdir -p slurm_logs

module purge
module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002893/nb-embed/huggingface_cache/
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

#Debug
# export NCCL_DEBUG=INFO

# Required on many Cray Slingshot systems for RCCL over CXI
export FI_CXI_ATS=0
export NCCL_CROSS_NIC=1

# Disable GPU Direct RDMA if it's failing silently during channel setup
export NCCL_NET_GDR_LEVEL=0

# For debugging — keep NCCL_DEBUG=INFO and add:
# export NCCL_DEBUG_SUBSYS=INIT,NET
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"


CMD="python -m dual_language_models.run_pretraining \
    --train_path /scratch/project_465002893/nb-embed/dual-language-models/data/train_tokens \
    --valid_path /scratch/project_465002893/nb-embed/dual-language-models/data/val_tokens \
    --name high_lr \
    --wandb_project australis_hybrid \
    --wandb_entity nbailab \
    --config_file /scratch/project_465002893/nb-embed/dual-language-models/configs/base.json \
    --tokenizer_path /scratch/project_465002893/nb-embed/dual-language-models/tokenizers/tokenizer_australis.json \
    --output_dir /scratch/project_465002893/nb-embed/dual-language-models/checkpoints \
    --hybrid_numerator 3 \
    --hybrid_denominator 4 \
    --max_seq_length 2048 \
    --local_batch_size 8 \
    --global_batch_size 2048 \
    --learning_rate 1e-2 \
    --number_of_tokens 100_000_000_000 \
    --max_steps 25000 \
    --validate_every 500 \
    --validation_steps 4 \
    --scheduler trapezoid \
    --warmup_proportion 0.0 \
    --cooldown_proportion 0.0 \
    --seed 42 \
    --save_every 1000 \
    --checkpoint_style exp \
    --first_checkpoint 64 \
    --weight_decay 0.1 \
    --optimizer muon \
    --optimizer_eps 1e-8 \
    --optimizer_beta1 0.9 \
    --optimizer_beta2 0.98 \
    --momentum 0.95 \
    --max_gradient 1e9 \
    --n_special_tokens 16 \
    --experiment initial_run \
    --n_repetitions 2 \
"

srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run $SIF bash -c \
    "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && $CMD"