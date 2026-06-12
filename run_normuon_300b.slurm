#!/bin/bash
#SBATCH --account=nn30001k
#SBATCH --job-name=australis_hybrid
#SBATCH --output=logs/multinode_container_%j.out
#SBATCH --error=logs/multinode_container_%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=accel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G

set -euo pipefail

CONTAINER_PATH="/cluster/work/projects/nn30001k/nb-embed/apptainer/dlm_25.09_liger_torch_2.12.sif"
SCRIPT_DIR="/cluster/work/projects/nn30001k/nb-embed/dual-language-models"
TRAINING_SCRIPT="run_pretraining_v2.py \
    --train_path ${SCRIPT_DIR}/data/aurora-open/train_tokens \
    --valid_path ${SCRIPT_DIR}/data/aurora-open/val_tokens \
    --name australis_normuon_v1\
    --wandb_project australis_hybrid \
    --wandb_entity nbailab \
    --config_file ${SCRIPT_DIR}/configs/australis_normuon_v1.json \
    --tokenizer_path ${SCRIPT_DIR}/tokenizers/tokenizer_australis.json \
    --output_dir ${SCRIPT_DIR}/checkpoints \
    --hybrid_numerator 1 \
    --hybrid_denominator 4 \
    --max_seq_length 2048 \
    --local_batch_size 16 \
    --global_batch_size 2048 \
    --learning_rate 4e-3 \
    --adam_learning_rate 4e-3 \
    --number_of_tokens 300_000_000_000 \
    --max_steps 71525 \
    --validate_every 1000 \
    --validation_steps 4 \
    --scheduler flat \
    --warmup_proportion 2500 \
    --cooldown_proportion 0.0 \
    --seed 42 \
    --save_every 1000 \
    --checkpoint_style exp \
    --first_checkpoint 256 \
    --weight_decay 0.1 \
    --optimizer muon \
    --optimizer_eps 1e-8 \
    --optimizer_beta1 0.9 \
    --optimizer_beta2 0.95 \
    --momentum 0.95 \
    --max_gradient 1.0 \
    --n_special_tokens 16 \
    --experiment Australis_NorMuon \
    --n_repetitions 6 \
    --polar_express \
    --kimi_adjust_lr \
    --normuon \
    --coeffs polar_five_iter \
    --ns_steps 5 \
"

export VENV="/cluster/work/projects/nn30001k/nb-embed/dlm_liger_venv.sqsh"
export APPTAINER_VENV="/opt/venvs/venv"

ml reset
ml load NRIS/GPU
ml load NCCL/2.30.4-GCCcore-14.3.0-CUDA-13.0.0

LIBFABRIC_LIB_PATH="${EBROOTLIBFABRIC}/lib"
LIBFABRIC_INCLUDE_PATH="${EBROOTLIBFABRIC}/include"
NCCL_ROOT_PATH="${EBROOTNCCL}"
AWS_OFI_NCCL_LIB_PATH="${EBROOTAWSMINOFIMINNCCL}/lib"
CXI_LIB_PATH="/usr/lib64"

echo "LIBFABRIC_LIB_PATH: ${LIBFABRIC_LIB_PATH}"
echo "LIBFABRIC_INCLUDE_PATH: ${LIBFABRIC_INCLUDE_PATH}"
echo "NCCL_ROOT_PATH: ${NCCL_ROOT_PATH}"
echo "AWS_OFI_NCCL_LIB_PATH: ${AWS_OFI_NCCL_LIB_PATH}"
echo "CXI_LIB_PATH: ${CXI_LIB_PATH}"

HF_ROOT="${SCRIPT_DIR}/hf_cache"
mkdir -p "${HF_ROOT}/hub" "${HF_ROOT}/datasets" "${HF_ROOT}/torch"

cd "${SCRIPT_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
rdzv_endpoint="${head_node_ip}:29500"

echo "Head node: ${head_node}"
echo "Head node IP: ${head_node_ip}"
echo "Rendezvous endpoint: ${rdzv_endpoint}"

srun apptainer exec --nv --writable-tmpfs \
  --bind "${SCRIPT_DIR}:${SCRIPT_DIR}" \
  --bind "${LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
  --bind "${LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
  --bind "${NCCL_ROOT_PATH}:/opt/nccl" \
  --bind "${AWS_OFI_NCCL_LIB_PATH}:/opt/aws-ofi-nccl/lib" \
  --bind "${CXI_LIB_PATH}:${CXI_LIB_PATH}" \
  --pwd "${SCRIPT_DIR}" \
  --env FI_PROVIDER="${FI_PROVIDER:-cxi}" \
  --env FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}" \
  --env NCCL_PROTO="${NCCL_PROTO:-^LL128}" \
  --env LIBFABRIC_HOME="/opt/libfabric" \
  --env NCCL_HOME="/opt/nccl" \
  --env AWS_OFI_NCCL_HOME="/opt/aws-ofi-nccl" \
  --env RDZV_ENDPOINT="${rdzv_endpoint}" \
  --env RDZV_ID="${SLURM_JOB_ID}" \
  --env SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}" \
  --env SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE}" \
  --env TRAINING_SCRIPT="${TRAINING_SCRIPT}" \
  --env HF_HOME="${HF_ROOT}" \
  --env HF_HUB_CACHE="${HF_ROOT}/hub" \
  --env HF_DATASETS_CACHE="${HF_ROOT}/datasets" \
  --env TRANSFORMERS_CACHE="${HF_ROOT}/hub" \
  --env TORCH_HOME="${HF_ROOT}/torch" \
  "${CONTAINER_PATH}" \
  bash -lc 'export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real/lib:${LIBFABRIC_HOME}/lib:${NCCL_HOME}/lib:${AWS_OFI_NCCL_HOME}/lib:/usr/lib64:${LD_LIBRARY_PATH}"; export CPATH="${LIBFABRIC_HOME}/include:${CPATH:-}"; torchrun --nnodes="${SLURM_JOB_NUM_NODES}" --nproc_per_node="${SLURM_GPUS_ON_NODE}" --rdzv_id="${RDZV_ID}" --rdzv_backend=c10d --rdzv_endpoint="${RDZV_ENDPOINT}" ${TRAINING_SCRIPT}'