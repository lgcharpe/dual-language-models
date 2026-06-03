#!/bin/bash -e
# Sets the base project directory on the HPC cluster.
export PROJECT_DIR="/cluster/work/projects/nn30001k/nb-embed"

# Defines the dual-language-models working directory.
export MyWD="$PROJECT_DIR/dual-language-models"

# Sets the target mount point inside the container.
export CONTAINER_WD="/workspace"

export VENV="$PROJECT_DIR/dlm_liger_venv.sqsh"
export APPTAINER_VENV="opt/venvs/venv"

# Path to the apptainer/ folder containing the container image.
CONTAINER_DIR="$PROJECT_DIR/apptainer"

# Defines the full path to the container image file.
APPTAINER_SIF="${CONTAINER_DIR}/dlm_25.09.sif"

echo $APPTAINER_SIF

# Launches an interactive shell session inside the container.
# The --nv flag enables NVIDIA GPU support.
apptainer shell --nv --overlay $VENV:ro --env VENV_PATH=$APPTAINER_VENV -B $MyWD:$CONTAINER_WD -B $PROJECT_DIR --env MyWD=$MyWD $APPTAINER_SIF

