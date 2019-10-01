#!/usr/bin/env bash

# Set up training environment.
# Feel free to change these as required:
export AICROWD_EVALUATION_NAME=LVAE
export AICROWD_DATASET_NAME=mpi3d_realistic

# Change these only if you know what you're doing:
# Check if the root is set; if not use the location of this script as root
if [ ! -n "${NDC_ROOT+set}" ]; then
  export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
fi
# Rest before submitting!!!!!!
export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
export AICROWD_OUTPUT_PATH=${NDC_ROOT}/scratch/shared
#export AICROWD_OUTPUT_PATH=/data/cs618/scratch/shared
export DISENTANGLEMENT_LIB_DATA=${NDC_ROOT}/scratch/dataset
#${NDC_ROOT}/scratch/dataset
