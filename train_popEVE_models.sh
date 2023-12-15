#!/bin/bash

# Add line here to activate conda environment, or make sure to activate conda environment before running this script.

export mapping_file='example_mapping.csv'
export losses_and_lengthscales_directory='./results/losses_and_lengthscales/'
export scores_directory='./results/scores/'
export states_directory='./results/states/'

# Check if directories exist and create them if not
mkdir -p "$losses_and_lengthscales_directory"
mkdir -p "$scores_directory"
mkdir -p "$states_directory"

# Loop through indices and run training script
for index in {1..3}; do
    python train_popEVE.py \
    --mapping_file "$mapping_file" \
    --gene_index "$index" \
    --losses_dir "$losses_and_lengthscales_directory" \
    --scores_dir "$scores_directory" \
    --model_states_dir "$states_directory"
done