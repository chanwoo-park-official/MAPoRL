#!/bin/bash

# Function to run a single training command
run_training() {
    config_file=$1
    output_file=$2
    alpha_value=$3
    accelerate launch --config_file config/deepspeed_config/ds_config_zero2.yaml train_ppo_v2_multi_agent.py --config "$config_file" --alpha "$alpha_value" > "$output_file" 2>&1
}

echo "Starting training iteration at $(date)"

# Alpha values to sweep
alpha_values=(0 0.5 1 2)

# Perform the training twice
for iteration in {1..2}; do
    echo "Starting iteration $iteration at $(date)"
    for alpha in "${alpha_values[@]}"; do
        echo "Running with alpha=$alpha for iteration $iteration at $(date)"
        if [ $iteration -eq 1 ]; then
            run_training "config/ppo_config/config_ppo2_multi_agent2_turn3_gsm8k_reloadF.py" "output_gsm1_alpha_${alpha}_iter${iteration}.log" "$alpha"
        else
            run_training "config/ppo_config/config_ppo2_multi_agent2_turn3_gsm8k_reloadT.py" "output_gsm2_alpha_${alpha}_iter${iteration}.log" "$alpha"
        fi
    done
    echo "Completed iteration $iteration at $(date)"
    sleep 30
done

echo "All iterations completed at $(date)"