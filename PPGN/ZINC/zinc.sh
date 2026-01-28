#!/bin/bash

# Define seeds for 4 runs
seeds=(41 95 35 12)

# Determine suffix based on config.yaml
if grep -q "use_rwse:.*True" config.yaml; then
    suffix="_rwse"
elif grep -q "use_lappe:.*True" config.yaml; then
    suffix="_lappe"
elif grep -q "use_subgraph_entropy:.*True" config.yaml; then
    suffix="_von"
else
    suffix=""
fi

result_file="zinc_results${suffix}.txt"

# Clear previous results
echo "Training Results" > "$result_file"

for seed in "${seeds[@]}"; do
    echo "Running with seed $seed..."
    # Use python -u for unbuffered output, tee to show in terminal and save to file
    python -u main.py lr=2e-3 lr_schedule_patience=10 depth=8 out_dim=1 hidden_dim=95 input_dim=3 dataset=ZINC seed=$seed | tee "zinc_seed_${seed}${suffix}.log"
    
    # Extract the final test loss to the result file
    echo "Seed $seed:" >> "$result_file"
    grep "Test Loss" "zinc_seed_${seed}${suffix}.log" | tail -n 1 >> "$result_file"
    echo "--------------------------------" >> "$result_file"
done