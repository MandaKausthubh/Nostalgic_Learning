#!/bin/bash

# SEEDS=(0 1 2)
# MODES=("nostalgia" "Adam")
# HESSIAN_DIMS=(8 16 32)
#
# for seed in "${SEEDS[@]}"; do
#   for mode in "${MODES[@]}"; do
#     for k in "${HESSIAN_DIMS[@]}"; do
#
#       echo "Running mode=$mode seed=$seed k=$k"
#
#       python train.py \
#         --mode "$mode" \
#         --seed "$seed" \
#         --hessian_eigenspace_dim "$k" \
#         --nostalgia_dimension "$k" \
#         --device cuda \
#         --batch_size 256 \
#         --learning_rate 5e-4
#
#     done
#   done
# done
#
#
#
#

python -m vision_experiments.nostalgia \
    --mode "nostalgia" \
    --root_dir "~/data" \
    --batch_size 256 \
    --batch_size_for_accumulate 16 \
    --learning_rate 3e-4 \
    --downstream_learning_rate 3e-3 \
    --weight_decay 1e-4 \
    --device "cuda" \
    --hessian_eigenspace_dim 16 \
    --validate_after_steps 100 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --seed 42 \
    --accumulate_mode 'accumulate' \
    --iterations_of_accumulate 32 \
    --num_workers 4 \
    --head_warmup_epochs 10 \
    --base_optimizer "adamw" \

