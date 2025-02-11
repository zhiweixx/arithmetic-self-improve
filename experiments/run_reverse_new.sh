#!/usr/bin/env zsh

set -e

for self_improve_round in {0..100}; do
    # CUDA_VISIBLE_DEVICES=0,1  WANDB_MODE=online torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 run_self_improve.py \
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python run_self_improve.py \
        --seed=42 \
        --architecture=llama \
        --from_pretrained=False \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        --rope_theta=Inf \
        --num_workers=4 \
        \
        \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=1024 \
        --n_digits_train='1,11' \
        --op_train='reverse' \
        --format_train='reverse' \
        --op_dist_train='1' \
        --dynamic_eval_range -4 11 2 \
        --op_eval='reverse' \
        --format_eval='reverse' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=20000 \
        --self_improve_round=$self_improve_round \
        \
        \
        --resume_from_checkpoint=False \
        --save_total_limit=1 \
        --run_name="" \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --self_improve_stable_steps=0 \
        --self_improve_decay_steps=500 \
        --max_steps=2000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 1300, "num_decay_steps": 500, "min_lr_ratio": 0.01}' \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=100 \
        --eval_strategy="steps" \
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=False \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=False \
        --bf16=True \
        --tf32=True
done
