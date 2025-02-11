#!/usr/bin/env zsh

set -e

for self_improve_round in {0..100}; do
    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 run_self_improve.py \
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python run_self_improve.py \
        --seed=42 \
        --from_pretrained=True \
        --use_character_tokenizer=True \
        --model_id=meta-llama/Llama-3.2-1B \
        --use_lora=True \
        --lora_config='{"r": 64,"lora_alpha": 128}' \
        --num_workers=8 \
        \
        \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=10000000 \
        --num_eval=1024 \
        --n_digits_train='1,17' \
        --op_train='add' \
        --format_train='reverse' \
        --op_dist_train='1' \
        --dynamic_eval_range -3 10 1 \
        --op_eval='add' \
        --format_eval='reverse' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=10000 \
        --fast_self_improve=True \
        --self_improve_round=$self_improve_round \
        --self_improve_ood_threshold=0.99 \
        \
        \
        --resume_from_checkpoint=True \
        --save_total_limit=1 \
        --run_name="" \
        --output_dir=out \
        --do_train=False \
        --do_eval=True \
        --self_improve_stable_steps=0 \
        --self_improve_decay_steps=600 \
        --max_steps=1200 \
        --learning_rate=1e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 480, "num_decay_steps": 600, "min_lr_ratio": 0.01}' \
        --warmup_ratio=0.1 \
        --logging_steps=100 \
        --eval_strategy="steps" \
        --eval_steps=600 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=False \
        --auto_find_batch_size=True \
        --per_device_train_batch_size=64 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=600 \
        --torch_compile=False \
        --bf16=True \
        --tf32=True
done