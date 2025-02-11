#!/bin/bash

for round in 1; do
    for seed in 41 42 43; do
        # train
        resume=False
        do_train=True
        num_eval=1024
        # additional_digit=$((30 + round))
        prev_round=$((round - 1))
        prev_round_x3=$((prev_round * 3))
        additional_digit="$((31 + prev_round_x3)),9 $((32 + prev_round_x3)),9 $((33 + prev_round_x3)),9"

        let max_steps=12000
        let num_stable_steps=6000
        let decay_steps=2800

        CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
            --eval_cache_loc='data/maze-wilson' \
            --self_improve_cache_loc='self_improve_data_maze_wilson_nodes_vanilla' \
            --n_digits_b_train='9,10' \
            --n_digits_b_eval='9,10' \
            --additional_train_data_digit="${additional_digit}" \
            \
            \
            --wandb_project='self-improve-maze-wilson-nodes' \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=Inf \
            \
            \
            --ignore_data_skip=True \
            --use_iterable_dataset=True \
            --num_train=5000000 \
            --num_eval=$num_eval \
            --n_digits_a_train='10,31' \
            --op_train='maze' \
            --format_train='wilson' \
            --n_digits_a_eval="10,100,3" \
            --op_eval='maze' \
            --format_eval='wilson' \
            --show_task_ids=False \
            --padding_side='right' \
            --num_self_improve_data=50000 \
            --self_improve_round=$round \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint=False \
            --run_name="round_${round}" \
            --output_dir=out/maze_wilson_nodes_vanilla \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=$max_steps \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 2800, \"min_lr_ratio\": 0.01}" \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.1 \
            --logging_steps=50 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=$resume \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=1 \
            --auto_find_batch_size=True \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
    done
done



for seed in 41 42 43; do
    # train
    for round in {2..19}; do
        resume=False
        do_train=True
        num_eval=1024
        prev_round=$((round - 1))
        prev_round_x3=$((prev_round * 3))
        additional_digit="$((31 + prev_round_x3)),9 $((32 + prev_round_x3)),9 $((33 + prev_round_x3)),9"

        steps=$((12000 + (round - 1) * 4000))
        stable_steps=$((9600 + (round - 1) * 4000))
        decay_steps=1000


        additional_train_data=""
        for i in $(seq 1 $prev_round); do
            j=$(( (i - 1) * 3 ))
            additional_train_data+="self_improve_data_maze_wilson_mv_v2_mult/maze-wilson-(($((31 + j ))-9)-($((32 + j ))-9)-($((33 + j ))-9))-50000-$i-${seed}-mv,"
        done

        additional_train_data="${additional_train_data%,}" # Remove trailing comma

        echo $additional_train_data

        CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
            --use_custom_scheduler=True \
            --reset_optimizer=True \
            --eval_cache_loc='data/maze-wilson' \
            --self_improve_cache_loc='self_improve_data_maze_wilson_nodes_vanilla' \
            --n_digits_b_train='9,10' \
            --n_digits_b_eval='9,10' \
            --additional_train_data_digit="${additional_digit}" \
            \
            \
            --wandb_project='self-improve-maze-wilson-nodes' \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=Inf \
            \
            \
            --ignore_data_skip=True \
            --use_iterable_dataset=True \
            --num_train=5000000 \
            --num_eval=$num_eval \
            --n_digits_a_train='10,31' \
            --op_train='maze' \
            --format_train='wilson' \
            --n_digits_a_eval="10,100,3" \
            --op_eval='maze' \
            --format_eval='wilson' \
            --show_task_ids=False \
            --padding_side='right' \
            --num_self_improve_data=50000 \
            --self_improve_round=$round \
            --additional_train_data="$additional_train_data" \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint="out/maze_wilson_nodes_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-wilson-digits-10_31_-9_10_-seed-${seed}" \
            --run_name="round_${round}" \
            --output_dir=out/maze_wilson_nodes_vanilla \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=$steps \
            --learning_rate=2e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs="{\"num_stable_steps\": $stable_steps, \"num_decay_steps\": $decay_steps, \"min_lr_ratio\": 0.01}" \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.1 \
            --logging_steps=50 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=$resume \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=1 \
            --auto_find_batch_size=True \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True

        # eval
        resume=False
        do_train=False
        num_eval=1024
        steps=$((12000 + (round - 1) * 4000))
        stable_steps=$((9600 + (round - 1) * 4000))
        decay_steps=1000

        CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
            --eval_cache_loc='data/maze-wilson' \
            --self_improve_cache_loc='self_improve_data_maze_wilson_nodes_vanilla' \
            --n_digits_b_train='9,10' \
            --n_digits_b_eval='9,10' \
            \
            \
            --wandb_project='self-improve-maze-wilson-nodes' \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=Inf \
            \
            \
            --use_iterable_dataset=True \
            --num_train=50000 \
            --num_eval=$num_eval \
            --n_digits_a_train='10,31' \
            --op_train='maze' \
            --format_train='wilson' \
            --n_digits_a_eval="10,100,3" \
            --op_eval='maze' \
            --format_eval='wilson' \
            --show_task_ids=False \
            --padding_side='right' \
            --num_self_improve_data=50000 \
            --self_improve_round=$round \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint="out/maze_wilson_nodes_vanilla/round_${round}-llama-384-6-6-1024-SI_round_${round}-wilson-digits-10_31_-9_10_-seed-${seed}" \
            --run_name="round_${round}" \
            --output_dir=out/maze_wilson_nodes_vanilla \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=$steps \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs="{\"num_stable_steps\": $stable_steps, \"num_decay_steps\": $decay_steps, \"min_lr_ratio\": 0.01}" \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.1 \
            --logging_steps=50 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=$resume \
            --per_device_train_batch_size=512 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=2 \
            --auto_find_batch_size=True \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
    done    
done

