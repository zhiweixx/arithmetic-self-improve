for seed in 41 42 43 44 45; do
    resume=False
    do_train=True
    num_eval=1024
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --self_improve_cache_loc='self_improve_data_forward2' \
        --wandb_project='self-improve-forward' \
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
        --num_train=2000000 \
        --num_eval=$num_eval \
        --n_digits_train='1,11' \
        --op_train='add' \
        --format_train='forward2' \
        --op_dist_train='1' \
        --n_digits_eval='1,21,1' \
        --op_eval='add' \
        --format_eval='forward2' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=1 \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint=$resume \
        --run_name='round_1' \
        --output_dir=out/forward2 \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=10000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 7000, "num_decay_steps": 2000, "min_lr_ratio": 0.01}' \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
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



for seed in 41 42 43 44 45; do
    for round in {2..10}; do
        resume=False
        do_train=True
        num_eval=1024
        max_digit=$((9 + round))
        prev_round=$((round - 1))
        steps=$((10000 + (round - 1) * 3000))
        stable_steps=$((9000 + (round - 1) * 3000))
        decay_steps=1000

        additional_train_data=""
        for i in $(seq 1 $prev_round); do
            additional_train_data+="self_improve_data_forward2/add-forward2-$((10 + i ))-$((10 + i + 1))-50000-$i-${seed},"
        done

        additional_train_data="${additional_train_data%,}" # Remove trailing comma

        CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
            --self_improve_cache_loc='self_improve_data_forward2' \
            --wandb_project='self-improve-forward' \
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
            --num_train=2000000 \
            --num_eval=$num_eval \
            --n_digits_train='1,11' \
            --op_train='add' \
            --format_train='forward2' \
            --op_dist_train='1' \
            --n_digits_eval="$((round)),$((20 + round)),1" \
            --op_eval='add' \
            --format_eval='forward2' \
            --op_dist_eval='1' \
            --show_task_ids=False \
            --padding_side='right' \
            --num_self_improve_data=50000 \
            --self_improve_round=$round \
            --additional_train_data="$additional_train_data" \
            --additional_train_data_max_digit=$max_digit \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint="out/forward2/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-forward2-digits-1_11_-seed-${seed}" \
            --reset_max_steps=True \
            --run_name="round_${round}" \
            --output_dir=out/forward2 \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=$steps \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs="{\"num_stable_steps\": $stable_steps, \"num_decay_steps\": $decay_steps, \"min_lr_ratio\": 0.01}" \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.0 \
            --logging_steps=50 \
            --eval_strategy="steps" \
            --eval_steps=250 \
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
