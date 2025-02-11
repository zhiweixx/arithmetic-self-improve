# # round 1 - sample 6,1 & 6,1
for seed in 41 42 43 44 45; do
    resume=False
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,7,1' \
        --n_digits_b_eval='1,7,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=1 \
        --additional_train_data_digit='1,6 6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint=$resume \
        --run_name='round_1' \
        --output_dir=out/mult_vanilla \
        --do_train=True \
        --do_eval=True \
        --max_steps=10000 \
        --learning_rate=5e-5 \
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
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done

# round 2 - sample 2,6 & 6,2
for seed in 41 42 43 44 45; do
    round=2
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}" \
        --additional_train_data_digit='2,6 6,2' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 3 - sample 3,6 & 6,3
for seed in 41 42 43 44 45; do
    round=3
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed},${si_path}((2-6)-(6-2))-50000-2-${seed}" \
        --additional_train_data_digit='3,6 6,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# round 4 - sample 4,6 & 6,4
for seed in 41 42 43 44 45; do
    round=4
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed},${si_path}((2-6)-(6-2))-50000-2-${seed},${si_path}((3-6)-(6-3))-50000-3-${seed}" \
        --additional_train_data_digit='4,6 6,4' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 5 - sample 5,6 & 6,5
for seed in 41 42 43 44 45; do
    round=5
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed},${si_path}((2-6)-(6-2))-50000-2-${seed},${si_path}((3-6)-(6-3))-50000-3-${seed},${si_path}((4-6)-(6-4))-50000-4-${seed}" \
        --additional_train_data_digit='5,6 6,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 6 - sample 6, 6
for seed in 41 42 43 44 45; do
    round=6
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed},${si_path}((2-6)-(6-2))-50000-2-${seed},${si_path}((3-6)-(6-3))-50000-3-${seed},${si_path}((4-6)-(6-4))-50000-4-${seed},${si_path}((5-6)-(6-5))-50000-5-${seed}" \
        --additional_train_data_digit='6,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 7 - sample 1,7 7,1
for seed in 41 42 43 44 45; do
    round=7
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=vanilla WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult' \
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
        --eval_cache_loc='data/mult' \
        --ignore_data_skip=True \
        --use_iterable_dataset=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,8,1' \
        --n_digits_b_eval='1,8,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed},${si_path}((2-6)-(6-2))-50000-2-${seed},${si_path}((3-6)-(6-3))-50000-3-${seed},${si_path}((4-6)-(6-4))-50000-4-${seed},${si_path}((5-6)-(6-5))-50000-5-${seed},${si_path}6-6-50000-6-${seed}" \
        --additional_train_data_digit='1,7 7,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_vanilla/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_vanilla \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=5e-5 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs="{\"num_stable_steps\": $num_stable_steps, \"num_decay_steps\": 1000, \"min_lr_ratio\": 0.01}" \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.0 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=250 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done

