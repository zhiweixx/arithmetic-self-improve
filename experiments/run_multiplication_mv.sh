# round 1 - sample 6,1 & 6,1
# generate data only - assume trained first on vanilla (or else train one yourself)
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --ignore_data_skip=True \
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
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --auto_find_batch_size=True \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 2 - sample 2,6 & 6,2 later
for seed in 41 42 43 44 45; do
    round=2
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=2
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# generate majority vote data
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=2 \
        --additional_train_data_digit='2,6 6,2' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_2-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_2' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done

# round 3 - going to sample 3,6 & 6,3
for seed in 41 42 43 44 45; do
    round=3
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=3
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate majority vote data
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=3 \
        --additional_train_data_digit='3,6 6,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_3-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# round 4 - majority_vote, going to sample 4,6 & 6,4
for seed in 41 42 43 44 45; do
    round=4
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=4
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate majority vote data
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=4 \
        --additional_train_data_digit='4,6 6,4' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_4-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# round 5 - going to sample 5,6 & 6,5
for seed in 41 42 43 44 45; do
    round=5
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done

# Do EVAL
for seed in 41 42 43 44 45; do
    round=5
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate majority vote data
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=5 \
        --additional_train_data_digit='5,6 6,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_5-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# round 6 - going to sampled 6,6
for seed in 41 42 43 44 45; do
    round=6
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done

# Do EVAL
for seed in 41 42 43 44 45; do
    round=6
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate majority vote data
for seed in 41 42 43 44 45; do
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=6 \
        --additional_train_data_digit='6,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_6-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done




# round 7 - majority_vote, trained on additional 6,6
for seed in 41 42 43 44 45; do
    round=7
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv" \
        --additional_train_data_digit='1,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=7
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 1,7 7,1
for seed in 41 42 43 44 45; do
    round=7
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='1,7 7,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# round 8 - training on additional 1,7 7,1
for seed in 41 42 43 44 45; do
    round=8
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv" \
        --additional_train_data_digit='2,7 7,2' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=8
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 2,7 7,2
for seed in 41 42 43 44 45; do
    round=8
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='2,7 7,2' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


Round 9 - training on additional 2,7 7,2
for seed in 41 42 43 44 45; do
    round=9
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=9
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 3,7 7,3
for seed in 41 42 43 44 45; do
    round=9
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done




# Round 10 - training on additional 3,7 7,3
for seed in 41 42 43 44 45; do
    round=10
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv,${si_path}((3-7)-(7-3))-50000-9-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=10
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 4,7 7,4
for seed in 41 42 43 44 45; do
    round=10
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='4,7 7,4' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done



# Round 11 - training on additional 4,7 7,4
for seed in 41 42 43 44 45; do
    round=11
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv,${si_path}((3-7)-(7-3))-50000-9-${seed}-mv,${si_path}((4-7)-(7-4))-50000-10-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=11
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 5,7 7,5
for seed in 41 42 43 44 45; do
    round=11
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='5,7 7,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done




# Round 12 - training on additional 5,7 7,5
for seed in 41 42 43 44 45; do
    round=12
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv,${si_path}((3-7)-(7-3))-50000-9-${seed}-mv,${si_path}((4-7)-(7-4))-50000-10-${seed}-mv,${si_path}((5-7)-(7-5))-50000-11-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=12
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done

# generate data 6,7 7,6
for seed in 41 42 43 44 45; do
    round=12
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='6,7 7,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done




# Round 13 - training on additional 6,7 7,6
for seed in 41 42 43 44 45; do
    round=13
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv,${si_path}((3-7)-(7-3))-50000-9-${seed}-mv,${si_path}((4-7)-(7-4))-50000-10-${seed}-mv,${si_path}((5-7)-(7-5))-50000-11-${seed}-mv,${si_path}((6-7)-(7-6))-50000-12-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=13
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 7,7
for seed in 41 42 43 44 45; do
    round=13
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='7,7' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# Round 14 - training on additional 7,7
for seed in 41 42 43 44 45; do
    round=14
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_mv/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(6-2))-50000-2-${seed}-mv,${si_path}((3-6)-(6-3))-50000-3-${seed}-mv,${si_path}((4-6)-(6-4))-50000-4-${seed}-mv,${si_path}((5-6)-(6-5))-50000-5-${seed}-mv,${si_path}6-6-50000-6-${seed}-mv,${si_path}((1-7)-(7-1))-50000-7-${seed}-mv,${si_path}((2-7)-(7-2))-50000-8-${seed}-mv,${si_path}((3-7)-(7-3))-50000-9-${seed}-mv,${si_path}((4-7)-(7-4))-50000-10-${seed}-mv,${si_path}((5-7)-(7-5))-50000-11-${seed}-mv,${si_path}((6-7)-(7-6))-50000-12-${seed}-mv,${si_path}7-7-50000-13-${seed}-mv" \
        --additional_train_data_digit='3,7 7,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# Do EVAL
for seed in 41 42 43 44 45; do
    round=14
    resume=False
    do_train=False
    num_eval=512
    let prev_round=round-1
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE=online python run_self_improve2.py \
        --wandb_project='self-improve-multiplication' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,9,1' \
        --n_digits_b_eval='1,9,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_mv \
        --do_train=False \
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
        --per_device_eval_batch_size=512 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done


# generate data 1,8 8,1
for seed in 41 42 43 44 45; do
    round=14
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=majority_vote WANDB_MODE='disabled' python run_self_improve.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.7 \
        --majority_vote_keep_seed=True \
        --majority_voting_threshold=0.7 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_mv' \
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
        --use_iterable_dataset=True \
        --ignore_data_skip=True \
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
        --self_improve_round=$round \
        --additional_train_data_digit='1,8 8,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_mv/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_mv \
        --do_train=$do_train \
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
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True
done
