# round 1 generate - 1,6 6,1
# generate data only - assume trained first on vanilla (or else train one yourself)

for seed in 41 42 43 44 45; do
    round=1
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
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
        --resume_from_checkpoint="out/mult_fast/round_1-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_1' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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

# round 2 - train on 1,6 6,1 / sample 2,6 3,6 6,2 6,3
for seed in 41 42 43 44 45; do
    round=2
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_1-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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


# generate majority vote data - sample 2,6 3,6 6,2 6,3
for seed in 41 42 43 44 45; do
    round=2
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=2 \
        --additional_train_data_digit='2,6 3,6 6,2 6,3' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_2-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_2' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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

# round 3 - train on 2,6 3,6 6,2 6,3 / sample 4,6 5,6 6,4 6,5
for seed in 41 42 43 44 45; do
    round=3
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate majority vote data - sample 4,6 5,6 6,4 6,5
for seed in 41 42 43 44 45; do
    round=3
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=3 \
        --additional_train_data_digit='4,6 5,6 6,4 6,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_3-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_3' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# # round 4 -  train on 4,6 5,6 6,4 6,5 / sample 6,6
for seed in 41 42 43 44 45; do
    round=4
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate majority vote data - sample 6,6
for seed in 41 42 43 44 45; do
    round=4
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        --majority_voting_threshold=0.9 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=4 \
        --additional_train_data_digit='6,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_4-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_4' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# round 5 - train on 6,6 / sample 1,7 7,1
for seed in 41 42 43 44 45; do
    round=5
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate majority vote data - sample 1,7 7,1
for seed in 41 42 43 44 45; do
    round=5
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=5 \
        --additional_train_data_digit='1,7 7,1' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_5-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_5' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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


# round 6 - train on 1,7 7,1 / sample 2,7 3,7 4,7 7,2 7,3 7,4
for seed in 41 42 43 44 45; do
    round=6
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate majority vote data - sample 2,7 3,7 4,7 7,2 7,3 7,4
for seed in 41 42 43 44 45; do
    round=6
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        --majority_voting_threshold=0.9 \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=6 \
        --additional_train_data_digit='2,7 3,7 4,7 7,2 7,3 7,4' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_6-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_6' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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


# round 7 - train on 2,7 3,7 4,7 7,2 7,3 7,4 / sample 5,7 6,7 7,5 7,6
for seed in 41 42 43 44 45; do
    round=7
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 5,7 6,7 7,5 7,6
for seed in 41 42 43 44 45; do
    round=7
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='5,7 6,7 7,5 7,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_7' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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


# round 8 - train on 5,7 6,7 7,5 7,6 / sample 7,7
for seed in 41 42 43 44 45; do
    round=8
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
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
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv" \
        --ignore_data_skip=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 7,7
for seed in 41 42 43 44 45; do
    round=8
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
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
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_8' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 9 - train 7,7 / sample 1,8 2,8 8,1 8,2
for seed in 41 42 43 44 45; do
    round=9
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,10,1' \
        --n_digits_b_eval='1,10,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 1,8 2,8 8,1 8,2
for seed in 41 42 43 44 45; do
    round=9
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='1,8 2,8 8,1 8,2' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_9' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 10 - train on 1,8 2,8 8,1 8,2 / sample 3,8 4,8 5,8 8,3 8,4 8,5
for seed in 41 42 43 44 45; do
    round=10
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,10,1' \
        --n_digits_b_eval='1,10,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done

# generate data 3,8 4,8 5,8 8,3 8,4 8,5
for seed in 41 42 43 44 45; do
    round=10
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='3,8 4,8 5,8 8,3 8,4 8,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_10' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 11 - train on 3,8 4,8 5,8 8,3 8,4 8,5 / sample 6,8 7,8 8,6 8,6 8,7
for seed in 41 42 43 44 45; do
    round=11
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,10,1' \
        --n_digits_b_eval='1,10,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done

# generate data 6,8 7,8 8,6 8,7
for seed in 41 42 43 44 45; do
    round=11
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,8 7,8 8,6 8,7' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_11' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 12 - train on 6,8 7,8 8,6 8,7 / sample 8,8
for seed in 41 42 43 44 45; do
    round=12
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,10,1' \
        --n_digits_b_eval='1,10,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 8,8
for seed in 41 42 43 44 45; do
    round=12
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='8,8' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_12' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 13 - train on 8,8 / sample 1,9 2,9 3,9 4,9 5,9 9,1 9,2 9,3 9,4 9,5
for seed in 41 42 43 44 45; do
    round=13
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,10,1' \
        --n_digits_b_eval='1,10,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 1,9 2,9 3,9 4,9 5,9 9,1 9,2 9,3 9,4 9,5
for seed in 41 42 43 44 45; do
    round=13
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='1,9 2,9 3,9 4,9 5,9 9,1 9,2 9,3 9,4 9,5' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_13' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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

# Round 14 - train on 1,9 2,9 3,9 4,9 5,9 9,1 9,2 9,3 9,4 9,5 sample 6,9 7,9 8,9 9,6 9,7 9,8
for seed in 41 42 43 44 45; do
    round=14
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,11,1' \
        --n_digits_b_eval='1,11,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done

# generate data 6,9 7,9 8,9 9,6 9,7 9,8
for seed in 41 42 43 44 45; do
    round=14
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='6,9 7,9 8,9 9,6 9,7 9,8' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_14' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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



# Round 15 - train on 6,9 7,9 8,9 9,6 9,7 9,8/ sample 9,9
for seed in 41 42 43 44 45; do
    round=15
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,11,1' \
        --n_digits_b_eval='1,11,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv,${si_path}((6-9)-(7-9)-(8-9)-(9-6)-(9-7)-(9-8))-50000-14-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done



# generate data 9,9
for seed in 41 42 43 44 45; do
    round=15
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='9,9' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_14' \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
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


# Round 16 - train on 9,9 / sample 1,10 2,10 3,10 4,10 5,10 6,10 10,1 10,2 10,3 10,4 10,5 10,6
for seed in 41 42 43 44 45; do
    round=16
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,11,1' \
        --n_digits_b_eval='1,11,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv,${si_path}((6-9)-(7-9)-(8-9)-(9-6)-(9-7)-(9-8))-50000-14-${seed}-mv,${si_path}9-9-50000-15-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done



# generate data 1,10 2,10 3,10 4,10 5,10 6,10 10,1 10,2 10,3 10,4 10,5 10,6
for seed in 41 42 43 44 45; do
    round=16
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='1,10 2,10 3,10 4,10 5,10 6,10 10,1 10,2 10,3 10,4 10,5 10,6' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_14' \
        --output_dir=out/mult_fast \
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
        --bf16=True
done



# # Round 17 - train on 1,10 2,10 3,10 4,10 5,10 6,10 10,1 10,2 10,3 10,4 10,5 10,6 / sample 7,10 8,10 9,10 10,7 10,8 10,9
for seed in 41 42 43 44 45; do
    round=17
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000
    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='6,12,1' \
        --n_digits_b_eval='6,12,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv,${si_path}((6-9)-(7-9)-(8-9)-(9-6)-(9-7)-(9-8))-50000-14-${seed}-mv,${si_path}9-9-50000-15-${seed}-mv,${si_path}((1-10)-(2-10)-(3-10)-(4-10)-(5-10)-(6-10)-(10-1)-(10-2)-(10-3)-(10-4)-(10-5)-(10-6))-50000-16-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 7,10 8,10 9,10 10,7 10,8 10,9
for seed in 41 42 43 44 45; do
    round=17
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='7,10 8,10 9,10 10,7 10,8 10,9' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_14' \
        --output_dir=out/mult_fast \
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
        --bf16=True
done



# Round 18 - train on 7,10 8,10 9,10 10,7 10,8 10,9 / sample 10,10
for seed in 41 42 43 44 45; do
    round=18
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000

    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='6,12,1' \
        --n_digits_b_eval='6,12,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv,${si_path}((6-9)-(7-9)-(8-9)-(9-6)-(9-7)-(9-8))-50000-14-${seed}-mv,${si_path}9-9-50000-15-${seed}-mv,${si_path}((1-10)-(2-10)-(3-10)-(4-10)-(5-10)-(6-10)-(10-1)-(10-2)-(10-3)-(10-4)-(10-5)-(10-6))-50000-16-${seed}-mv,${si_path}((7-10)-(8-10)-(9-10)-(10-7)-(10-8)-(10-9))-50000-17-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# generate data 10,10
for seed in 41 42 43 44 45; do
    round=18
    resume=True
    do_train=True
    num_eval=128 
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE='disabled' python run_self_improve2.py \
        --only_generate_data=True \
        --majority_voting=True \
        --majority_vote_seeds='41,42,43,44,45' \
        --majority_voting_threshold=0.9 \
        --majority_vote_keep_seed=True \
        --filter_shorter_additional_data=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=50000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='1,2,1' \
        --n_digits_b_eval='1,2,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data_digit='10,10' \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name='round_14' \
        --output_dir=out/mult_fast \
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
        --bf16=True
done


# Round 19 - train on 10,10
for seed in 41 42 43 44 45; do
    round=19
    resume=False
    do_train=True
    num_eval=128
    let prev_round=round-1
    let max_steps=10000+prev_round*3000
    let num_stable_steps=max_steps-1000

    si_path="self_improve_data_mult_fast/mult-COT-"
    CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run_self_improve2.py \
        --majority_voting=True \
        \
        \
        --wandb_project='self-improve-multiplication' \
        --self_improve_cache_loc='self_improve_data_mult_fast' \
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
        --use_iterable_dataset=False \
        --num_train=5000000 \
        --num_eval=$num_eval \
        --n_digits_a_train='1,6' \
        --n_digits_b_train='1,6' \
        --op_train='mult' \
        --format_train='COT' \
        --op_dist_train='1' \
        --n_digits_a_eval='6,12,1' \
        --n_digits_b_eval='6,12,1' \
        --op_eval='mult' \
        --format_eval='COT' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --num_self_improve_data=50000 \
        --self_improve_round=$round \
        --additional_train_data="${si_path}((1-6)-(6-1))-50000-1-${seed}-mv,${si_path}((2-6)-(3-6)-(6-2)-(6-3))-50000-2-${seed}-mv,${si_path}((4-6)-(5-6)-(6-4)-(6-5))-50000-3-${seed}-mv,${si_path}6-6-50000-4-${seed}-mv,${si_path}((1-7)-(7-1))-50000-5-${seed}-mv,${si_path}((2-7)-(3-7)-(4-7)-(7-2)-(7-3)-(7-4))-50000-6-${seed}-mv,${si_path}((5-7)-(6-7)-(7-5)-(7-6))-50000-7-${seed}-mv,${si_path}7-7-50000-8-${seed}-mv,${si_path}((1-8)-(2-8)-(8-1)-(8-2))-50000-9-${seed}-mv,${si_path}((3-8)-(4-8)-(5-8)-(8-3)-(8-4)-(8-5))-50000-10-${seed}-mv,${si_path}((6-8)-(7-8)-(8-6)-(8-7))-50000-11-${seed}-mv,${si_path}8-8-50000-12-${seed}-mv,${si_path}((1-9)-(2-9)-(3-9)-(4-9)-(5-9)-(9-1)-(9-2)-(9-3)-(9-4)-(9-5))-50000-13-${seed}-mv,${si_path}((6-9)-(7-9)-(8-9)-(9-6)-(9-7)-(9-8))-50000-14-${seed}-mv,${si_path}9-9-50000-15-${seed}-mv,${si_path}((1-10)-(2-10)-(3-10)-(4-10)-(5-10)-(6-10)-(10-1)-(10-2)-(10-3)-(10-4)-(10-5)-(10-6))-50000-16-${seed}-mv,${si_path}((7-10)-(8-10)-(9-10)-(10-7)-(10-8)-(10-9))-50000-17-${seed}-mv,${si_path}10-10-50000-18-${seed}-mv" \
        --ignore_data_skip=True \
        --auto_find_batch_size=True \
        \
        \
        --save_total_limit=1 \
        --resume_from_checkpoint="out/mult_fast/round_${prev_round}-llama-384-6-6-1024-SI_round_${prev_round}-COT-digits-1_6_-1_6_-seed-${seed}" \
        --run_name="round_${round}" \
        --output_dir=out/mult_fast \
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
        --eval_steps=500 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=$resume \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=256 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done


# EVAL
for round in 19; do
    for seed in 41 42 43 44 45; do
        resume=False
        do_train=False
        num_eval=512
        CUDA_VISIBLE_DEVICES=0 WANDB_RUN_GROUP=mult_fast WANDB_MODE=online python run.py \
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
            --use_iterable_dataset=False \
            --num_train=50000 \
            --num_eval=$num_eval \
            --n_digits_a_train='1,6' \
            --n_digits_b_train='1,6' \
            --op_train='mult' \
            --format_train='COT' \
            --op_dist_train='1' \
            --n_digits_a_eval='1,13,1' \
            --n_digits_b_eval='1,13,1' \
            --op_eval='mult' \
            --format_eval='COT' \
            --op_dist_eval='1' \
            --show_task_ids=False \
            --padding_side='right' \
            --num_self_improve_data=50000 \
            --self_improve_round=$round \
                \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint="out/mult_fast/round_${round}-llama-384-6-6-1024-SI_round_${round}-COT-digits-1_6_-1_6_-seed-${seed}" \
            --run_name="round_${round}" \
            --output_dir=out/mult_acc_mv_len \
            --do_train=False \
            --do_eval=True \
            --max_steps=10000 \
            --learning_rate=5e-5 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 7000, "num_decay_steps": 2000, "min_lr_ratio": 0.01}' \
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
            --per_device_eval_batch_size=512 \
            --gradient_accumulation_steps=1 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
    done
done
