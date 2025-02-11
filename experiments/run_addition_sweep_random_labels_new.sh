#!/usr/bin/env zsh
set -e

MAX_RETRIES=3
RETRY_DELAY=1  # seconds

retry_command() {
    local attempt=1
    local result=0

    while (( attempt <= MAX_RETRIES )); do
        echo "Attempt $attempt of $MAX_RETRIES..."
        
        if $@; then
            echo "Command succeeded"
            return 0
        else
            result=$?
            echo "Attempt $attempt failed with exit code $result"
            
            if (( attempt < MAX_RETRIES )); then
                echo "Waiting ${RETRY_DELAY} seconds before retry..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    echo "All retry attempts failed"
    return $result
}

# 0
for start_round in 5 10 20; do
# 0.5 0.6 0.7 0.8 0.9 1.0
# 0.1 0.2
# 0.05 0.15
for add_noise in 0.05 0.5 0.1 0.2 0.4 0.3 0.6 0.7 0.8 0.9; do
# drop-preturb uniform drop-digits preturb
for noise_type in uniform drop-digits preturb drop-preturb; do

for r in {0..5}; do
for trial in {0..2}; do

wd=$(pwd)
i=$((start_round + r))

if [[ -e "${wd}/self_improve_data/trial-${trial}-llama-384-6-6-1024-SI_round_${i}-${noise_type}_${add_noise}_at_${start_round}-reverse-digits-1_17_-seed-42-50000/dataset_info.json" ]]; then
    echo "skipped" 
    continue
fi

echo "Running trial $trial with noise_type $noise_type, add_noise $add_noise, start_round $start_round, round $r"
if [[ $r -gt 0 ]]; then
    resume_from_checkpoint=False
else
    resume_from_checkpoint=True
    for i in {0..$((start_round + r))}; do
    #     rm -rf ${wd}/self_improve_data/-llama-384-6-6-1024-SI_round_${i}-${noise_type}_${add_noise}-reverse-digits-1_17_-seed-42-50000
        ln -sf ${wd}/self_improve_data/-llama-384-6-6-1024-SI_round_${i}-reverse-digits-1_17_-seed-42-50000 ${wd}/self_improve_data/trial-${trial}-llama-384-6-6-1024-SI_round_${i}-${noise_type}_${add_noise}_at_${start_round}-reverse-digits-1_17_-seed-42-50000
    done
    i=$start_round
    # rm -rf ${wd}/out/-llama-384-6-6-1024-SI_round_${i}-${noise_type}_${add_noise}-reverse-digits-1_17_-seed-42
    ln -sf ${wd}/out/-llama-384-6-6-1024-SI_round_${i}-reverse-digits-1_17_-seed-42 ${wd}/out/trial-${trial}-llama-384-6-6-1024-SI_round_${i}-${noise_type}_${add_noise}_at_${start_round}-reverse-digits-1_17_-seed-42
fi

# CUDA_VISIBLE_DEVICES=0,1  WANDB_MODE=online torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --master_port=42068 run_self_improve.py \
# accelerate launch run_self_improve.py \
retry_command env CUDA_VISIBLE_DEVICES=1  WANDB_MODE=online python run_self_improve.py \
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
    --num_train=10000000 \
    --num_eval=1024 \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --op_dist_train='1' \
    --dynamic_eval_range -3 5 1 \
    --op_eval='add' \
    --format_eval='reverse' \
    --op_dist_eval='1' \
    --show_task_ids=False \
    --padding_side='right' \
    --num_self_improve_data=50000 \
    --self_improve_round=$((start_round + r)) \
    --add_noise=$add_noise \
    --noise_type=$noise_type \
    --add_noise_at=$start_round \
    \
    \
    --resume_from_checkpoint=$resume_from_checkpoint \
    --save_total_limit=1 \
    --run_name='trial-'$trial \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --self_improve_stable_steps=500 \
    --self_improve_decay_steps=500 \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='warmup_stable_decay' \
    --lr_scheduler_kwargs='{"num_stable_steps": 7000, "num_decay_steps": 2000, "min_lr_ratio": 0.01}' \
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
    --torch_compile=True \
    --bf16=True \
    --tf32=True
done
done
done
done
done
