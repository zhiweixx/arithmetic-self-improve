import os

import copy
from lib.data_utils import get_self_improve_dataset, get_self_improve_dataset_path
from preamble import get_args, get_run_name, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer
import transformers
import torch

torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.capture_scalar_outputs = True

transformers.logging.set_verbosity_info()

args, train_args = get_args()
si_round = args.self_improve_round
if si_round > 0:
    additional_train_data = []
    for r in range(1, si_round + 1):
        # Prepare the self-improve datasets
        args.self_improve_round = r - 1
        train_args.run_name = get_run_name(args, train_args)
        additional_train_data.append(get_self_improve_dataset_path(args, train_args))
    args.additional_train_data = additional_train_data

    # Resume from the last checkpoint
    args.self_improve_round = si_round - 1
    if train_args.resume_from_checkpoint == 'False':
        train_args.resume_from_checkpoint = f"out/{get_run_name(args, train_args)}"
    args.self_improve_round = si_round
    train_args.max_steps = train_args.max_steps + (train_args.self_improve_stable_steps + train_args.self_improve_decay_steps) * args.self_improve_round
    train_args.lr_scheduler_kwargs = {
        "num_stable_steps": train_args.max_steps - train_args.self_improve_decay_steps,
        "num_decay_steps": train_args.self_improve_decay_steps,
        "min_lr_ratio": 0.01
    }
    train_args.warmup_ratio = 0.0
    train_args.reset_max_steps = True

# check local rank
if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
    import wandb
    wandb.init(project='self-improve', name=get_run_name(args, train_args), reinit=True, group=f'round_{si_round}', settings=wandb.Settings(_disable_stats=True))

    # Workaround for incrorrect global metrics
    # define our custom x axis metric
    wandb.define_metric("train/global_step")
    # set all other train/ metrics to use this step
    wandb.define_metric("*", step_metric="train/global_step")

tokenizer = get_tokenizer(args)

train_dataset, eval_datasets, current_n_digits, max_train_digits = get_all_datasets(args, train_args, tokenizer)

model = get_model(args, train_args, tokenizer)

train_args = prepare_train_args(args, train_args, tokenizer, max_train_digits)

trainer = get_trainer(args, model, tokenizer, train_args, train_dataset, eval_datasets)

if args.only_generate_data: # skip training or eval, but only generate self-improve data
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    _ = get_self_improve_dataset(train_args, args, trainer, tokenizer)
    exit()

if train_args.do_train:
    assert si_round == 0 or train_args.resume_from_checkpoint is not None
    if train_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        # import cProfile
        # with cProfile.Profile() as pr:
        #     trainer.train()
        # pr.dump_stats(f"{train_args.output_dir}/profile")
        trainer.train()
elif train_args.do_eval:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()

if args.skip_self_improve_generation:
    wandb.finish(quiet=True)
    exit()

if train_args.do_train and not args.majority_voting:
    _, metrics = get_self_improve_dataset(args, train_args, trainer, tokenizer, current_n_digits)

wandb.finish(quiet=True)
