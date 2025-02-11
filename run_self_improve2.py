import os

import copy
from lib.data_utils import get_self_improve_dataset2
from preamble import get_args, get_run_name, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer
import transformers
import torch

torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.capture_scalar_outputs = True

transformers.logging.set_verbosity_info()

args, train_args = get_args()

if args.additional_train_data is not None and args.additional_train_data != []:
    # args.additional_train_data is a list...
    additional_train_data_list = args.additional_train_data[0].split(',')
    args.additional_train_data = additional_train_data_list
    print("self improve dataset: ", args.additional_train_data)


tokenizer = get_tokenizer(args)

train_dataset, eval_datasets, current_n_digits, max_train_digits = get_all_datasets(args, train_args, tokenizer)

model = get_model(args, train_args, tokenizer)

train_args = prepare_train_args(args, train_args, tokenizer, max_train_digits)

trainer = get_trainer(args, model, tokenizer, train_args, train_dataset, eval_datasets)

if args.only_generate_data: # skip training or eval, but only generate self-improve data
    print(f"Generating self-improve data only. \n Resuming from: {train_args.resume_from_checkpoint}")
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    _ = get_self_improve_dataset2(args, train_args, trainer, tokenizer)
    exit()


# check local rank
if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
    import wandb
    wandb.init(project=args.wandb_project, name=get_run_name(args, train_args), reinit=True, settings=wandb.Settings(_disable_stats=True))

    # Workaround for incrorrect global metrics
    # define our custom x axis metric
    wandb.define_metric("train/global_step")
    # set all other train/ metrics to use this step
    wandb.define_metric("*", step_metric="train/global_step")


if train_args.do_train:
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
    _, metrics = get_self_improve_dataset2(args, train_args, trainer, tokenizer)

wandb.finish(quiet=True)
