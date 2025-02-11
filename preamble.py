import json
import os
import re
import string
from functools import partial

import torch
from lib.configs import ScriptArguments, MyTrainingArguments
from lib.data_utils import get_additonal_train_data, get_task_length, get_train_dataset, get_eval_dataset, PromptAnswerDataCollator
from lib.eval_utils import compute_metrics, process_generate_output
from lib.modeling.llama import LlamaForCausalLMWithNoPE, MyLlamaConfig
from lib.trainer_utils import AddWandbConfigCallback, DataMixtureSchedulingCallback, EarlyStoppingCallback, Seq2SeqTrainerNoEvalLoss
from charactertokenizer import CharacterTokenizer

from typing import Tuple, cast
from transformers import HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig, PreTrainedModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# torch._dynamo.config.optimize_ddp=False

def get_args():
    args, train_args = HfArgumentParser((ScriptArguments, MyTrainingArguments)).parse_args_into_dataclasses()
    args = cast(ScriptArguments, args)
    train_args = cast(MyTrainingArguments, train_args)
    args.block_size = args.max_position_embeddings
    if args.rope_theta == 'Inf':
        args.rope_theta = torch.inf

    set_seed(train_args.seed)
    return args, train_args

def get_tokenizer(args: ScriptArguments):
    # We don't pad when generating the datasets
    # During eval, the inputs are padded on the left and the labels are padding on the right using custom data collator

    if args.op_train[0] == 'maze' or args.op_eval[0] == 'maze':
        print("="*30 + "Using Tokenizer for maze (Tokenized 1~99 separately)\n" + "="*30)
        from charactertokenizer import CharacterNumberTokenizer
        all_chars = string.digits + string.punctuation + ' ' + '\n'
        tokenizer = CharacterNumberTokenizer(all_chars, args.max_position_embeddings)
        tokenizer.padding_side == 'left'

        return tokenizer
        
    if args.use_character_tokenizer:
        if args.char_list is not None:
            all_chars = args.char_list
        else:
            all_chars = string.ascii_letters + string.digits + string.punctuation + ' '
        tokenizer = CharacterTokenizer(all_chars, args.max_position_embeddings)
        tokenizer.padding_side == 'left'
        tokenizer.backtrack_token_id = tokenizer.convert_tokens_to_ids(['X'])[0]

        if args.model_id is not None:
            old_tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN", None))
            overlap_special_keys = [k for k in tokenizer.special_tokens_map_extended.keys() if k in old_tokenizer.special_tokens_map_extended]
            all_vocab = [k for k,v in tokenizer.get_vocab().items() if v >= 0 and k in old_tokenizer.get_vocab()] # filter out the -100 token, which will never appear in the input
            # old_ids = old_tokenizer.convert_tokens_to_ids(all_vocab + [old_tokenizer.special_tokens_map[k] for k in overlap_special_keys])
            # new_ids = tokenizer.convert_tokens_to_ids(all_vocab + [tokenizer.special_tokens_map[k] for k in overlap_special_keys])
            # emb = model.get_input_embeddings()
            # assert isinstance(emb, torch.nn.Embedding), "Only nn.Embedding is supported for now"
            # new_emb = torch.nn.Embedding(len(tokenizer), emb.embedding_dim, dtype=emb.weight.dtype)
            # new_emb.weight.data[torch.tensor(new_ids)] = emb(torch.tensor(old_ids))
            # model.set_input_embeddings(emb)
            # model.resize_token_embeddings(len(tokenizer))
            for voc in all_vocab:
                tokenizer._vocab_str_to_int[voc] = old_tokenizer.convert_tokens_to_ids(voc)
            tokenizer._vocab_int_to_str = {v: k for k, v in tokenizer._vocab_str_to_int.items()}
            tokenizer.add_special_tokens({k: getattr(old_tokenizer, k) for k in overlap_special_keys})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN", None))
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_all_datasets(args: ScriptArguments, train_args: MyTrainingArguments, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset, int]:
    train_dataset, eval_datasets = None, None
    n_digits_a_end, n_digits_b_end = args.n_digits_a_train[0][1], args.n_digits_b_train[0][1]
    additional_train_data = get_additonal_train_data(args, train_args, tokenizer)
    if len(additional_train_data) > 0 and not (args.op_train[0] in ['mult', 'maze']):
        recent_ds = additional_train_data[-1]
        if recent_ds.info.description is not None and len(recent_ds.info.description) > 0:
            (n_digits_a_start, n_digits_a_end), (n_digits_b_start, n_digits_b_end) = json.loads(recent_ds.info.description)
        elif recent_ds.split is not None and '.' in str(recent_ds.split):
            # legacy format
            n_digits_a_start, n_digits_a_end = list(map(int, str(recent_ds.split).split('.')[1:]))
            n_digits_b_start, n_digits_b_end = list(map(int, str(recent_ds.split).split('.')[1:]))
        else:
            (n_digits_a_start, n_digits_a_end) = args.n_digits_a_train[0][0], args.n_digits_a_train[0][1] + args.self_improve_round
            (n_digits_b_start, n_digits_b_end) = args.n_digits_b_train[0][0], args.n_digits_b_train[0][1] + args.self_improve_round
    else:
        (n_digits_a_start, n_digits_a_end), (n_digits_b_start, n_digits_b_end) = (args.n_digits_a_train[0], args.n_digits_b_train[0])

    if args.dynamic_eval_range:
        args.n_digits_eval = args.n_digits_a_eval = args.n_digits_b_eval = (n_digits_a_end + args.dynamic_eval_range[0], n_digits_a_end + args.dynamic_eval_range[1], args.dynamic_eval_range[2])
    elif args.dynamic_eval_range_a and args.dynamic_eval_range_b:
        args.n_digits_a_eval = (n_digits_a_end - 1 + args.dynamic_eval_range_a[0], n_digits_a_end - 1 + args.dynamic_eval_range_a[1], args.dynamic_eval_range_a[2])
        args.n_digits_b_eval = (n_digits_b_end - 1 + args.dynamic_eval_range_b[0], n_digits_b_end - 1 + args.dynamic_eval_range_b[1], args.dynamic_eval_range_b[2])

    if train_args.do_eval:
        eval_datasets, unmapped_eval_datasets = get_eval_dataset(args, train_args, tokenizer)
    if train_args.do_train:
        train_dataset, max_train_digits = get_train_dataset(args, train_args, tokenizer, no_sample_from=unmapped_eval_datasets, additional_train_data=additional_train_data)
    else:
        max_train_digits = 0
    tokenizer.padding_side = 'left' # in case it was changed by the data generator
    return train_dataset, eval_datasets, (n_digits_a_end, n_digits_b_end), max_train_digits

def get_non_hf_model_id(args: ScriptArguments):
    return f"{args.architecture}-{args.hidden_size}-{args.num_attention_heads}-{args.num_layers}-{args.max_position_embeddings}"

def get_model(args: ScriptArguments, train_args: MyTrainingArguments, tokenizer: PreTrainedTokenizer):
    if args.model_id is not None:
        model: PreTrainedModel
        if args.from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN", None))
        else:
            model_config = AutoConfig.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN", None))
            model = AutoModelForCausalLM.from_config(model_config)

        if args.freeze_except is not None:
            for p in model.parameters():
                p.requires_grad = False
        for name, module in model.named_modules():
            if args.freeze is not None:
                if re.search(args.freeze, name) is not None:
                    for p in module.parameters():
                        p.requires_grad = False
                    print(f"Freezing {name}")
            elif args.freeze_except is not None:
                if re.search(args.freeze_except, name) is not None:
                    for p in module.parameters():
                        p.requires_grad = True
                    print(f"Freezing except {name}")

        args.architecture = model.config.architectures[0]
        model.generation_config = None
    else:
        if args.architecture.startswith("llama"):
            model_config = MyLlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_attention_heads=args.num_attention_heads,
                num_hidden_layers=args.num_layers,
                max_position_embeddings=args.max_position_embeddings,
                # _attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa',
                _attn_implementation='sdpa',
                # rope_theta=torch.inf
                rope_theta=args.rope_theta,
                partial_rotary_factor=args.partial_rotary_factor,
                use_lpe=args.architecture == 'llama-lpe',
                attention_dropout=args.dropout
            )
            model = LlamaForCausalLMWithNoPE(model_config)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

    if args.use_lora:
        default_lora_config = {
            "task_type": "CAUSAL_LM",
            "r": 32,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        extra_lora_config = args.lora_config if args.lora_config is not None else {}
        default_lora_config.update(extra_lora_config)
        lora_config =  LoraConfig(
            **default_lora_config
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print(f"Number of parameters: {model.num_parameters()}")
    print(f"Number of trainable parameters: {model.num_parameters(only_trainable=True)}")

    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    return model

def get_run_name(args: ScriptArguments, train_args: MyTrainingArguments):
    run_name = train_args.run_name_prefix

    if args.model_id is None:
        run_name += f"-{get_non_hf_model_id(args)}"
    else:
        run_name += f"-{args.model_id}"
    if args.from_pretrained:
        run_name += "-pretrained"
    if not args.use_character_tokenizer:
        run_name += "-nochar"
    if args.use_lora:
        run_name += "-lora"
    if args.freeze:
        run_name += "-frz-" + args.freeze
    if args.freeze_except:
        run_name += "-frzex-" + args.freeze_except
    if args.rope_theta != torch.inf:
        run_name += f"-rope"
    if len(set(args.num_train)) != 1:
        run_name += f"-train-{args.num_train}"
    if args.self_improve_round is not None:
        if args.fast_self_improve:
            run_name += f"-fast_SI"
        else:
            run_name += f"-SI"
        run_name += f"_round_{args.self_improve_round}"
    if args.filter_self_improve:
        run_name += f"-filter_{args.filter_self_improve}"
    if args.add_noise and args.noise_type:
        run_name += f"-{args.noise_type}_{args.add_noise}_at_{args.add_noise_at}"
    disp_task = [args.format_train[i] if args.format_train[i] != 'None' else args.op_train[i] for i in range(len(args.format_train))]
    run_name += f'-{disp_task}'
    if args.n_digits_train is not None:
        run_name += f'-digits-{args.n_digits_train}'
    else:
        run_name += f'-digits-{args.n_digits_a_train}-{args.n_digits_b_train}'
    run_name += f'-seed-{train_args.seed}'
    # run_name += f'-{train_args.seed}'
    translator = str.maketrans('/,', '__', ''.join(set(string.punctuation + string.whitespace) - set('/,_-.')))
    run_name = str.translate(run_name, translator)

    if not train_args.do_train:
        train_args.run_name += '-eval'

    return run_name

def prepare_train_args(args: ScriptArguments, train_args: MyTrainingArguments, tokenizer: PreTrainedTokenizer, max_train_digits: int):
    args.task_length = max_train_digits
    
    max_length = get_task_length(args, max_train_digits)
    if max_length > 140:
        train_args.per_device_train_batch_size //= 2
        train_args.gradient_accumulation_steps *= 2
    # if max_length > 180:
    #     train_args.per_device_train_batch_size //= 2
    #     train_args.gradient_accumulation_steps *= 2
    if max_length > 200:
        train_args.per_device_train_batch_size //= 2
        train_args.per_device_eval_batch_size //= 2
        train_args.gradient_accumulation_steps *= 2

    # calculate train_pad_to because dynamic padding does not work with DDP
    if os.environ.get("WORLD_SIZE") is not None:
        if args.train_pad_to is None:
            args.train_pad_to = max_length
            print(f"train_pad_to: {args.train_pad_to}")
    
    train_args.generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        max_length=0, # because we set it in the trainer prediction_step
        # forced_eos_token_id=tokenizer.eos_token_id
        return_dict_in_generate=args.entropy_metrics,
        output_logits=args.entropy_metrics
    )

    train_args.run_name = get_run_name(args, train_args)

    train_args.output_dir = f"{train_args.output_dir}/{train_args.run_name}"
    train_args.save_safetensors = False # supposed to fix "There were missing keys in the checkpoint model loaded: ['lm_head.weight']."
    train_args.dataloader_num_workers = args.num_workers
    # train_args.dataloader_persistent_workers = args.num_workers > 0
    train_args.dataloader_persistent_workers = False
    # train_args.dataloader_prefetch_factor = 4 if args.num_workers > 0 else None # Seemed to be causing segfault in the workers
    train_args.remove_unused_columns = False
    # train_args.ignore_data_skip = True # It takes a long time to skip the data
    train_args.eval_do_concat_batches = False

    if train_args.resume_from_checkpoint == 'True':
        # Try finding a checkpoint in the output directory
        train_args.resume_from_checkpoint = get_last_checkpoint(train_args.output_dir)
        if train_args.resume_from_checkpoint is None:
            raise ValueError(f"No checkpoint found in {train_args.output_dir}")
    elif train_args.resume_from_checkpoint == 'False':
        train_args.resume_from_checkpoint = None
    else:
        # Try finding a checkpoint in the provided path
        try:
            ckpt_dir = get_last_checkpoint(train_args.resume_from_checkpoint)
            if ckpt_dir is not None:
                train_args.resume_from_checkpoint = ckpt_dir
        except:
            pass

    return train_args

def get_trainer(args: ScriptArguments, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_args: MyTrainingArguments, train_dataset: Dataset, eval_datasets: Dataset):
    trainer = Seq2SeqTrainerNoEvalLoss(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_datasets if train_args.do_eval else None,
        compute_metrics=partial(compute_metrics, tokenizer, args=train_args),
        preprocess_logits_for_metrics=process_generate_output if args.entropy_metrics else None,
        data_collator=PromptAnswerDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=-100,
            train_pad_side=args.padding_side,
            train_pad_to=args.train_pad_to,
            eval_pad_to=args.eval_pad_to
        )
    )

    if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
        AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, args.__dict__, args.__dict__])
        trainer.add_callback(AddConfigCB)

    if train_args.early_stop:
        EarlyStoppingCB = EarlyStoppingCallback(metric_names=train_args.early_stop_metrics, thresholds=train_args.early_stop_thresholds, patience=1)
        trainer.add_callback(EarlyStoppingCB)
    
    if train_args.use_custom_scheduler is not None:
        from lib.trainer_utils import CustomLRCallback
        trainer.add_callback(CustomLRCallback(train_args))

    
    if len(args.op_dist_train) > 1:
        MixtureCB = DataMixtureSchedulingCallback(init=args.op_dist_train[0], end=args.op_dist_train[1], **args.mixture_scheduling_kwargs)
        trainer.callback_handler.callbacks.insert(0, MixtureCB)

    # if train_args.ignore_data_skip:
    #     # instead of skipping the data, we will just restore the random state
    #     SaveRandomStateCB = SaveRandomStateCallback()
    #     trainer.add_callback(SaveRandomStateCB)

    return trainer
