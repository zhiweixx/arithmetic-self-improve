from dataclasses import dataclass
from functools import partial
import itertools
import json
import os
from typing import List, Tuple, Dict, Union

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence

from .configs import MyTrainingArguments, ScriptArguments
from .data_formats import get_copy, get_reverse, get_reverse_add, get_forward, get_forward2, get_COT_mult
from .data_formats_maze import generate_maze_with_wilson

import random
import numpy as np
from datasets import IterableDataset, Dataset, interleave_datasets, concatenate_datasets
from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments
from trl.trainer.utils import DPODataCollatorWithPadding

# disable_caching()

def get_line(a, b, op=None, format=None, train=None):
    if op == 'add':
        if format == 'reverse':
            return get_reverse_add(a, b)
        elif format == 'forward':
            return get_forward(a, b)
        elif format == 'forward2':
            return get_forward2(a, b)
    elif op == 'copy':
        if format == 'copy':
            return get_copy(a)
    elif op == 'reverse':
        return get_reverse(a)
    elif op =='mult':
        if format == 'COT':
            return get_COT_mult(a, b)
    elif op == 'maze':
        if format == 'wilson':
            return generate_maze_with_wilson(a, b)
        

    raise ValueError(f'Unknown op or format: {op}, {format}')

def data_generator(
    op: str,
    format: str,
    task_id: int,
    n_digits_a_range: Tuple[int] = None,
    n_digits_b_range: Union[Tuple[int], None] = None,
    train: bool = True,
    shard: List[range] = None,
    no_sample_set: set = None, # this is not used
    seed: int = None,
    lock_operand_length: bool = False,
):
    if n_digits_b_range is None:
        n_digits_b_range = n_digits_a_range
    
    if op in ['mult', 'maze']:
        lock_operand_length = False

    assert len(shard) == 1, f'Shard should be a list of one range, but got: {shard}'
    no_sample_hit = 0
    if not train:
        seed = 1000 + seed
    random.seed(seed + shard[0].start + n_digits_a_range[1] + n_digits_b_range[1])
    print(f'Generating data for {op} {format} {task_id} {n_digits_a_range} {n_digits_b_range} {train} {shard[0]}')
    for _ in shard[0]:
        if not op == 'maze':
            nda = random.sample(range(*n_digits_a_range), 1)[0]
            if lock_operand_length:
                ndb = nda
            else:
                ndb = random.sample(range(*n_digits_b_range), 1)[0]

            a = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(nda - 1)])
            b = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(ndb - 1)])
            # print('-----------')
            # print(format, _, a, b)
            # print('-----------')
            if no_sample_set is not None and (a, b) in no_sample_set:
                no_sample_hit += 1
                if no_sample_hit > 100:
                    raise ValueError(f'No sample hit {no_sample_hit} times')
                continue
        else: # maze case
            nda = random.sample(range(*n_digits_a_range), 1)[0]
            ndb = random.sample(range(*n_digits_b_range), 1)[0]
            a, b = nda, ndb

        prompt, target, loss_mask = get_line(a, b, op=op, format=format, train=train)

        if loss_mask is None:
            loss_mask = [1] * len(target)

        yield {
            'prompt': prompt,
            'target': target,
            'loss_mask': loss_mask,
            'n_digits': (nda, ndb),
            'task_id': task_id
        }

def get_dataset_display_name(n_digits, op, format, n_digit_range):
    if isinstance(n_digits, int) and n_digits in n_digit_range or isinstance(n_digits, tuple) and all([nd in range(*ndr) for nd, ndr in zip(n_digits, n_digit_range)]):
        prefix = 'ID'
    else:
        prefix = 'OOD'

    if isinstance(n_digits, tuple) and n_digits[0] == n_digits[1]:
        n_digits = n_digits[0]
    return f'{prefix}-{n_digits}-{op}-{format}'

def get_dataset_display_name_mult(n_digit_a, n_digit_b, op, format, max_digits):
    # for mult or mazes
    if op == 'mult':
        if n_digit_a <= max_digits and n_digit_b <= max_digits:
            prefix = 'ID'
        else:
            prefix = 'OOD'
        return f'{prefix}-{n_digit_a}-{n_digit_b}-{op}-{format}'
    else:
        return f'{n_digit_a}-{n_digit_b}-{op}-{format}'

def get_task_length(args: ScriptArguments, num_digits: int):
    max_length = 0
    for op in args.op_train:
        if op == 'add':
            max_length = max(max_length, num_digits * 3 + 5)
        elif op == 'copy' or op == 'reverse':
            max_length = max(max_length, num_digits * 2 + 3)
        else:
            # raise ValueError(f'Max length not implemented for {op}')
            return 0
    
    return max_length

def get_shards(num_total, num_workers=0):
    if num_workers > 0:
        shards = [range(i * round(num_total // num_workers), max(1, (i + 1) * round(num_total // num_workers))) for i in range(num_workers)]
        shards = [sh for sh in shards if len(sh) > 0]
    else:
        shards = [range(round(num_total))]
    return shards

def get_additonal_train_data(args: ScriptArguments, train_args: MyTrainingArguments, tokenizer: PreTrainedTokenizer, ) -> List[Dataset]:
    extra_ds = []
    if args.additional_train_data is not None:
        for dat in args.additional_train_data:
            print(f'Loading additional training data from {dat}')
            ds2 = Dataset.load_from_disk(dat, keep_in_memory=True)
            # ds2 = ds2.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
            # ds2 = ds2.map(tokenization, batched=True, batch_size=1000, remove_columns=remove_columns)
            # ds2 = ds2.map(features=ds.features)
            extra_ds.append(ds2)

    return extra_ds

def get_train_dataset(args: ScriptArguments, train_args: MyTrainingArguments, tokenizer: PreTrainedTokenizer, no_sample_from: Dict[str, Dataset]=None, additional_train_data: List[Dataset]=None):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def mask_target(target_ids, loss_mask):
        return [t if m == 1 else -100 for t, m in zip(target_ids, loss_mask)]

    def tokenization(batch):
        batch_new = {}
        prompt_ids = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False)
        target_ids = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)
        batch_new['input_ids'] = [p + t for p, t in zip(prompt_ids['input_ids'], target_ids['input_ids'])]
        batch_new['labels'] = [[-100] * (len(p)) + mask_target(t, m) for p, t, m in zip(prompt_ids['input_ids'], target_ids['input_ids'], batch['loss_mask'])]
        batch_new['attention_mask'] = [p + t for p, t in zip(prompt_ids['attention_mask'], target_ids['attention_mask'])]
        # batch_new['labels'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        # batch_new['example_ids'] = [[i] * len(p + t) for i, (p, t) in enumerate(zip(prompt_ids, target_ids))]
        return batch_new
    
    for key in no_sample_from:
        no_sample_from[key] = no_sample_from[key].to_dict()
        no_sample_from[key]['prompt'] = set(no_sample_from[key]['prompt']) # convert to set for faster lookup

    def filter_eval(example, opi=None):
        if args.op_train[0] == 'mult' or args.op_train[0] == 'maze':
            if example['n_digits'][0] < 5 or example['n_digits'][1] < 5:
                # We cannot avoid repeating examples with low n_digits
                return True
            max_train_digits = max(range(*args.n_digits_a_train[i]).stop - 1 for i in range(len(args.op_train)))
            key = get_dataset_display_name_mult(example['n_digits'][0], example['n_digits'][1], args.op_train[opi], args.format_train[opi], max_train_digits)
        else:
            if example['n_digits'][0] != example['n_digits'][1]:
                # We don't sample asymmetric examples in test, so these are definitely good
                return True
            elif example['n_digits'][0] < 8:
                # We cannot avoid repeating examples with low n_digits
                return True
            
            key = get_dataset_display_name(example['n_digits'], args.op_train[opi], args.format_train[opi], (args.n_digits_a_train[opi], args.n_digits_b_train[opi]))

        if key not in no_sample_from:
            return True
        return example['prompt'] not in no_sample_from[key]['prompt']

    ds_list = []
    if len(args.op_dist_train) > 1:
        fracs = [max(x,y) for x,y in zip(*args.op_dist_train)]
    else:
        fracs = args.op_dist_train[0]

    for opi, frac in enumerate(fracs):
        if args.use_iterable_dataset:
            ds_class = IterableDataset
            kwargs = {}
        else:
            ds_class = Dataset
            kwargs = {'num_proc': args.num_workers}
        gen_kwargs = {
            'train': True,
            'op': args.op_train[opi],
            'task_id': opi,
            'format': args.format_train[opi],
            'n_digits_a_range': args.n_digits_a_train[opi],
            'n_digits_b_range': args.n_digits_b_train[opi] if args.n_digits_b_train is not None else None,
            'shard': get_shards(args.num_train[opi] * frac, args.num_workers),
            'seed': train_args.seed,
            'lock_operand_length': args.lock_operand_length
        }
        if args.op_train[0] == 'mult' or args.op_train[0] == 'maze':
            gen_kwargs['lock_operand_length'] = False
            gen_kwargs['n_digits_b_range'] = args.n_digits_b_train[opi]

        ds = ds_class.from_generator(
            data_generator,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        ds.info.description = json.dumps((args.n_digits_a_train[opi], args.n_digits_b_train[opi]))
        ds = ds.filter(partial(filter_eval, opi=opi), **kwargs)
        ds = ds.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens}, **kwargs)
        remove_columns = ['prompt', 'target', 'loss_mask', 'n_digits']
        if not train_args.track_num_tokens_seen_by_task:
            remove_columns.append('task_id')
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=remove_columns, **kwargs)
        # if not args.use_iterable_dataset:
        #     ds = ds.to_iterable_dataset(num_shards=args.num_workers)
        ds_list.append(ds)

    init_probs = [frac / sum(args.op_dist_train[0]) for frac in args.op_dist_train[0]]
    if len(args.op_dist_train) > 1:
        from multiprocessing import Array
        from ctypes import c_double
        init_probs = Array(c_double, init_probs)

    init_probs_uniform = np.full(len(args.op_dist_train[0]), 1/len(args.op_dist_train[0]))
    is_close_to_uniform = np.allclose(init_probs, init_probs_uniform)
    if is_close_to_uniform:
        init_probs = None

    ds = interleave_datasets(ds_list, probabilities=init_probs, seed=train_args.seed, stopping_strategy='all_exhausted')

    max_train_digits = max(max(nda[1], ndb[1]) for nda, ndb in zip(args.n_digits_a_train, args.n_digits_b_train))

    if isinstance(additional_train_data, list) and len(additional_train_data) > 0:
        additional_probs = []
        for i, ds2 in enumerate(additional_train_data):
            if args.op_train[0] == 'mult' or args.op_train[0] == 'maze':
                nds, nde = 0, 0
            else:
                nds, nde = json.loads(ds2.info.description)[0]
            max_train_digits = max(max_train_digits, int(nde))
            print(max_train_digits)
            if args.fast_self_improve:
                additional_probs.append(float(nde) - float(nds))
            else:
                additional_probs.append(1)
            remove_columns = [name for name in ds2.column_names if 'real' in name]
            features_to_align = None
            if {k: v for k, v in ds2.features.items() if k not in remove_columns} != ds.features:
                features_to_align = ds.features
            ds2 = ds2.map(remove_columns=remove_columns, batched=True, batch_size=50000, features=features_to_align) # To prevent unaligned features
    
            if args.use_iterable_dataset:
                ds2 = ds2.to_iterable_dataset(num_shards=args.num_workers)
            additional_train_data[i] = ds2
        if args.n_digits_train is not None:
            probs = [len(range(*args.n_digits_train[0]))] + additional_probs[:-1]
        else:
            probs = [len(range(*args.n_digits_a_train[0]))] + additional_probs[:-1]
        probs = [p / sum(probs) * 0.5 for p in probs] + [0.5]
        ds = interleave_datasets([ds] + additional_train_data, probabilities=probs, seed=train_args.seed, stopping_strategy='all_exhausted')

    # l = []
    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 3):
        # print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        # print(example['labels'])
        print(tokenizer.decode(example['labels']))
        # breakpoint()
    #     l.append(len(example['input_ids']))
    
    if not args.use_train_attention_mask:
        ds = ds.remove_columns('attention_mask')

    return ds, max_train_digits

def get_eval_dataset(args: ScriptArguments, train_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False, return_token_type_ids=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        for k in batch_new.keys():
            batch_new['eval_' + k] = batch_new.pop(k)
        batch_new['eval_loss_mask'] = batch['loss_mask']
        return batch_new

    ds_list = {}
    unmapped_ds_list = {}
    if args.n_digits_eval is None:
        eval_n_digits = itertools.product(range(*args.n_digits_a_eval), range(*args.n_digits_b_eval))
    else:
        eval_n_digits = zip(range(*args.n_digits_a_eval), range(*args.n_digits_b_eval))

    for n_digits_a, n_digits_b in eval_n_digits:
        n_digits = (n_digits_a, n_digits_b)
        for opi, frac in enumerate(args.op_dist_eval):
            os.makedirs(args.eval_cache_loc, exist_ok=True)
            eval_file = os.path.join(args.eval_cache_loc, f'{args.op_eval[opi]}-{args.format_eval[opi]}-{n_digits}-{args.num_eval}-{train_args.seed}-{tokenizer.__class__.__name__}')
            # if os.path.exists(eval_file):
            if False:
                ds0 = Dataset.load_from_disk(eval_file)
            else:
                ds0 = Dataset.from_generator(
                    data_generator,
                    gen_kwargs={
                        'train': False, 
                        'op': args.op_eval[opi],
                        'task_id': opi,
                        'format': args.format_eval[opi],
                        'n_digits_a_range': (n_digits_a, n_digits_a + 1),
                        'n_digits_b_range': (n_digits_b, n_digits_b + 1),
                        'shard': get_shards(args.num_eval * frac, args.num_workers),
                        'seed': train_args.seed,
                        'lock_operand_length': args.lock_operand_length # it doesn't matter here because range is always 1
                    },
                    num_proc=args.num_workers if args.num_workers > 0 else None,
                    keep_in_memory=True,
                    cache_dir=eval_file,
                    split='test',
                )
                ds0.info.description = json.dumps(((n_digits_a, n_digits_a + 1), (n_digits_b, n_digits_b + 1)))
                ds0.cleanup_cache_files()
                # for f in ds0.cache_files:
                #     shutil.rmtree(os.path.dirname(f['filename']))
                # ds0.save_to_disk(eval_file) 
            ds = ds0.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
            ds = ds.map(tokenization, batched=True, batch_size=args.num_eval, remove_columns=['prompt', 'target', 'n_digits', 'loss_mask', 'task_id'])
            key = get_dataset_display_name(n_digits, args.op_eval[opi], args.format_eval[opi], (args.n_digits_a_train[opi], args.n_digits_b_train[opi]))
            ds_list[key] = ds
            unmapped_ds_list[key] = ds0
            # print(f'cleaned up {ds_list[n_digits].cleanup_cache_files()}')

    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in ds.take(1):
            print(example['eval_input_ids'])
            print(tokenizer.decode(example['eval_input_ids']))
            print(example['eval_labels'])
            print(tokenizer.decode(example['eval_labels']))

    return ds_list, unmapped_ds_list

from transformers import Trainer, PreTrainedModel, GenerationConfig
from torch.utils.data import DataLoader

def get_self_improve_dataset_path(args: ScriptArguments, train_args: Seq2SeqTrainingArguments):
    return os.path.join(args.self_improve_cache_loc, train_args.run_name + f'-{args.num_self_improve_data}')

def get_self_improve_dataset(args: ScriptArguments, train_args: Seq2SeqTrainingArguments, trainer: Trainer, tokenizer: PreTrainedTokenizer, current_n_digits: Tuple[int, int] = None):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False, return_token_type_ids=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        for k in batch_new.keys():
            batch_new['eval_' + k] = batch_new.pop(k)
        batch_new['eval_loss_mask'] = batch['loss_mask']
        return batch_new
    
    def mask_target(target_ids, loss_mask):
        return [t if m == 1 else -100 for t, m in zip(target_ids, loss_mask)]
    
    def filter_self_improve(example, filter_type):
        if filter_type == 'length':
            return len(example['predictions']) == len(example['eval_labels']) and tokenizer.eos_token_id in example['predictions']
        else:
            raise ValueError(f'Unknown filter type: {filter_type}')
    
    def add_noise(x, tokenizer, noise_prob, noise_type, digit_toks):
        noise_count = round(noise_prob * len(x['input_ids']))
        random_idx = list(range(len(x['input_ids'])))
        random.shuffle(random_idx)
        noise_idx = random_idx[:noise_count]
        for idx in noise_idx:
            for key in ['input_ids', 'labels']:
                inp = x[key][idx]
                if noise_type == 'uniform':
                    noise = random.choices(digit_toks, k=len(inp)) # this is specific to the character tokenizer
                elif noise_type == 'drop-uniform':
                    trunc = random.randint(1, 3)
                    noise = random.choices(digit_toks, k=len(inp))
                    noise[-trunc:] = [tokenizer.pad_token_id] * trunc
                elif noise_type == 'drop-digits':
                    drop_idx = random.sample(range(len(inp) - 3, len(inp)), random.randint(1, 3))
                    # drop_idx = [len(inp) - 1]
                    noise = [tok for i, tok in enumerate(inp) if i not in drop_idx] + [tokenizer.pad_token_id] * len(drop_idx)
                elif noise_type == 'preturb':
                    preturb_idx = random.sample(range(len(inp) - 3, len(inp)), random.randint(1, 3))
                    # preturb_idx = [len(inp) - 1]
                    noise = [np.clip(tok + random.randint(-1, 1), min(digit_toks), max(digit_toks)) if i in preturb_idx else tok for i, tok in enumerate(inp)]
                elif noise_type == 'fix-digits':
                    fix = digit_toks[0]
                    noise = [fix if i > len(inp) - 5 else tok for i, tok in enumerate(inp)]
                elif noise_type == 'drop-preturb':
                    preturb_idx = random.sample(range(len(inp) - 3, len(inp)), random.randint(1, 3))
                    inp = [np.clip(tok + random.randint(-1, 1), min(digit_toks), max(digit_toks)) if i in preturb_idx else tok for i, tok in enumerate(inp)]
                    drop_idx = random.sample(range(len(inp) - 3, len(inp)), random.randint(1, 3))
                    noise = [tok for i, tok in enumerate(inp) if i not in drop_idx] + [tokenizer.pad_token_id] * len(drop_idx)
                else:
                    raise ValueError(f'Unknown noise type: {noise_type}')

                x[key][idx] = noise

        return x
    
    def convert_to_train(example):
        real_input_ids = example['eval_input_ids'] + example['eval_labels']
        input_ids = example['eval_input_ids'] + example['predictions']
        real_labels = [-100] * len(example['eval_input_ids']) + mask_target(example['eval_labels'], example['eval_loss_mask'])
        labels = [-100] * len(example['eval_input_ids']) + example['predictions']
        attention_mask = example['eval_attention_mask'] + [1] * len(example['predictions'])
        
        if args.sanity_check_self_improve:
            input_ids = real_input_ids
            labels = real_labels
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'real_input_ids': real_input_ids,
            'real_labels': real_labels
        }

    opi = 0
    frac = 1 # hardcode because we assume only one eval task
    n_digits_a_start, n_digits_b_start = current_n_digits # current_n_digits is the end of the range of the most recent training data
    if args.lock_operand_length_SI:
        # increase both operands at the same time
        n_digits_a_end = n_digits_a_start + 1
        n_digits_b_end = n_digits_b_start + 1
    else:
        # increase the shortest operand first
        n_digits_a_end = n_digits_a_start + 1 if n_digits_a_start <= n_digits_b_start else n_digits_a_start
        n_digits_b_end = n_digits_b_start + 1 if n_digits_b_start < n_digits_a_start else n_digits_b_start

    # test for the highest n_digit_end that still has good performance
    while args.self_improve_ood_threshold is not None:
        assert args.lock_operand_length_SI, 'fast_self_improve only works with lock_operand_length_SI'
        ds0 = Dataset.from_generator(
            data_generator,
            gen_kwargs={
                'train': False, 
                'op': args.op_eval[opi],
                'task_id': opi,
                'format': args.format_eval[opi],
                'n_digits_a_range': (n_digits_a_end-1, n_digits_a_end), # test performancne of n_digits_end
                'n_digits_b_range': (n_digits_b_end-1, n_digits_b_end), # test performancne of n_digits_end
                'shard': get_shards(args.num_eval * 2 * frac, args.num_workers),
                'seed': train_args.seed,
                'lock_operand_length': args.lock_operand_length
            },
            num_proc=args.num_workers if args.num_workers > 0 else None,
            keep_in_memory=True,
            split='pre_self_improve'
        )
        ds0.info.description = json.dumps(((n_digits_a_end-1, n_digits_a_end), (n_digits_b_end-1, n_digits_b_end)))
        ds0.cleanup_cache_files()
        ds = ds0.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=['prompt', 'target', 'n_digits', 'loss_mask', 'task_id'])

        result = trainer.evaluate(ds, metric_key_prefix=f'pre_self_improve_{n_digits_a_end-1}')
        print(f'Test n_digits: ({n_digits_a_end-1}, {n_digits_b_end-1}) {result[f"pre_self_improve_{n_digits_a_end-1}_accuracy"]}')
        if result[f'pre_self_improve_{n_digits_a_end-1}_accuracy'] >= args.self_improve_ood_threshold:
            n_digits_a_end += 1
            n_digits_b_end += 1
        else:
            if n_digits_a_end - n_digits_a_start > 1: # We have to collect at least 1
                n_digits_a_end -= 1
            if n_digits_b_end - n_digits_b_start > 1:
                n_digits_b_end -= 1
            break

    if not args.fast_self_improve:
        n_digits_a_end = n_digits_a_start + 1
        n_digits_b_end = n_digits_b_start + 1
    print(f'Using n_digits: ({n_digits_a_start}, {n_digits_a_end}), ({n_digits_b_start}, {n_digits_b_end})')

    os.makedirs(args.self_improve_cache_loc, exist_ok=True)
    cache_file = get_self_improve_dataset_path(args, train_args)

    count = 0
    self_improve_dataset = []
    while count < args.num_self_improve_data: # outer loop for filtering
        ds0 = Dataset.from_generator(
            data_generator,
            gen_kwargs={
                'train': False,
                'op': args.op_eval[opi],
                'task_id': opi,
                'format': args.format_eval[opi],
                'n_digits_a_range': (n_digits_a_start, n_digits_a_end),
                'n_digits_b_range': (n_digits_b_start, n_digits_b_end),
                'shard': get_shards(args.num_self_improve_data * frac, args.num_workers),
                'seed': train_args.seed,
                'lock_operand_length': True
            },
            num_proc=args.num_workers if args.num_workers > 0 else None,
            keep_in_memory=True
        )
        ds0.info.description = json.dumps(((n_digits_a_start, n_digits_a_end), (n_digits_b_start, n_digits_b_end)))
        ds = ds0.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=['prompt', 'target', 'n_digits', 'loss_mask', 'task_id'])
        ds.cleanup_cache_files()

        pred = trainer.predict(ds)
        print(pred.metrics)
        pred_sequences = []
        padded_pred = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        for batch in padded_pred:
            for seq in batch.tolist():
                pred_sequences.append([p for p in seq if p != tokenizer.pad_token_id])
        ds = ds.add_column('predictions', pred_sequences)

        if args.filter_self_improve is not None:
            ds = ds.filter(filter_self_improve, fn_kwargs={'filter_type': args.filter_self_improve})
        if args.num_self_improve_data - count < len(ds):
            ds = ds.take(args.num_self_improve_data - count)
        count += len(ds)
        self_improve_dataset.append(ds)
        print(f'Collected {count}/{args.num_self_improve_data} examples')

    assert len(self_improve_dataset) > 0, 'No self-improve data collected'
    self_improve_dataset = concatenate_datasets(self_improve_dataset) if len(self_improve_dataset) > 1 else self_improve_dataset[0]
    if args.filter_self_improve is not None:
        pred = trainer.predict(self_improve_dataset.remove_columns(['predictions']), metric_key_prefix='post_filter')
        print('After filtering:')
        print(pred.metrics)
    if args.save_self_improve_data:
        self_improve_dataset = self_improve_dataset.map(
            convert_to_train,
            batched=False, num_proc=args.num_workers if args.num_workers > 0 else None,
            remove_columns=['eval_input_ids', 'eval_labels', 'eval_attention_mask', 'eval_loss_mask', 'predictions'],
            features=datasets.Features({'input_ids': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'labels': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'attention_mask': datasets.Sequence(feature=datasets.Value(dtype='int32')), 
                    'real_input_ids': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'real_labels': datasets.Sequence(feature=datasets.Value(dtype='int32'))})
        )
        if args.add_noise is not None and args.add_noise > 0 and (args.add_noise_at is None or args.add_noise_at == args.self_improve_round):
            digit_toks = tokenizer.convert_tokens_to_ids([str(i) for i in range(10)])
            # self_improve_dataset.cleanup_cache_files()
            self_improve_dataset = self_improve_dataset.map(
                add_noise, batched=True, batch_size=1000,
                fn_kwargs={'tokenizer': tokenizer, 'noise_prob': args.add_noise, 'noise_type': args.noise_type, 'digit_toks': digit_toks},
                num_proc=args.num_workers if args.num_workers > 0 else None
            )
        self_improve_dataset.save_to_disk(cache_file)

    print('----------- Examples from self-improve: -------------')
    for example in itertools.islice(self_improve_dataset, 0, 5):
        # print(len(example['input_ids']))
        print(tokenizer.decode(example['input_ids']))
        # print(len(example['labels']))
        # print(tokenizer.decode(example['labels']))

    return self_improve_dataset, pred.metrics


def get_self_improve_dataset2(args: ScriptArguments, train_args: Seq2SeqTrainingArguments, trainer: Trainer, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False, return_token_type_ids=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        for k in batch_new.keys():
            batch_new['eval_' + k] = batch_new.pop(k)
        batch_new['eval_loss_mask'] = batch['loss_mask']
        return batch_new
    
    def mask_target(target_ids, loss_mask):
        return [t if m == 1 else -100 for t, m in zip(target_ids, loss_mask)]

    def do_majority_vote(trainer, ds, train_args, args, tokenizer):
        # load trainer for all models with args.majority_vote_seeds
        # for each example in eval dataset, do majority vote on the predictions of each models
        from collections import Counter
        print("="*20 + 'Doing majority vote' + "="*20)
        model_seeds = args.majority_vote_seeds.split(',')

        # first check if all models are available
        print(f'Checking if all models are available')
        model_list = []
        for seed in model_seeds:
            ckpt_num = train_args.resume_from_checkpoint.split('/')[-1]
            model_checkpoint_to_test = train_args.resume_from_checkpoint.split('seed-')[0] + f'seed-{seed}/{ckpt_num}'
            if not os.path.exists(model_checkpoint_to_test):
                print(f'Model checkpoint {model_checkpoint_to_test} not found. Exiting...')
                exit()
            else:
                print(f'Exists model from {model_checkpoint_to_test}')
                model_list.append(model_checkpoint_to_test)

        print("="*20 + 'All models available' + "="*20)

        prediction_list = [] # num_models x num_examples x seq_len
        pred_metric_dict = {}

        for model_checkpoint_to_test in model_list:
            print(f'Loading model from {model_checkpoint_to_test}')
            trainer._load_from_checkpoint(resume_from_checkpoint=model_checkpoint_to_test)
            pred = trainer.predict(ds)
            pred_metric_dict[model_checkpoint_to_test] = pred.metrics
            
            pred_sequences = []
            padded_pred = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            
            for batch in padded_pred:
                for seq in batch.tolist():
                    pred_sequences.append([p for p in seq])
            
            prediction_list.append(pred_sequences)

        # Majority voting
        majority_voted_predictions = []
        selected_indices = []
        average_selected_votes = 0
        threshold = args.majority_voting_threshold * len(model_seeds)

        # Transpose the list to get predictions per example
        transposed_predictions = list(zip(*prediction_list)) # num_examples x num_models x seq_len

        print("="*20 + 'Majority voting from predictions' + "="*20)
        for idx, example_preds in enumerate(transposed_predictions):
            # Convert each prediction to a tuple for hashing
            example_preds_as_tuples = [tuple(pred) for pred in example_preds]
            
            # Count votes for each unique tuple
            vote_counts = Counter(example_preds_as_tuples)
            most_common_pred, count = vote_counts.most_common(1)[0]
            
            # Apply threshold
            if count >= threshold:
                majority_voted_predictions.append(list(most_common_pred))
                average_selected_votes += count
                selected_indices.append(idx)
            # else:
            #     majority_voted_predictions.append(None)  # None if threshold is not met
        
        print("="* 20)
        print(f'Number of examples selected: {len(majority_voted_predictions)}')
        print(f'Average votes among selected: {average_selected_votes / len(majority_voted_predictions)}')
        
        pred_metric_dict['num_selected'] = len(majority_voted_predictions)
        pred_metric_dict['average_votes'] = average_selected_votes / len(majority_voted_predictions)
        pred_metric_dict['short'] = None

        truncated_ds = ds.select(selected_indices)

        # additionally truncate pad_tokens
        truncated_mv_pred = []
        remove_indices = []
        for i, mv_pred in enumerate(majority_voted_predictions):
            if args.filter_shorter_additional_data:
                if mv_pred[(-1 * args.filter_shorter_additional_data_threshold)] == tokenizer.pad_token_id:
                    remove_indices.append(i)
                else:
                    truncated_mv_pred.append([p for p in mv_pred if p != tokenizer.pad_token_id])
            else:
                truncated_mv_pred.append([p for p in mv_pred if p != tokenizer.pad_token_id])
        
        if args.filter_shorter_additional_data:
            truncated_ds = truncated_ds.select([i for i in range(len(truncated_ds)) if i not in remove_indices])
            pred_metric_dict['short'] = len(remove_indices)
            print(f'Removed {len(remove_indices)} short examples')

        assert len(truncated_ds) == len(truncated_mv_pred)

        truncated_ds = truncated_ds.add_column('predictions', truncated_mv_pred)

        return truncated_mv_pred, truncated_ds, pred_metric_dict
    
    def convert_to_train(example):
        real_input_ids = example['eval_input_ids'] + example['eval_labels']
        input_ids = example['eval_input_ids'] + example['predictions']
        real_labels = [-100] * len(example['eval_input_ids']) + mask_target(example['eval_labels'], example['eval_loss_mask'])
        labels = [-100] * len(example['eval_input_ids']) + example['predictions']
        attention_mask = example['eval_attention_mask'] + [1] * len(example['predictions'])
        
        if args.sanity_check_self_improve:
            input_ids = real_input_ids
            labels = real_labels
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'real_input_ids': real_input_ids,
            'real_labels': real_labels
        }

    opi = 0
    frac = 1 # hardcode because we assume only one eval task
    if args.additional_train_data_max_digit is not None:
        n_digits_start = args.additional_train_data_max_digit + 1 
    elif args.n_digits_train is not None:
        n_digits_start = range(*args.n_digits_train[0]).stop
    else:
        n_digits_start = range(*args.n_digits_a_train[0]).stop

    n_digits_end = n_digits_start + 1
    
    os.makedirs(args.self_improve_cache_loc, exist_ok=True)
    cache_file = os.path.join(args.self_improve_cache_loc, f'{args.op_eval[opi]}-{args.format_eval[opi]}-{n_digits_start}-{n_digits_end}-{args.num_self_improve_data}-{args.self_improve_round}-{train_args.seed}')

    gen_kwargs={
        'train': False,
        'op': args.op_eval[opi],
        'task_id': opi,
        'format': args.format_eval[opi],
        'n_digits_a_range': (n_digits_start, n_digits_end),
        'shard': get_shards(args.num_self_improve_data * frac, args.num_workers),
        'seed': train_args.seed,
        'lock_operand_length': True,
    }
    
    num_pairs = 1 # pairs of self-improve data digits

    if args.op_train[opi] == 'mult' or args.op_train[0] == 'maze':
        num_pairs = len(args.additional_train_data_digit)
        if num_pairs == 1:
            b = args.additional_train_data_digit[0]
            name = str(b).replace(', ', '-').replace('(', '').replace(')', '')
        else:
            name = str(args.additional_train_data_digit).replace(', ', '-')
        cache_file = os.path.join(args.self_improve_cache_loc, f'{args.op_eval[opi]}-{args.format_eval[opi]}-{name}-{args.num_self_improve_data}-{args.self_improve_round}-{train_args.seed}')
        print('file name: ', cache_file)
    
    if args.majority_voting:
        if args.majority_vote_keep_seed:
            cache_file = cache_file + '-mv'
            gen_kwargs['seed'] = train_args.seed
        else:
            cache_file = '-'.join(cache_file.split('-')[:-1]) + '-mv'
            gen_kwargs['seed'] = 0 # Let's first have the generator seed fixed

    ds_list = []
    for pair in range(num_pairs):
        if args.op_train[opi] == 'mult' or args.op_train[0] == 'maze':
            gen_kwargs['n_digits_a_range'] = (args.additional_train_data_digit[pair][0], args.additional_train_data_digit[pair][0] + 1)
            gen_kwargs['n_digits_b_range'] = (args.additional_train_data_digit[pair][1], args.additional_train_data_digit[pair][1] + 1)

        else:
            gen_kwargs['n_digits_a_range'] = (n_digits_start, n_digits_end)
            gen_kwargs['n_digits_b_range'] = (n_digits_start, n_digits_end)
        print(f'Generating self-improve data for digits: {gen_kwargs["n_digits_a_range"]} and {gen_kwargs["n_digits_b_range"]}')
        
        self_improve_dataset = []
        ds0 = Dataset.from_generator(
            data_generator,
            gen_kwargs=gen_kwargs,
            num_proc=args.num_workers if args.num_workers > 0 else None,
            keep_in_memory=True
        )
        ds0.info.description = json.dumps((gen_kwargs['n_digits_a_range'], gen_kwargs['n_digits_b_range']))
        ds = ds0.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=['prompt', 'target', 'n_digits', 'loss_mask', 'task_id'])
        ds.cleanup_cache_files()

        if args.majority_voting:
            trunc_pred, ds, pred_metric_dict = do_majority_vote(trainer, ds, train_args, args, tokenizer)
            metric = pred_metric_dict
            ds.info.description += "\n" + json.dumps((pred_metric_dict['num_selected'], pred_metric_dict['average_votes'], pred_metric_dict['short']))
        else:
            pred = trainer.predict(ds)
            print(pred.metrics)
            pred_sequences = []
            padded_pred = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
            
            total_seq_idx = 0
            remove_indices = []
            for batch in padded_pred:
                for seq in batch.tolist():
                    if args.filter_shorter_additional_data:
                        if seq[(-1 * args.filter_shorter_additional_data_threshold)] == tokenizer.pad_token_id:
                            remove_indices.append(total_seq_idx)
                        else:
                            pred_sequences.append([p for p in seq if p != tokenizer.pad_token_id])
                        total_seq_idx += 1
                    else:
                        pred_sequences.append([p for p in seq if p != tokenizer.pad_token_id])
            
            if args.filter_shorter_additional_data:
                # remove remove_indices from ds
                ds = ds.select([i for i in range(len(ds)) if i not in remove_indices])
                print(f'Removed {len(remove_indices)} short examples')
                assert len(ds) == len(pred_sequences), 'Length mismatch after filtering'

            ds = ds.add_column('predictions', pred_sequences)
            metric = pred.metrics

        self_improve_dataset.append(ds)

        assert len(self_improve_dataset) > 0, 'No self-improve data collected'
        self_improve_dataset = concatenate_datasets(self_improve_dataset) if len(self_improve_dataset) > 1 else self_improve_dataset[0]
        print(metric)

        if args.save_self_improve_data:
            self_improve_dataset = self_improve_dataset.map(
                convert_to_train,
                batched=False, num_proc=args.num_workers if args.num_workers > 0 else None,
                remove_columns=['eval_input_ids', 'eval_labels', 'eval_attention_mask', 'eval_loss_mask', 'predictions'],
                features=datasets.Features({'input_ids': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'labels': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'attention_mask': datasets.Sequence(feature=datasets.Value(dtype='int32')), 
                        'real_input_ids': datasets.Sequence(feature=datasets.Value(dtype='int32')), 'real_labels': datasets.Sequence(feature=datasets.Value(dtype='int32'))})
            )
            ds_list.append(self_improve_dataset)

    self_improve_ds = interleave_datasets(ds_list, seed=train_args.seed, stopping_strategy='all_exhausted')

    if args.save_self_improve_data:
        self_improve_ds.save_to_disk(cache_file)
        print(f'Self-improve dataset saved to {cache_file}')

    print('----------- Examples from self-improve: -------------')
    for example in itertools.islice(self_improve_ds, 0, 5):
        # print(len(example['input_ids']))
        print(tokenizer.decode(example['input_ids']))
        # print(len(example['labels']))
        # print(tokenizer.decode(example['labels']))
    
    print(f'Self-improve dataset saved to {cache_file}')
    print(f'Self-improve dataset size: {len(self_improve_ds)}')


    return self_improve_ds, metric


class PromptAnswerDataCollator(DPODataCollatorWithPadding):
    left_pad_list = ['prompt', 'eval_input_ids', 'eval_attention_mask']
    rand_pad_list = []
    ignore_list = ['task_id']

    def __init__(self, pad_token_id=None, label_pad_token_id=None, train_pad_side='right', train_pad_to=None, eval_pad_to=None):
        super().__init__(pad_token_id=pad_token_id, label_pad_token_id=label_pad_token_id)
        if train_pad_side == 'left':
            self.left_pad_list += ['input_ids', 'attention_mask', 'labels']
        elif train_pad_side == 'random':
            self.rand_pad_list += ['input_ids', 'attention_mask', 'labels']

        self.train_pad_to = train_pad_to
        self.eval_pad_to = eval_pad_to

    def get_rand_pad(self, features):
        key = self.rand_pad_list[0]
        if key in features:
            feat = features[key]
            max_len = max([len(ex) for ex in feat])
            pad_amt = [max_len - len(ex) for ex in feat]
            self.rand_pad_amt = [random.randint(0, pad) for pad in pad_amt]
            # self.rand_pad_amt = [pad for pad in pad_amt]
        else:
            self.rand_pad_amt = None

    def __call__(self, features):
        features = {
            key: [example[key] for example in features] for key in features[0].keys()
        }

        if len(self.rand_pad_list) > 0:
            self.get_rand_pad(features)

        padded_batch = {}
        for k, feat in features.items():
            if k in self.ignore_list:
                padded_batch[k] = torch.tensor(feat)
                continue

            if k in self.left_pad_list:
                to_pad = [torch.LongTensor(ex[::-1]) for ex in feat]
            else:
                to_pad = [torch.LongTensor(ex) for ex in feat]
            
            if k.endswith("input_ids") or k.endswith('eval_labels'):
                if self.pad_token_id is None:
                    raise ValueError(
                        "Padding is enabled, but the tokenizer is not configured with a padding token."
                        " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                        " before calling the trainer."
                    )
                padding_value = self.pad_token_id
            elif k.endswith("labels"):
                padding_value = self.label_pad_token_id
            elif k.endswith("attention_mask"):
                padding_value = 0
            elif k.endswith('loss_mask'):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            
            # remove the eval_ prefix to conform to model input names
            if 'eval_' in k:
                is_train = False
                input_k = k.replace('eval_', '')
            else:
                is_train = True
                input_k = k

            if k in self.rand_pad_list:
                max_len = max([len(ex) for ex in to_pad])
                pad_amt = [max_len - len(ex) for ex in to_pad]
                left_pad = self.rand_pad_amt
                padded_batch[input_k] = torch.stack([torch.nn.functional.pad(ex, (lp, pad - lp), value=padding_value) for ex, lp, pad in zip(to_pad, left_pad, pad_amt)], dim=0)
            else:
                padded_batch[input_k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            
            # max_len = max([len(ex) for ex in to_pad])
            # max_len = (max_len // 16 + 1) * 16
            # pad_amt = [max_len - len(ex) for ex in to_pad]
            # left_pad = self.rand_pad_amt if k in self.rand_pad_list else [0] * len(to_pad)
            # padded_batch[input_k] = torch.stack([torch.nn.functional.pad(ex, (lp, pad - lp), value=padding_value) for ex, lp, pad in zip(to_pad, left_pad, pad_amt)], dim=0)

            pad_to = self.train_pad_to if is_train else self.eval_pad_to
            if pad_to is not None:
                if padded_batch[input_k].shape[1] <= pad_to:
                    padded_batch[input_k] = torch.nn.functional.pad(padded_batch[input_k], (0, pad_to - padded_batch[input_k].shape[1]), value=padding_value)
                else:
                    raise ValueError(f"Cannot pad {k} to max_length {pad_to} because it is already longer than that ({padded_batch[input_k].shape[1]})")

            if k in self.left_pad_list:
                padded_batch[input_k] = padded_batch[input_k].flip(dims=[1])

        return padded_batch
