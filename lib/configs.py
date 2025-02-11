from dataclasses import dataclass, field
import json
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers import Seq2SeqTrainingArguments

@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    run_name_prefix: str = ''
    do_backtrack_decoding: bool = False
    track_num_tokens_seen_by_task: bool = False
    early_stop: bool = False
    early_stop_metrics: Optional[List[str]] = field(default_factory=lambda: [r'ID.*accuracy'])
    early_stop_thresholds: Optional[List[float]] = field(default_factory=lambda: [0.99])
    reset_max_steps: bool = False

    eval_maze: Optional[bool] = False # if True, evaluate maze

    self_improve_stable_steps: Optional[int] = 0
    self_improve_decay_steps: Optional[int] = 0

    use_custom_scheduler: Optional[bool] = None # if not None, use CustomLRCallback
    reset_optimizer: Optional[bool] = False # if True, reset optimizer after loading trainer & before training
    
    def __post_init__(self):
        if self.run_name is not None: 
            self.run_name_prefix = self.run_name
        super().__post_init__()

@dataclass
class ScriptArguments:
    foo: str = 'bar'
    entropy_metrics: bool = False
    task_length: Optional[int] = None
    wandb_project: Optional[str] = 'self-improve'

    self_improve_round: Optional[int] = None

    # Model arguments
    model_id: Optional[str] = None
    from_pretrained: bool = False
    use_lora: bool = False
    lora_config: Optional[Union[dict, str]] = None
    architecture: Optional[str] = 'mamba'
    rope_theta: Optional[float] = np.inf
    partial_rotary_factor: Optional[float] = 1.0
    hidden_size: Optional[int] = 768
    intermediate_size: Optional[int] = 3072
    num_attention_heads: Optional[int] = 12
    state_size: Optional[int] = 16
    num_layers: Optional[int] = 32
    max_position_embeddings: Optional[int] = 1024
    freeze: Optional[str] = None
    freeze_except: Optional[str] = None
    dropout: Optional[float] = 0.0
    use_character_tokenizer: bool = True

    # Data arguments
    char_list: Optional[str] = None
    use_iterable_dataset: bool = True
    num_train: Optional[Union[Tuple[Tuple[int]], str]] = '20_000_000'
    num_eval: int = 100
    eval_cache_loc: str = 'data'
    n_digits_train: Optional[Union[Tuple[Tuple[int]], str]] = None
    n_digits_a_train: Optional[Union[Tuple[int], str]] = None
    n_digits_b_train: Optional[Union[Tuple[int], str]] = None
    n_digits_eval: Optional[Union[Tuple[int], str]] = None
    n_digits_a_eval: Optional[Union[Tuple[int], str]] = None
    n_digits_b_eval: Optional[Union[Tuple[int], str]] = None
    block_size: Optional[int] = 1024
    op_train: Optional[Union[Tuple[str], str]] = 'add'
    op_eval: Optional[Union[Tuple[str], str]] = 'add'
    op_dist_train: Optional[Union[Tuple[float], str]] = '1'
    op_dist_eval: Optional[Union[Tuple[float], str]] = '1'
    num_workers: Optional[int] = 8
    format_train: Optional[Union[Tuple[str], str]] = 'reverse'
    format_eval: Optional[Union[Tuple[str], str]] = 'reverse'
    add_special_tokens: bool = True
    show_task_ids: bool = True
    disjoint_tokens: bool = False
    padding_side: str = 'right'
    use_train_attention_mask: bool = True
    train_pad_to: Optional[int] = None
    eval_pad_to: Optional[int] = None
    mixture_scheduling_kwargs: Optional[Union[dict, str]] = field(default_factory=dict)
    lock_operand_length: bool = False
    save_self_improve_data: bool = True
    
    self_improve_cache_loc: str = 'self_improve_data'
    num_self_improve_data: int = 10000
    additional_train_data: List[str] = field(default_factory=list)
    additional_train_data_max_digit: Optional[int] = None # max digit for additional training data
    fast_self_improve: bool = False
    self_improve_ood_threshold: Optional[float] = None
    sanity_check_self_improve: bool = False
    add_noise: Optional[float] = None
    noise_type: Optional[str] = None
    add_noise_at: Optional[int] = None
    filter_self_improve: Optional[str] = None
    noisy_difficulty: Optional[float] = None
    # extra_SI_difficulty: Optional[int] = None
    lock_operand_length_SI: bool = True
    dynamic_eval_range: Optional[List[int]] = None
    dynamic_eval_range_a: Optional[List[int]] = None
    dynamic_eval_range_b: Optional[List[int]] = None

    skip_self_improve_generation: Optional[bool] = False # if True, skip self-improve data generation
    only_generate_data: Optional[bool] = False # if True, only generate data and exit
    
    filter_shorter_additional_data: Optional[bool] = False # if True, filter out additional data with shorter result
    filter_shorter_additional_data_threshold: Optional[int] = 1 # threshold for filtering out additional data with shorter result
    additional_train_data_digit: Optional[Union[Tuple[int], str]] = None # ex. '5,6', number of digits for additional training data

    majority_voting: Optional[bool] = False # if True, use majority voting for additional training data
    majority_voting_threshold: Optional[float] = 0.5
    majority_vote_seeds: Optional[Union[Tuple[int], str]] = '43' # seed index for models participating in majority voting ex. '41,42,43,44,45,46'
    majority_vote_keep_seed: Optional[bool] = False # if True, keep the seed index for majority voting (used for generating multiple mv data for each seed)


    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.lora_config is not None: 
            self.lora_config = json.loads(self.lora_config) if isinstance(self.lora_config, str) else self.lora_config
    #     if self.format.startswith("{"):
    #         self.format = json.loads(self.format)
        self.op_dist_train = tuple(map(lambda x: tuple(map(float, x.split(','))), self.op_dist_train.split(' ')))
        self.op_train = tuple(self.op_train.split(' '))
        self.format_train = tuple(self.format_train.split(' '))
        if self.n_digits_train is not None:
            self.n_digits_train = self.n_digits_a_train = self.n_digits_b_train = tuple(map(lambda x: tuple(map(int, x.split(','))), self.n_digits_train.split(' ')))
        else:
            assert self.n_digits_b_train is not None and self.n_digits_a_train is not None
            self.n_digits_a_train = tuple(map(lambda x: tuple(map(int, x.split(','))), self.n_digits_a_train.split(' ')))
            self.n_digits_b_train = tuple(map(lambda x: tuple(map(int, x.split(','))), self.n_digits_b_train.split(' ')))
        self.num_train = tuple(map(int, self.num_train.split(' ')))
        if len(self.num_train) == 1:
            self.num_train = self.num_train * len(self.op_train)
        # assert len(self.num_train) == len(self.op_train) == len(self.op_dist_train[0]) == len(self.format_train) == len(self.n_digits_a_train) == len(self.n_digits_b_train), 'You must provide the same number of values for num_train, op_train, op_dist_train, format_train, and n_digits_train'
        self.op_eval = tuple(self.op_eval.split(' '))
        self.op_dist_eval = tuple(map(float, self.op_dist_eval.split(' ')))
        self.format_eval = tuple(self.format_eval.split(' '))
        if self.n_digits_eval is not None:
            self.n_digits_eval = self.n_digits_a_eval = self.n_digits_b_eval = tuple(map(int, self.n_digits_eval.split(',')))
        else:
            assert self.n_digits_a_eval is not None and self.n_digits_b_eval is not None or self.dynamic_eval_range_a is not None and self.dynamic_eval_range_b is not None or self.dynamic_eval_range is not None
            if self.n_digits_a_eval is not None:
                self.n_digits_a_eval = tuple(map(int, self.n_digits_a_eval.split(',')))
            if self.n_digits_b_eval is not None:
                self.n_digits_b_eval = tuple(map(int, self.n_digits_b_eval.split(',')))
        # assert len(self.op_eval) == len(self.op_dist_eval) == len(self.format_eval) == len(self.n_digits_eval), 'You must provide the same number of values for op_eval, op_dist_eval, format_eval, and n_digits_eval'
        self.mixture_scheduling_kwargs = json.loads(self.mixture_scheduling_kwargs) if isinstance(self.mixture_scheduling_kwargs, str) else self.mixture_scheduling_kwargs

        if self.dynamic_eval_range is not None: 
            self.dynamic_eval_range_a = self.dynamic_eval_range_b = self.dynamic_eval_range
        
        if self.additional_train_data_digit is not None:
            self.additional_train_data_digit = tuple(map(lambda x: tuple(map(int, x.split(','))), self.additional_train_data_digit.split(' ')))
            print(self.additional_train_data_digit)
