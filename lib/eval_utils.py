import os
from typing import cast
from transformers import EvalPrediction, PreTrainedTokenizer, TrainerCallback, Trainer
from transformers.integrations import WandbCallback
from transformers.generation.utils import GenerateDecoderOnlyOutput
import Levenshtein
import numpy as np
import torch

from lib.configs import MyTrainingArguments

def process_generate_output(generate_output: tuple, labels):
    # we have to preprocess the logits because they live on the GPU until the entire evaluation is ran, so it gets too big
    logits = generate_output[1].permute(1, 0, 2)  # (batch_size, seq_len, vocab_size)
    # probs = torch.softmax(logits, dim=-1)
    # torch.save({'probs': probs, 'labels': labels, 'pred': pred}, f"round-60-probs-{labels.shape[-1]}.pt")
    pertoken_entropy = torch.distributions.Categorical(logits=logits).entropy()

    return (generate_output[0], pertoken_entropy)

def np_pad(arrays, pad_value=-100, max_len=None, side='left'):
    if max_len is None:
        max_len = max(arr.shape[1] for arr in arrays)
    padded = np.vstack([
        np.pad(arr, ((0, 0), (max_len - arr.shape[1], 0)), constant_values=pad_value) if side == 'left' else
        np.pad(arr, ((0, 0), (0, max_len - arr.shape[1])), constant_values=pad_value)
        for arr in arrays
    ])
    return padded, max_len

def compute_metrics(tokenizer: PreTrainedTokenizer, pred_obj: EvalPrediction, args: MyTrainingArguments):
    if isinstance(pred_obj.predictions, tuple):
        pred_sequences, pertoken_entropy = pred_obj.predictions
    else:
        pred_sequences = pred_obj.predictions

    labels, labels_max_len = np_pad(pred_obj.label_ids, pad_value=tokenizer.pad_token_id, side='right')
    inputs, _ = np_pad(pred_obj.inputs, pad_value=tokenizer.pad_token_id, side='left')
    pred, _ = np_pad(pred_sequences, pad_value=tokenizer.pad_token_id, side='right', max_len=labels_max_len)

    # Label and pred can still have unequal lengths, for example when the model consistently predicts short
    # max_len = max(pred.shape[1], labels.shape[1])
    # pred = np.pad(pred, ((0, 0), (0, max_len - pred.shape[1])), constant_values=tokenizer.pad_token_id)
    # labels = np.pad(labels, ((0, 0), (0, max_len - labels.shape[1])), constant_values=tokenizer.pad_token_id)
    # pred[pred == -100] = tokenizer.pad_token_id
    # labels[labels == -100] = tokenizer.pad_token_id
    # assert (pred >= 0).all() and (labels >= 0).all()

    if args.do_backtrack_decoding:
        # we need to clean the backtrack tokens
        backtrack_token_id = tokenizer.backtrack_token_id
        cleaned_pred = np.zeros_like(pred)
        for bi, p in enumerate(pred):
            delete_mask = p == backtrack_token_id
            delete_mask[:-1] |= delete_mask[1:]
            p = p[~delete_mask]
            cleaned_pred[bi, :len(p)] = p
        pred = cleaned_pred[:, :labels.shape[1]]

    correct = (pred == labels).all(axis=1)
    accuracy = correct.mean()
    distance = sum([Levenshtein.ratio(pred[bi].tolist(), labels[bi].tolist()) for bi in range(pred.shape[0])]) / pred.shape[0]
    length_acc = ((pred != tokenizer.pad_token_id).sum(axis=1) == (labels != tokenizer.pad_token_id).sum(axis=1)).mean()
    
    extra_metrics = {}

    if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
        prompt_str = tokenizer.batch_decode(inputs[:5])
        pred_str = tokenizer.batch_decode(pred[:5])
        label_str = tokenizer.batch_decode(labels[:5])
        for pr, p, l in zip(prompt_str, pred_str, label_str):
            print("="*80)
            print(f"Prompt: {repr(pr)}")
            print(f"Pred  : {repr(p)}")
            print(f"Label : {repr(l)}")

    
    if args.eval_maze:
        from .data_formats_maze import parse_graph_to_dict, evaluate_path

        prompt_str = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        format_correct_list, move_correct_list, end_correct_list = [], [], []

        # Parse the graph
        for i, p in enumerate(pred_str):
            graph_dict, start_node, end_node = parse_graph_to_dict(prompt_str[i])
            format_correct, move_correct, end_correct = evaluate_path(graph_dict, p, start_node, end_node)
            format_correct_list.append(format_correct)
            move_correct_list.append(move_correct)
            end_correct_list.append(end_correct)

        extra_metrics = {'format_correct': np.mean(format_correct_list), 'move_correct': np.mean(move_correct_list), 'end_correct': np.mean(end_correct_list)}

    return {'accuracy': accuracy, 'distance': distance, 'input_length': pred_obj.inputs[0].shape[1], 'length_acc': length_acc} | extra_metrics

class WandbEvalCallback(WandbCallback):
    def __init__(self, trainer: Trainer):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        breakpoint()
