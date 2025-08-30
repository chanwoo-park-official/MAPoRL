# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import is_deepspeed_available
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import List, Dict

from utils.utils_general import check_initial_weights, zero_and_freeze_adapter, make_grad_work_adapter, make_grad_nowork_adapter
from utils.patch.peft_tuner import apply_patch_peft_tuner
from peft.tuners.tuners_utils import BaseTunerLayer
apply_patch_peft_tuner()
from transformers.trainer import TrainerCallback
from transformers.trainer_utils import has_length

from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig


if is_peft_available():
    from peft import LoraConfig, PeftConfig


if is_deepspeed_available():
    import deepspeed

import requests
import torch
import json
TEMPERATURE_DICT = {
    "microsoft/Phi-3-mini-128k-instruct" : 0.7,
    "meta-llama/Meta-Llama-3-8B-Instruct" : 0.7,
    'meta-llama/Llama-3.1-8B-Instruct': 1,
    "meta-llama/Llama-3.2-3B": 1,
    "Qwen/Qwen2.5-3B-Instruct": 0.7,
    "gen_answer": 0.7
}

def set_adapter_for_models_ppo(main_model, other_models=None, adapter_name=None, diff_model_training=False):
    main_model.policy.set_adapter(adapter_name)
    assert sorted(main_model.policy.active_adapters) == [adapter_name]
    
    if diff_model_training and other_models is not None:
        for other_model in other_models:
            other_model.policy.set_adapter(adapter_name)
            assert sorted(other_model.policy.active_adapters) == [adapter_name]


def create_optimizer(learning_rate,params):
    # Common optimizer setup for PyTorch
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,  # You can adjust the learning rate as needed
    )
    return optimizer

def create_scheduler(optimizer, num_training_steps):
    # Using PyTorch's built-in scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps
    )
    return scheduler


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class SyncRefModelCallback(TrainerCallback):
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, torch.nn.Module],
        accelerator: Optional[Accelerator],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    SyncRefModelCallback._sync_target_model(model, target_model, alpha)
        else:
            SyncRefModelCallback._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch




@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question} ### Answer: {answer}"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
            append_concat_token ('bool', *optional*, defaults to True)
                If true, appends `eos_token_id` at the end of each sample being packed.
            add_special_tokens ('bool', *optional*, defaults to True)
                If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
    

class RunningMoments:
    def __init__(self, accelerator):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24
        self.accelerator = accelerator

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


@torch.no_grad()
def get_global_statistics(accelerator, xs: torch.Tensor, mask=None, device="cpu") -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.to(device)


def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


# copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/stat_tracking.py#L5
class PerPromptStatTracker:
    r"""
    Class for tracking statistics per prompt. Mainly used to calculate advantage for the DPPO algorithm

    Args:
        buffer_size (`int`):
            Size of the buffer to keep for each prompt.
        min_count (`int`):
            Minimum number of samples to keep in the buffer before calculating the mean and std.
    """

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in self.stats.items()}


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs=None):
    if is_unsloth_available():
        # Unsloth adds a new attribute in the model config `unsloth_version`
        # to keep track of models that have been patched with unsloth.
        if hasattr(model, "config") and getattr(model.config, "unsloth_version", None) is not None:
            tag_names.append("unsloth")

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names
    return kwargs


def get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
        )
    elif model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    if is_xpu_available():
        return {"": f"xpu:{PartialState().local_process_index}"}
    elif torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_peft_config(model_config: ModelConfig) -> "Optional[PeftConfig]":
    if model_config.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=model_config.lora_task_type,
        target_modules=model_config.lora_target_modules,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


class RichProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation using Rich.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

        self.training_task_id = None
        self.prediction_task_id = None

        self.rich_group = None
        self.rich_console = None

        self.training_status = None
        self.current_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = Progress()
            self.prediction_bar = Progress()

            self.rich_console = Console()

            self.training_status = self.rich_console.status("Nothing to log yet ...")

            self.rich_group = Live(Panel(Group(self.training_bar, self.prediction_bar, self.training_status)))
            self.rich_group.start()

            self.training_task_id = self.training_bar.add_task("[blue]Training the model", total=state.max_steps)
            self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(self.training_task_id, advance=state.global_step - self.current_step, update=True)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_task_id is None:
                self.prediction_task_id = self.prediction_bar.add_task(
                    "[blue]Predicting on the evaluation dataset", total=len(eval_dataloader)
                )
            self.prediction_bar.update(self.prediction_task_id, advance=1, update=True)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_task_id is not None:
                self.prediction_bar.remove_task(self.prediction_task_id)
                self.prediction_task_id = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_task_id is not None:
                self.prediction_bar.remove_task(self.prediction_task_id)
                self.prediction_task_id = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_status.update(f"[bold green]Status = {str(logs)}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.rich_group.stop()

            self.training_bar = None
            self.prediction_bar = None
            self.training_task_id = None
            self.prediction_task_id = None
            self.rich_group = None
            self.rich_console = None
            self.training_status = None
            self.current_step = None


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


SIMPLE_SFT_CHAT_TEMPLATE = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
# SIMPLE_SFT_CHAT_TEMPLATE simply ends things with an EOS token, this helps the SFT model learn to end the completions with EOS tokens

SIMPLE_QUERY_CHAT_TEMPLATE = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}"
# SIMPLE_QUERY_CHAT_TEMPLATE is a variant of SIMPLE_SFT_CHAT_TEMPLATE, which does not end the content with EOS token. The idea
# is to have the generated response to end with an EOS token, but the query itself should not end with EOS tokens.


@dataclass
class OnpolicyRuntimeConfig:
    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""


def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True
            in each row. If no True value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(
    model: torch.nn.Module,ref:bool,  turn: int, query_responses: torch.Tensor, pad_token_id: int, task_training: bool
):
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    model.policy.set_adapter(f"value_{turn}")

    assert sorted(model.policy.active_adapters) == [f"value_{turn}"]
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model.policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )

    # reward_logits = model.score(output.hidden_states[-1])
    value_head = getattr(model, f"value_heads_{turn}")
    reward_logits = value_head(output.hidden_states[-1])
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits 

def logit_from_different_server(query_responses, context_length, pad_token_id, ip):
    # Convert the tensor to a list for JSON serialization
    query_responses_list = query_responses.tolist()
    json_post = {'query_response': query_responses_list, 'context_length': context_length, 'pad_token_id': pad_token_id}
        
    # Send the POST request to the server
    try:
        response = requests.post(ip, json=json_post)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert the result back to a tensor
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        return None

    return torch.tensor(result['result'])


def forward(
    model: torch.nn.Module,
    ref,
    turn,
    query_responses: torch.Tensor,
    pad_token_id: int,
    round_num: int,
    policy_separation: bool, 
    collaboration_separation: bool,
    task_training: bool
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)



    # original code
    # if turn == None:
    #     model.set_adapter(f"policy")
    #     return model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         return_dict=True,
    #         output_hidden_states=True,
    #     )


    if ref == False:
        if policy_separation and not collaboration_separation:
            model.set_adapter(f"policy_{turn}")    
        elif policy_separation and collaboration_separation:
            if turn == 0:
                model.set_adapter(f"policy")
            else:
                model.set_adapter(f"col")
        else:
            model.set_adapter(f"policy")
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )
    else:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )


def prepare_deepspeed(model: torch.nn.Module, per_device_train_batch_size: int):
    """
    Prepares the model for training with DeepSpeed (both for stage 2 and 3), configuring the appropriate settings based on the model and
    batch size.

    Args:
        model (`torch.nn.Module`):
            The model to be prepared for DeepSpeed training.
        per_device_train_batch_size (`int`):
            The training batch size per device.

    Returns:
        `torch.nn.Module`:
            The model initialized and configured with DeepSpeed for training.
    """
    import deepspeed

    deepspeed_plugin = AcceleratorState().deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
        config_kwargs = {
            "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
    else:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

def truncate_gemma_repetitive(responses: torch.Tensor, pad_token_id: int):
    target_tokens = torch.tensor([235248, 108, 111], device=responses.device)
    batch_size = responses.shape[0]
    seq_length = responses.shape[1]
    
    # Initialize truncation indices to sequence length (no truncation by default)
    trunc_idxs = torch.full((batch_size,), seq_length, device=responses.device)
    
    # Process each sequence in the batch independently
    for batch_idx in range(batch_size):
        sequence = responses[batch_idx]
        
        # For each position in this sequence
        for i in range(seq_length - 3):
            window = sequence[i:i+4]
            
            # Count matches in window
            window_mask = torch.zeros_like(window, dtype=torch.bool)
            for token in target_tokens:
                window_mask |= (window == token)
            
            # If we find 4 consecutive matches
            if window_mask.sum() >= 4:
                trunc_idxs[batch_idx] = i  # truncate at start of pattern
                break  # Move to next sequence in batch
    
    # Create mask for each position in each sequence
    idxs = torch.arange(seq_length, device=responses.device)
    mask = idxs >= trunc_idxs.unsqueeze(1)
    
    # Apply truncation with padding
    postprocessed_responses = responses.clone()
    postprocessed_responses[mask] = pad_token_id
    
    return postprocessed_responses
def truncate_response(stop_token_ids, pad_token_id: int, responses: torch.Tensor):
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    
    # Create a mask for all stop tokens
    stop_mask = torch.zeros_like(responses, dtype=torch.bool)
    for stop_token_id in stop_token_ids:
        stop_mask |= (responses == stop_token_id)

    # Find the first true index for each row
    trunc_idxs = first_true_indices(stop_mask).unsqueeze(-1)

    # Truncate and pad
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re


def extract_score(text):
    match = re.search(r'(\d\.\d{3}) out of 1.0', text)  # Look for 3 decimal places
    if match:
        return float(match.group(1))  # Extract the floating point number
    return 0.0  # Return 0.0 if no score is found

# Function to trim the question, removing the part starting from 'Please determine...'
def trim_question(question):
    if 'Please determine' in question:
        return question.split('Please determine')[0].strip()
    return question

def visualize_chat_in_columns(chat_data, idx, epoch, dataset_name):
    num_agents = len(chat_data)
    plt.rc('text', usetex=False)
    fig, ax = plt.subplots(figsize=(40, 7))  # Adjust figure size for better visibility
    ax.axis('off')

    column_width = 1 / num_agents
    color_thresholds = [(0.9, 'green'), (0.75, 'yellow'), (0.5, 'orange'), (0, 'red')]

    def get_color(score):
        for threshold, color in color_thresholds:
            if score >= threshold:
                return color
        return 'red'

    question = trim_question(chat_data[0][0]['content']).replace('$', '')  # Remove dollar signs
    question_lines = '\n'.join(question[i:i + 400] for i in range(0, len(question), 400))  # Multi-line title

    for agent_idx, agent_data in enumerate(chat_data):
        x_pos = agent_idx * column_width
        y_start = 1

        # Add Agent title
        ax.text(x_pos + 0.02, y_start, f"Agent {agent_idx}", fontsize=14, va='top', wrap=True)
        y_start -= 0.1  # Space below agent title

        # Iterate over the agent's content and display in column format
        for i in range(1, len(agent_data), 3):  # Start at 1, skip by 3 to get the chat (3i + 1)
            content = agent_data[i]['content'].replace('$', '')  # Clean redundant content and remove dollar signs
            
            score_text = agent_data[i+1]['content'].replace('$', '')  # Remove dollar signs from score text as well
            score = extract_score(score_text)
            color = get_color(score)
            
            # Split long content into lines
            content_lines = '\n'.join(content[j:j + 300] for j in range(0, len(content), 300))
            
            # Content
            ax.text(x_pos + 0.02, y_start, content_lines, fontsize=5, va='top', wrap=True)
            
            # Score color block next to content
            ax.add_patch(Rectangle((x_pos + 0.47, y_start - 0.2), 0.03, 0.03, color=color))
            ax.text(x_pos + 0.47, y_start - 0.18, f"{score:.3f}", fontsize=10, va='top')

            y_start -= 0.3  # More spacing between rows for better clarity

    # Set multi-line question as the title
    plt.suptitle(f"Question: {question_lines}", fontsize=12, va='top')
    plt.subplots_adjust(top=0.85)  # Adjust top margin for title space

    # Create directory if it doesn't exist
    if not os.path.exists(f"visualization_{dataset_name}"):
        os.makedirs(f"visualization_{dataset_name}")

    # Save the figure
    plt.savefig(f"visualization_{dataset_name}/{idx}_{epoch}.png")
    plt.close()



def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`dict`):
            The configuration dictionary for generation settings.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits

def generate_only_response(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`dict`):
            The configuration dictionary for generation settings.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        generation_config=generation_config,
        return_dict_in_generate = False,
        output_scores=False,
    )
    return output[:, context_length:]

def graph_gen(output_dir):

# Load the JSON data
    with open(f'{output_dir}/metrics.json', 'r') as file:
        data = json.load(file)

# Convert the JSON data into a structured format
    eps = [entry['eps'] for entry in data]
    objective_kl = [entry['objective/kl'] for entry in data]
    objective_entropy = [entry['objective/entropy'] for entry in data]
    objective_non_score_reward = [entry['objective/non_score_reward'] for entry in data]
    objective_rlhf_reward = [entry['objective/rlhf_reward'] for entry in data]
    objective_scores = [entry['objective/scores'] for entry in data]
    policy_approxkl_avg = [entry['policy/approxkl_avg'] for entry in data]
    policy_clipfrac_avg = [entry['policy/clipfrac_avg'] for entry in data]
    loss_policy_avg = [entry['loss/policy_avg'] for entry in data]
    loss_value_avg = [entry['loss/value_avg'] for entry in data]
    val_clipfrac_avg = [entry['val/clipfrac_avg'] for entry in data]
    policy_entropy_avg = [entry['policy/entropy_avg'] for entry in data]
    val_ratio = [entry['val/ratio'] for entry in data]
    val_ratio_var = [entry['val/ratio_var'] for entry in data]
    val_num_eos_tokens = [entry['val/num_eos_tokens'] for entry in data]
    lr = [entry['lr'] for entry in data]
    episode = [entry['episode'] for entry in data]
    update = [entry['update'] for entry in data]

    # Plotting the graphs
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 5, 1)
    plt.plot(update, objective_kl, label='objective/kl')
    plt.xlabel('update')
    plt.ylabel('objective/kl')
    plt.legend()

    plt.subplot(3, 5, 2)
    plt.plot(update, objective_entropy, label='objective/entropy')
    plt.xlabel('update')
    plt.ylabel('objective/entropy')
    plt.legend()

    plt.subplot(3, 5, 3)
    plt.plot(update, objective_non_score_reward, label='objective/non_score_reward')
    plt.xlabel('update')
    plt.ylabel('objective/non_score_reward')
    plt.legend()

    plt.subplot(3, 5, 4)
    plt.plot(update, objective_rlhf_reward, label='objective/rlhf_reward')
    plt.xlabel('update')
    plt.ylabel('objective/rlhf_reward')
    plt.legend()

    plt.subplot(3, 5, 5)
    plt.plot(update, objective_scores, label='objective/scores')
    plt.xlabel('update')
    plt.ylabel('objective/scores')
    plt.legend()

    plt.subplot(3, 5, 6)
    plt.plot(update, policy_approxkl_avg, label='policy/approxkl_avg')
    plt.xlabel('update')
    plt.ylabel('policy/approxkl_avg')
    plt.legend()

    plt.subplot(3, 5, 7)
    plt.plot(update, policy_clipfrac_avg, label='policy/clipfrac_avg')
    plt.xlabel('update')
    plt.ylabel('policy/clipfrac_avg')
    plt.legend()

    plt.subplot(3, 5, 8)
    plt.plot(update, loss_policy_avg, label='loss/policy_avg')
    plt.xlabel('update')
    plt.ylabel('loss/policy_avg')
    plt.legend()

    plt.subplot(3, 5, 9)
    plt.plot(update, loss_value_avg, label='loss/value_avg')
    plt.xlabel('update')
    plt.ylabel('loss/value_avg')
    plt.legend()

    plt.subplot(3, 5, 10)
    plt.plot(update, val_clipfrac_avg, label='val/clipfrac_avg')
    plt.xlabel('update')
    plt.ylabel('val/clipfrac_avg')
    plt.legend()

    plt.subplot(3, 5, 11)
    plt.plot(update, policy_entropy_avg, label='policy/entropy_avg')
    plt.xlabel('update')
    plt.ylabel('policy/entropy_avg')
    plt.legend()

    plt.subplot(3, 5, 12)
    plt.plot(update, val_ratio, label='val/ratio')
    plt.xlabel('update')
    plt.ylabel('val/ratio')
    plt.legend()

    plt.subplot(3, 5, 13)
    plt.plot(update, val_ratio_var, label='val/ratio_var')
    plt.xlabel('update')
    plt.ylabel('val/ratio_var')
    plt.legend()

    plt.subplot(3, 5, 14)
    plt.plot(update, val_num_eos_tokens, label='val/num_eos_tokens')
    plt.xlabel('update')
    plt.ylabel('val/num_eos_tokens')
    plt.legend()

    plt.subplot(3, 5, 15)
    plt.plot(update, lr, label='lr')
    plt.xlabel('update')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig(f'{output_dir}/graph.png')

def graph_gen_turn_specific(output_dir):
    # Load the JSON data
    with open(f'{output_dir}/metrics.json', 'r') as file:
        data = json.load(file)

    # Extract updates and turns
    updates = [entry['update'] for entry in data]
    turns = sorted(set(k.split('_')[-1] for k in data[0].keys() if 'turn_' in k))

    # Initialize dictionaries to store metrics
    metrics = {key.split('_turn_')[0]: {turn: [] for turn in turns} for key in data[0] if 'turn_' in key}

    # Populate the metrics dictionary with data
    for entry in data:
        for key in metrics:
            for turn in turns:
                metric_key = f'{key}_turn_{turn}'
                if metric_key in entry:
                    metrics[key][turn].append(entry[metric_key])

    # Separate graphs for specific metrics
    specific_metrics = ['objective/rlhf_reward', 'objective/scores', 'loss/value_avg', 'loss/policy_avg']

    # Calculate the number of subplots required
    num_subplots = len(specific_metrics) * len(turns)
    cols = len(turns)  # Number of columns of subplots
    rows = len(specific_metrics)  # Calculate rows needed

    # Set a fixed width and height for each subplot
    subplot_width = 5
    subplot_height = 4
    figsize = (cols * subplot_width, rows * subplot_height)

    # Plotting the graphs
    plt.figure(figsize=figsize)

    for i, metric_base in enumerate(specific_metrics):
        for j, turn in enumerate(turns):
            plt.subplot(rows, cols, i * cols + j + 1)
            plt.plot(updates, metrics[metric_base][turn], label=f'{metric_base}_turn_{turn}')
            plt.xlabel('Update')
            plt.ylabel(metric_base.split('/')[1])
            plt.title(f'{metric_base} Turn {turn}')
            plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/specific_metrics.png')
    plt.show()


def graph_gen_turn(output_dir):
    # Load the JSON data
    with open(f'{output_dir}/metrics.json', 'r') as file:
        data = json.load(file)

    # Extract updates and turns
    updates = [entry['update'] for entry in data]
    turns = set(k.split('_')[-1] for k in data[0].keys() if 'turn_' in k)

    # Initialize dictionaries to store metrics
    metrics = {key.split('_turn_')[0]: {turn: [] for turn in turns} for key in data[0] if 'turn_' in key}

    # Populate the metrics dictionary with data
    for entry in data:
        for key in metrics:
            for turn in turns:
                metric_key = f'{key}_turn_{turn}'
                if metric_key in entry:
                    metrics[key][turn].append(entry[metric_key])

    # Check if metrics are populated correctly

    # Calculate the number of subplots required
    num_subplots = len(metrics)
    cols = 4  # Number of columns of subplots
    rows = (num_subplots + cols - 1) // cols  # Calculate rows needed

    # Set a fixed width and height for each subplot
    subplot_width = 5
    subplot_height = 4
    figsize = (cols * subplot_width, rows * subplot_height)

    # Plotting the graphs
    plt.figure(figsize=figsize)

    for i, (metric, turn_data) in enumerate(metrics.items()):
        plt.subplot(rows, cols, i + 1)
        for turn, values in turn_data.items():
            if len(updates) != len(values):
                continue
            plt.plot(updates, values, label=f'{metric}_turn_{turn}')
        plt.xlabel('update')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/graph.png')









def construct_message_multi_agent_summary(summary, question, reward_feedback=False):
    prefix_string = "These are the solutions's summary to the problem from other agents: "
    response = "\n\n Summary of other agents' solution: ```{}```".format(summary)
    prefix_string = prefix_string + response
    if reward_feedback:
        prefix_string = prefix_string + (
            f"Here, each reward means the possibility that each suggested answer would be correct. Consider this reward as well, "
            f"but keep in mind that it might not be accurate. The reward is between 0 and 1, and if it is close to 1, the corresponding answer is more likely to be true, and if it is close to 0, the corresponding asnwer is more likely to be false.\n\n"
            f"Carefully think about whether your previous solution was correct, and if you believe it was incorrect, feel free to change your opinion. "
            f"Provide a concise solution with reason to your answer, including any key steps or thought processes that led to your conclusion. "
            f"Justify why your final choice is the most appropriate, and ensure that your reasoning supports this choice. Once again, the question is: {question}"
        )
    else:
        prefix_string = prefix_string + ( 
            f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
            f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
            f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
            f"Once again, the question is: {question}"
        )
    return {"role": "user", "content": prefix_string}    




def bonus_rule(correctnesses_turn_agent: dict, total_rounds: int, total_agents: int, finished_question: list, turn: int, agent: int, incentive: bool, device, alpha: List, correct_threshold = 0.5, wrong_threshold = 0.5):
    correctnesses = correctnesses_turn_agent[(turn, agent)]
    def get_score_for_turn(q, t, a):
        if finished_question[q] != -1 and t > finished_question[q]:
            return torch.tensor(-1.0)
        else:
            actual_turn = t
            # Find the correct index corresponding to q
            valid_indices = [idx for idx, f in enumerate(finished_question) if f == -1 or f >= actual_turn]
            q_index = valid_indices.index(q)  # find the index of q in the valid indices
            return correctnesses_turn_agent[(actual_turn, a)][q_index]

   
    def apply_incentive(scores_list, turn, agent, alpha):
        scores_new = [torch.tensor(-1.0) if x == -1.0 else torch.tensor(0.0) for x in scores_list]
        scores_new = torch.stack(scores_new)
        if turn == 0:
            return scores_new
        # Modify scores_list based on incentive rules
        if turn > 0:
            for q in range(len(scores_list)):
                prev_score = get_score_for_turn(q, turn - 1, agent)
                if total_agents >1: 
                    prev_others_score = torch.sum(
                    torch.stack([get_score_for_turn(q, turn - 1, a) for a in range(total_agents) if a != agent], dim=0)
                    ) / (total_agents - 1)
                current_score = get_score_for_turn(q, turn, agent)
                if prev_score > correct_threshold:
                    if current_score < wrong_threshold:
                        if total_agents > 1:
                            if prev_others_score > correct_threshold:
                                scores_new[q] -= alpha[1] # since they generated a new answer wrong
                                if scores_new[q] == -1:
                                    scores_new[q] += 0.001

                            elif prev_others_score < wrong_threshold:
                                scores_new[q] -= alpha[0] # since they got persuaded by others wrongly
                                if scores_new[q] == -1:
                                    scores_new[q] += 0.001

                        else:
                            scores_new[q] -= alpha[1]  # since they generated a new answer wrong
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001

                if prev_score < wrong_threshold:
                    if current_score > correct_threshold:
                        if total_agents > 1:
                            if prev_others_score < wrong_threshold:
                                scores_new[q] += alpha[1]
                                if scores_new[q] == -1:
                                    scores_new[q] += 0.001


                            elif prev_others_score > correct_threshold:
                                scores_new[q] += alpha[0]
                                if scores_new[q] == -1:
                                    scores_new[q] += 0.001

                        else:
                            scores_new[q] += alpha[1]
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001
        if turn < total_rounds - 1 and total_agents > 1:
            for q in range(len(scores_list)):
                next_others_score = torch.sum(
                torch.stack([get_score_for_turn(q, turn + 1, a) for a in range(total_agents) if a != agent], dim=0)
                ) / (total_agents - 1)
                current_score = get_score_for_turn(q, turn, agent)
                current_others_score = torch.sum(
                torch.stack([get_score_for_turn(q, turn, a) for a in range(total_agents) if a != agent], dim=0)
                ) / (total_agents - 1)
                                
                if next_others_score > correct_threshold:
                    if current_others_score < wrong_threshold:
                        if current_score > correct_threshold:
                            scores_new[q] -= alpha[3] 
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001 
                        elif current_score < wrong_threshold:
                            scores_new[q] -= alpha[2] 
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001
                if next_others_score < wrong_threshold:
                    if current_others_score > correct_threshold:
                        if current_score < wrong_threshold:
                            scores_new[q] += alpha[3] 
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001
                        elif current_score > correct_threshold:
                            scores_new[q] += alpha[2] 
                            if scores_new[q] == -1:
                                scores_new[q] += 0.001
        
                           

        return scores_new        

    scores = torch.stack([get_score_for_turn(q, turn, agent) for q in range(len(finished_question))])
    scores_list = apply_incentive(scores, turn, agent, alpha)
    return scores_list

    
def score_rule(rule_horizon: str, rule_agent_share: str, discount_factor: float, scores_turn_agent: dict, total_rounds: int, total_agents: int, finished_question: list, turn: int, agent: int, incentive: bool, device, alpha: List, correct_threshold = 0.7, wrong_threshold = 0.3):
    scores = scores_turn_agent[(turn, agent)]
    def get_score_for_turn(q, t, a):
        if finished_question[q] != -1 and t > finished_question[q]:
            return torch.tensor(-1.0)
        else:
            actual_turn = t
            # Find the correct index corresponding to q
            valid_indices = [idx for idx, f in enumerate(finished_question) if f == -1 or f >= actual_turn]
            q_index = valid_indices.index(q)  # find the index of q in the valid indices
            return scores_turn_agent[(actual_turn, a)][q_index]
    
    if (rule_horizon == "last") & (rule_agent_share == "all"):
        scores = torch.stack([
            torch.sum(torch.stack([get_score_for_turn(q, total_rounds - 1, a) for a in range(total_agents)], dim=0)) / total_agents
            for q in range(len(finished_question))
        ])

    elif (rule_horizon == "last") & (rule_agent_share == "individual"):
        scores = torch.stack([
            get_score_for_turn(q, total_rounds - 1, agent)
            for q in range(len(finished_question))
        ])

    elif (rule_horizon == "discounted_sum"):
        scores_list = []
        for q in range(len(finished_question)):
            x = total_rounds - 1 if finished_question[q] == -1 else finished_question[q]
            if turn > x:
                scores_list.append(torch.tensor(-1.0))
            else:
                # Calculate the discounted sum for current action and future impact
                discounted_factors = [discount_factor ** (t_prime - turn) for t_prime in range(turn, x + 1)]
                discount_factor_sum = torch.sum(torch.tensor(discounted_factors))
                
                if rule_agent_share == "all":
                    # print("line 1978, , turn == ", turn)
                    # Influence-aware verification reward
                    verifier_current = get_score_for_turn(q, turn, agent)
                    # print("line 1981 stack", torch.stack([
                    #     torch.sum(torch.stack([
                    #         discount_factor ** (t_prime - turn) * get_score_for_turn(q, t_prime, a)
                    #         for a in range(total_agents)
                    #     ])) / total_agents
                    #     for t_prime in range(turn + 1, x + 1)
                    # ]))
                    if turn!= total_rounds - 1:
                        future_impact_sum = torch.sum(torch.stack([
                            torch.sum(torch.stack([
                                discount_factor ** (t_prime - turn) * get_score_for_turn(q, t_prime, a)
                                for a in range(total_agents)
                            ])) / total_agents
                            for t_prime in range(turn + 1, x + 1)
                        ]))
                    else:
                        future_impact_sum = 0
                    
                    # Final score calculation
                    total_score = verifier_current + future_impact_sum
                    total_score /= discount_factor_sum
                    scores_list.append(total_score)
                elif (rule_agent_share == "individual"):
                    score_sum = torch.sum(torch.stack([
                        get_score_for_turn(q, t, agent) * discount_factor ** (t - turn)
                        for t in range(turn, x + 1)
                    ], dim=0))
                    score = score_sum / discount_factor_sum
                    scores_list.append(score)

        
        scores = torch.stack(scores_list)

    elif (rule_horizon == "current") & (rule_agent_share == "individual"):
        scores = torch.stack([get_score_for_turn(q, turn, agent) for q in range(len(finished_question))])

    elif (rule_horizon == "current") & (rule_agent_share == "all"):
        scores = torch.stack([
            torch.sum(torch.stack([get_score_for_turn(q, turn, a) for a in range(total_agents)], dim=0)) / total_agents
            for q in range(len(finished_question))
        ])

    else:
        raise ValueError("Rule not supported")

    return scores










                    
                    
