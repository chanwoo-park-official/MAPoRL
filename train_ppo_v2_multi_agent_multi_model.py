# Standard library imports
import argparse
import ast
import copy
import io
import logging
import multiprocessing
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model
from transformers import AutoTokenizer

# Local imports
from utils.patch.peft_tuner import apply_patch_peft_tuner
apply_patch_peft_tuner()
from utils.utils_general import set_seed, load_config_from_python, zero_and_freeze_adapter
from utils.utils_cooperateLLM import (
    create_invalid_entry, safe_execute_code,
    MAX_INPUT_LENGTH_MATH, TRAIN_DATASET_SIZE_MATH, TEST_DATASET_SIZE_MATH, DEBUG_DATASET_SIZE
)
from utils.utils_general import (
    print_gpu_memory_usage, extract_last_number_after_slash, find_latest_folder, parse_list,
    is_effectively_integer
)
from utils.utils_model import ModelManager
from trl.trl.trainer.ppov2_config import PPOv2Config
from trl.trl.trainer.ppov2_trainer_multi_different_model import PPOv2Trainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


"""
Multi-Agent PPO Training Script for Cooperative LLM Training.

This module implements a sophisticated multi-agent PPO training
system designed for cooperative language model training scenarios. It supports various training
configurations including policy separation, collaboration separation, and different model
architectures for different agents.

Key Features:
- Multi-agent training with configurable agent numbers
- Policy and value head separation for different training rounds
- Support for different models per agent
- Resume training from checkpoints
- Integration with Weights & Biases for experiment tracking
- Support for GSM8K, ANLI, and FANTOM datasets

Main Components:
- ModelManager: Handles initialization and management of multiple agent models
- Dataset preparation functions for different dataset types
- Utility functions for model loading, adapter management, and training setup

Usage:
    python train_ppo_v2_multi_agent_multi_model.py --config path/to/config.py --alpha "[0.1, 0.2]"

"""





def prepare_dataset(dataset_name: str, tokenizer, split: str, debug: bool = False):
    """pre-tokenize the dataset before training; only collate during training"""
    if dataset_name == "GSM8k":
        dataset = load_dataset("TinyGSM/TinyGSM")["train"]
        if split == "train":
            dataset = dataset.select(range(TRAIN_DATASET_SIZE_MATH))
        elif split == "test":
            dataset = dataset.select(range(TRAIN_DATASET_SIZE_MATH, TRAIN_DATASET_SIZE_MATH + TEST_DATASET_SIZE_MATH ))

        def tokenize(sample):
            try:
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["question"]}],
                    padding=False,
                    add_generation_prompt=True,
                )

                # Check if input_ids length exceeds maximum allowed length
                if len(input_ids) > MAX_INPUT_LENGTH_MATH:
                    return create_invalid_entry(input_ids, tokenizer)

                # Safely execute the mathematical code
                result = safe_execute_code(sample['code'])
                if result == INVALID_ANSWER or not is_effectively_integer(result):
                    return create_invalid_entry(input_ids, tokenizer)

                # Convert to int if it's a float but effectively an integer
                result = int(result) if isinstance(result, float) else result

                return {
                    "input_ids": input_ids,
                    "lengths": len(input_ids),
                    "answer": tokenizer.encode(str(result))
                }
            except Exception:
                return create_invalid_entry(input_ids, tokenizer)

    elif dataset_name == "ANLI":
        dataset = load_dataset('anli', split=f"{split}_r3")
        label = ['entailment', 'neutral', 'contradiction']
        def tokenize(sample):
            input_ids = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": (
                        "Premise: " + sample["premise"] + "\n\n"
                        "Hypothesis: " + sample["hypothesis"] + "\n\n"
                        "Please determine the relationship between the premise and hypothesis. Choose one of the following: 'entailment', 'neutral', or 'contradiction'. "
                        "Start with a concise reasoning for your choice, and conclude with your final answer. You do not need to restate the premise and hypothesis."
                    )
                }],
                padding=False,
                add_generation_prompt=True,
            )

            answer = tokenizer.encode(label[sample['label']])
            return {"input_ids": input_ids, "lengths": len(input_ids), "answer": answer}
    elif dataset_name == "FANTOM":
        dataset = load_dataset('json', data_files=f'{split}_data_FANTOM.json')['train']

        if split == "train":
            dataset_replica = dataset.map(lambda sample: sample.copy())
            dataset = concatenate_datasets([dataset, dataset_replica])

        def tokenize(sample):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Conversation: " + sample["document"] + "\n\n Question: " + sample["question"] + "Answer the given question based on the conversation. When you answer, think carefully. "}],
                padding=False,
                add_generation_prompt=True,
            )
            answer = tokenizer.encode(sample["correct_answer"])
            wrong_answer = tokenizer.encode(sample["wrong_answer"])
            return {"input_ids": input_ids, "lengths": len(input_ids), "answer": answer, "wrong_answer": wrong_answer}


    if debug and split == "train":
        dataset = dataset.select(range(DEBUG_DATASET_SIZE))


    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
        load_from_cache_file=not config.sanity_check,
    )
    dataset = dataset.filter(lambda x: x['answer'] != tokenizer.encode(str(-1)))


    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Agent PPO Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    parser.add_argument('--alpha', type=parse_list, required=True, help='Alpha value for reward shaping')
    parser.add_argument('--reload', action='store_true', help='Whether to reload from a previous run')
    args = parser.parse_args()

    # Validate configuration file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    # Load and validate configuration
    try:
        input_from_config = load_config_from_python(args.config)
        if not isinstance(input_from_config, dict):
            raise ValueError("Configuration file must return a dictionary")
        globals().update(input_from_config)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {args.config}: {e}")
    # Validate alpha parameter
    if not isinstance(alpha, list) or len(alpha) == 0:
        raise ValueError("Alpha must be a non-empty list")
    for a in alpha:
        if not isinstance(a, (int, float)) or a < 0:
            raise ValueError("All alpha values must be non-negative numbers")

    reload = args.reload
    output_dir = output_dir + "_alpha_" + str(alpha)
    
    config = PPOv2Config(
        exp_name="ppo",
        run_name=run_name,
        num_mini_batches=num_mini_batches,
        total_episodes=total_episodes,
        local_rollout_forward_batch_size=local_rollout_forward_batch_size,
        num_sample_generations=num_sample_generations,
        base_model=base_model,
        response_length=max_output_length,
        stop_token=stop_token,
        penalty_reward_value=penalty_reward_value,
        non_eos_penalty=non_eos_penalty,
        reward_model_path=reward_model_dir,
        sft_model_path=sft_model_path,
        num_ppo_epochs=num_ppo_epochs,
        vf_coef=vf_coef,
        kl_coef=kl_coef,
        learning_rate=learning_rate,
        report_to=report_to,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=random_seed,
        output_dir=output_dir,
    )
    wandb.init(name=run_name)

    if not reload:
        shutil.rmtree(output_dir, ignore_errors=True)
    set_seed(random_seed)
    
    

    if reload:
        reload_dir = reload_dir + "_alpha_" + str(alpha)
        reload_dir = find_latest_folder(reload_dir)
    model_manager = ModelManager(
        agent_num=agent_num,
        base_model=base_model,
        sft_model_path=sft_model_path,
        peft_config=peft_config,
        round_num=round_num,
        policy_separation=policy_separation,
        collaboration_separation=collaboration_separation,
        reload=reload,
        reload_dir=reload_dir,
        task_training=task_training,
        diff_model_training=diff_model_training,
        another_policy_base_model=another_policy_base_model,
        another_policy_sft_model_path=another_policy_sft_model_path
    )

    # Access models for each agent
    tokenizer, policy, value_heads = model_manager.tokenizers[0], model_manager.policies[0], model_manager.value_heads_list[0]
    if diff_model_training:
        another_policy = model_manager.policies[1:]
        another_value_heads = model_manager.value_heads_list[1:]
        another_policy_tokenizers = model_manager.tokenizers[1:]
    else:
        another_policy = None
        another_value_heads = None
        another_policy_tokenizers = None


    eval_dataset = prepare_dataset(dataset_name, tokenizer, "test", debug)
    logger.info(f"Evaluation dataset length: {len(eval_dataset)}")
    train_dataset = prepare_dataset(dataset_name, tokenizer, "train", debug)
    logger.info(f"Training dataset length: {len(train_dataset)}")
    if reload:
        last_update_num = extract_last_number_after_slash(reload_dir)
        checkpoint = torch.load(f"{reload_dir}/checkpoint/checkpoint_update_{last_update_num}.pt")
        trial_num = checkpoint["trial"]
        update_last = extract_last_number_after_slash(reload_dir)
        # Calculate start index to resume training from the correct position
        # Formula: batch_size * num_gpus * (last_update + buffer_trials * trial_number)
        start_index = local_rollout_forward_batch_size * 8 * (update_last + 10 * trial_num)
    else:
        start_index = 0 
        update_last = 0
    if len(train_dataset) > start_index:
        train_dataset = train_dataset.select(range(start_index, len(train_dataset))) 
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        dataset_name=dataset_name,
        min_length=min_output_length,
        criteria_for_consensus_percentage=criteria_for_consensus_percentage,
        score_penalty_for_nonvalid=score_penalty_for_nonvalid,
        answer_detecting=answer_detecting,
        criteria_for_consensus_reward_threshold=criteria_for_consensus_reward_threshold,
        value_heads=value_heads,
        train_dataset=train_dataset,
        summary=summary,
        reward_feedback=reward_feedback,
        eval_dataset=eval_dataset,
        agent_num=agent_num,
        round_num=round_num,
        rule_horizon=rule_horizon,
        rule_agent_share=rule_agent_share,
        server_dict=server_dict,
        rule_discount=rule_discount,
        value_simplification=value_simplification,
        policy_separation=policy_separation,
        task_training=task_training,
        collaboration_separation=collaboration_separation,
        non_eos_penalty=non_eos_penalty,
        non_box_penalty=non_box_penalty,
        penalty_reward_value=penalty_reward_value,
        reload=reload,
        reload_epoch=update_last,
        turn_based_training=turn_based_training,
        reload_dir=reload_dir,
        alpha=alpha,
        no_reward_model=no_reward_model,
        
        diff_model_training=diff_model_training,
        another_policy=another_policy,
        another_value_heads=another_value_heads,
        another_policy_tokenizers=another_policy_tokenizers,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    


