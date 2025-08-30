# Standard library imports
import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
from peft import get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Local imports
from utils.utils_general import zero_and_freeze_adapter

# Configure logging
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self,
                 agent_num: int,
                 base_model: str,
                 sft_model_path: str,
                 peft_config,
                 round_num: int = 1,
                 policy_separation: bool = False,
                 collaboration_separation: bool = False,
                 reload: bool = False,
                 reload_dir: Optional[str] = None,
                 task_training: bool = True,
                 diff_model_training: bool = False,
                 another_policy_base_model: Optional[List[str]] = None,
                 another_policy_sft_model_path: Optional[List[str]] = None):
        """
        Initialize models for multiple agents.

        Args:
            agent_num: Number of agents
            base_model: Path or name of the base model for main agent
            sft_model_path: Path to the fine-tuned model for main agent
            peft_config: PEFT configuration object
            round_num: Number of rounds for policy/value heads
            policy_separation: Whether to use separate policy adapters
            collaboration_separation: Whether to use collaboration separation
            reload: Whether to reload existing adapters and weights
            reload_dir: Directory containing saved adapters and weights
            task_training: Whether to enable task training
            diff_model_training: Whether to use different models for different agents
            another_policy_base_model: List of base models for additional agents
            another_policy_sft_model_path: List of SFT model paths for additional agents
        """
        self.tokenizers = []
        self.policies = []
        self.value_heads_list = []
        peft_config_new = copy.deepcopy(peft_config)
        # Initialize main agent's model
        tokenizer, policy, value_heads = self._initialize_single_model(
            base_model=base_model,
            sft_model_path=sft_model_path,
            peft_config=peft_config_new,
            round_num=round_num,
            policy_separation=policy_separation,
            collaboration_separation=collaboration_separation,
            reload=reload,
            reload_dir=reload_dir,
            task_training=task_training
        )

        self.tokenizers.append(tokenizer)
        self.policies.append(policy)
        self.value_heads_list.append(value_heads)

        # Initialize additional agents' models if using different models
        if diff_model_training and agent_num > 1:
            if not (another_policy_base_model and another_policy_sft_model_path):
                raise ValueError("another_policy_base_model and another_policy_sft_model_path required for diff_model_training")

            for agent in range(agent_num - 1):
                reload_dir_agent = os.path.join(reload_dir, f"model_{agent+1}") if reload_dir else None
                peft_config_new = copy.deepcopy(peft_config)
                tokenizer, policy, value_heads = self._initialize_single_model(
                    base_model=another_policy_base_model[agent],
                    sft_model_path=another_policy_sft_model_path[agent],
                    peft_config=peft_config_new,
                    round_num=round_num,
                    policy_separation=policy_separation,
                    collaboration_separation=collaboration_separation,
                    reload=reload,
                    reload_dir=reload_dir_agent,
                    task_training=task_training
                )
                self.tokenizers.append(tokenizer)
                self.policies.append(policy)
                self.value_heads_list.append(value_heads)

    @staticmethod
    def _initialize_single_model(
        base_model: str,
        sft_model_path: str,
        peft_config,
        round_num: int,
        policy_separation: bool,
        collaboration_separation: bool,
        reload: bool,
        reload_dir: Optional[str],
        task_training: bool
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, Dict]:
        """Initialize a single model with its tokenizer and value heads."""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            padding_side="left",
            trust_remote_code=True
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        if base_model!="google/gemma-2-2b-it":
            policy = AutoModelForCausalLM.from_pretrained(
                sft_model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
        else:
            policy = AutoModelForCausalLM.from_pretrained(
                sft_model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                # attn_implementation ='eager'
                attn_implementation = 'flash_attention_2'
            )
        policy.resize_token_embeddings(len(tokenizer))
        def load_adapter(model, adapter_name: str, adapter_path: str, peft_config):
            if os.path.exists(adapter_path):
                model.load_adapter(adapter_path, adapter_name)
                logger.info(f"Loaded adapter {adapter_name} from {adapter_path}")
            else:
                logger.info(f"Adapter path {adapter_path} not found. Adding new adapter.")
                model.add_adapter(adapter_name, peft_config)

        def load_or_add_adapter(model, adapter_name: str, peft_config, reload: bool, reload_dir: str):
            if reload:
                load_adapter(model, adapter_name, os.path.join(reload_dir, adapter_name), peft_config)
            else:
                model.add_adapter(adapter_name, peft_config)


        # Initialize value heads
        value_heads = {}

        # Configure policy based on separation settings
        if policy_separation and not collaboration_separation:
            policy = get_peft_model(policy, peft_config, adapter_name="policy_0")
            if reload:
                load_adapter(policy, "policy_0", os.path.join(reload_dir, "policy_0"), peft_config)
            for r in range(1, round_num):
                load_or_add_adapter(policy, f"policy_{r}", peft_config, reload, reload_dir)

        elif policy_separation and collaboration_separation:
            policy = get_peft_model(policy, peft_config, adapter_name="policy")
            if reload:
                load_adapter(policy, "policy", os.path.join(reload_dir, "policy"), peft_config)
            load_or_add_adapter(policy, "col", peft_config, reload, reload_dir)
        elif not policy_separation and collaboration_separation:
            raise ValueError(
                "Invalid configuration: collaboration_separation cannot be True "
                "while policy_separation is False. Please set policy_separation=True "
                "when using collaboration_separation=True."
            )
        else:
            policy = get_peft_model(policy, peft_config, adapter_name="policy")
            if reload:
                load_adapter(policy, "policy", os.path.join(reload_dir, "policy"), peft_config)

        # Setup value heads
        for r in range(round_num):
            adapter_name = f"value_{r}"
            value_heads[adapter_name] = nn.Linear(policy.config.hidden_size, 1, bias=False)
            load_or_add_adapter(policy, adapter_name, peft_config, reload, reload_dir)

            if reload:
                value_head_dir = os.path.join(reload_dir, f"value_head_value_heads_{r}")
                safetensors_path = os.path.join(value_head_dir, "model.safetensors")

                if os.path.exists(safetensors_path):
                    try:
                        state_dict = load_file(safetensors_path)
                        value_heads[adapter_name].load_state_dict(state_dict)
                        logger.info(f"Successfully loaded value head weights for {adapter_name}")
                    except Exception as e:
                        logger.error(f"Error loading value head weights for {adapter_name}: {str(e)}")
                else:
                    logger.warning(f"Value head weights file not found for {adapter_name}")

        # Add and configure reference adapter
        policy.add_adapter("ref", peft_config)
        zero_and_freeze_adapter(policy, "ref")

        if not task_training:
            if policy_separation:
                if collaboration_separation:
                    zero_and_freeze_adapter(policy, "policy")
                else:
                    zero_and_freeze_adapter(policy, "policy_0")
            else:
                print("error: policy_separation and task_training can't be False at the same time")

        return tokenizer, policy, value_heads
