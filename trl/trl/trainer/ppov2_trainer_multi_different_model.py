# Standard library imports
import gc
import json
import os
import pickle
import re
import time
from collections import Counter, OrderedDict, defaultdict
from contextlib import ExitStack, nullcontext
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_peft_available

# Local imports
from utils.dataset import DatasetFactory, extract_ans_from_response, extract_ans_from_response_reasoning
from utils.patch.peft_tuner import apply_patch_peft_tuner
apply_patch_peft_tuner()
from utils.utils_general import check_initial_weights, make_grad_nowork_adapter, make_grad_work_adapter, zero_and_freeze_adapter, extract_last_number_after_slash
from utils.utils_cooperateLLM import update_turn_based_training, stat_all, check_repeated_sequences, extract_reward

from ..core import masked_mean, masked_whiten
from ..models.utils import unwrap_model_for_generation
from ..trainer.utils_multi_unified import (
    bonus_rule,
    construct_message_multi_agent_summary,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    generate_only_response,
    get_reward,
    graph_gen_turn,
    graph_gen_turn_specific,
    logit_from_different_server,
    prepare_deepspeed,
    print_rich_table,
    score_rule,
    truncate_gemma_repetitive,
    truncate_response,
    visualize_chat_in_columns,
    create_optimizer,
    create_scheduler,
    set_adapter_for_models_ppo,
    TEMPERATURE_DICT,
)
from ..trainer.utils_multi_unified_chat import (
    chat_for_answer,
    chat_for_no_reasoning_check,
    combine_agent_contexts,
    construct_assistant_message,
    construct_message_multi_agent,
    construct_message_multi_agent_eval,
    formatting_question,
    message_formatter
)
from utils.utils_cooperateLLM import (
    reward_from_different_server,
    grad_requires_setting,
    check_agent_consensus,
    calculate_reward_from_answer
)
from ..trainer.ppov2_config import PPOv2Config

# Constants
INVALID_LOGPROB = 1.0


# open_ended_answer <- compare model's answer with true answer and wrong answer
class CustomDataCollator:
    def __init__(self, tokenizer, open_ended, padding=True):
        self.tokenizer = tokenizer
        self.open_ended = open_ended
        self.padding = padding

    def __call__(self, batch):
        # Extract input_ids, lengths, and answers
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        lengths = [item["lengths"] for item in batch]
        answers = [torch.tensor(item["answer"]) for item in batch]

        # Pad the input_ids and answers
        if self.padding:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            answers = pad_sequence(answers, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.open_ended:
            wrong_answers = [torch.tensor(item["wrong_answer"]) for item in batch]
            if self.padding:
                wrong_answers = pad_sequence(wrong_answers, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return {
                "input_ids": input_ids,       # Tensor of input IDs with padding
                "lengths": torch.tensor(lengths),  # Tensor of original lengths
                "answers": answers,            # Tensor of answers with padding
                "wrong_answers": wrong_answers  # Tensor of wrong answers with padding
            }            
        return {
            "input_ids": input_ids,       # Tensor of input IDs with padding
            "lengths": torch.tensor(lengths),  # Tensor of original lengths
            "answers": answers            # Tensor of answers with padding
        }


class PolicyAndValueWrapper(nn.Module):
    """
    A wrapper that combines policy and value heads for PPO training.

    This class serves as a container to hold both the policy model and multiple
    value heads together, allowing them to be treated as a single model for
    optimization and gradient computation.

    Attributes:
        policy: The policy model (typically a language model with LoRA adapters)
        value_heads_X: Dynamically created attributes for value heads, where X
                      is the turn number (0, 1, 2, etc.)
    """

    def __init__(
        self,
        policy: nn.Module,
        value_heads: Dict[str, nn.Module]
    ) -> None:
        """
        Initialize the PolicyAndValueWrapper.

        Args:
            policy: The policy model component
            value_heads: Dictionary mapping turn names to value head models
                        (e.g., {"value_0": value_head_0, "value_1": value_head_1})
        """
        super().__init__()
        self.policy = policy

        # Store value heads as attributes for easy access
        for turn_name, value_head in value_heads.items():
            if turn_name.startswith("value_"):
                turn_num = turn_name.split("_")[-1]
                setattr(self, f"value_heads_{turn_num}", value_head)

    def forward(self, turn: int, **kwargs):
        pass


class PPOv2Trainer(Trainer):
    def __init__(
        self,
        # Core required parameters
        config: PPOv2Config,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        dataset_name: str,
        train_dataset: Dataset,
        value_heads,
        server_dict: dict,

        # Optional core parameters
        data_collator=None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,

        # Checkpoint/resume functionality
        reload: bool = False,
        reload_dir: Optional[str] = None,
        reload_epoch: int = 0,

        # Multi-agent training parameters
        agent_num: int = 1,
        round_num: int = 1,

        # Reward and scoring parameters
        min_length: int = 0,
        penalty_reward_value: float = 0,
        alpha: List[float] = [0, 0, 0, 0],

        # Consensus parameters
        criteria_for_consensus_percentage: float = 0.8,
        criteria_for_consensus_reward_threshold: float = 0.7,

        # Training control parameters
        turn_based_training: bool = False,
        turn_based_training_num: int = 40,
        update_turn: int = 20,
        update_initial_turn: int = 20,
        value_simplification: bool = True,

        # Multi-agent specific features
        reward_feedback: bool = False,
        diff_model_training: bool = False,
        another_policy: Optional[List[nn.Module]] = None,
        another_value_heads: Optional[List] = None,
        another_policy_tokenizers: Optional[List[PreTrainedTokenizer]] = None,

        # Advanced training features
        summary: bool = False, 
        #### summary part is not implemented yet. 
        policy_separation: bool = True,
        collaboration_separation: bool = False,
        task_training: bool = False,
        no_reward_model: bool = False,

        # Penalty and validation parameters
        non_eos_penalty: bool = True,
        non_box_penalty: bool = True,
        repeat_penalty: bool = True,
        score_penalty_for_nonvalid: bool = False,
        answer_detecting: bool = False,

        # Reward rule parameters
        rule_horizon: str = "discounted_sum",
        rule_agent_share: str = "individual",
        rule_discount: float = 1.00,

    ) -> None:
        """
            alpha (List[float], default=[0, 0, 0, 0])
                Purpose: Weighting coefficients for incentive mechanisms in multi-agent reward calculation
                Usage: Passed to bonus_rule() and score_rule() functions to control how different types of incentives are weighted
                Context: Used in collaborative learning scenarios to balance different reward components

            reward_feedback (bool, default=False)
                Purpose: Controls whether agents receive feedback about rewards during training
                Usage: Affects message construction in construct_message_multi_agent() and construct_message_multi_agent_summary()
                Context: When True, agents can see reward information from other agents, enabling more informed decision-making

            turn_based_training (bool, default=False)
                Purpose: Enables progressive training where the model learns different skills at different training stages
                Usage: Combined with turn_based_training_num to determine training phases via update_turn_based_training()
                Context: Allows curriculum learning where complexity increases over time -- for memory issue but if you can address this issue, you do not need it. 

            value_simplification (bool, default=True)
                Purpose: Simplifies value function computation to reduce computational overhead
                Usage: Controls whether to use simplified value estimation methods

            policy_separation (bool, default=True)
                Purpose: Train one model to be good with every multi agent system if it is False. Default setting is True. 

            collaboration_separation (bool, default=True)
                Purpose: We use separate model from turn1 and turn 2+ because turn1 is more about domain knowledge and turn 2+ is more about making collaborative strategy. 

            task_training (bool, default=False)
                Purpose: Decide to train turn 1 agent as well so that we can acquire domain knowledge 

            no_reward_model (bool, default=False)
                Purpose: Disables the use of an external reward model during training
                Usage: When True, relies on answer based rewards instead of learned reward functions
                Context: Useful when ground-truth rewards are available or reward modeling is not needed

            update_initial_turn (int, default=20)
                Purpose: Initial training steps before advanced features are activated -- just for learning the answer format. 
                Usage: Used in update_turn_based_training() to determine when to start progressive training phases
                Context: Warm-up period for stable training before introducing complexity

            penalty_reward_value (float, default=0)
                Purpose: Penalty value applied to invalid or incorrect responses
                Usage: Applied to scores when penalty_reward_value conditions are met
                Context: Encourages valid responses by penalizing inappropriate outputs

            criteria_for_consensus_percentage (float, default=0.8)
                Purpose: Minimum percentage of agents that must agree for consensus
                Usage: Used in check_consensus() to determine if agents have reached agreement
                Context: Controls how strict consensus requirements are in multi-agent scenarios

            criteria_for_consensus_reward_threshold (float, default=0.7)
                Purpose: Minimum average reward threshold for consensus validation
                Usage: Combined with criteria_for_consensus_percentage in consensus checking
                Context: Ensures consensus is not just majority rule but also quality-based

            score_penalty_for_nonvalid (bool, default=False)
                Purpose: Applies penalties to invalid or malformed responses
                Usage: Controls whether to modify training scores for non-valid outputs
                Context: Quality control mechanism to discourage invalid generations

            reload (bool, default=False)
                Purpose: Enables checkpoint reloading to resume interrupted training -- especially memory issue makes training hard so I made automatic shell file to resume training. 
                Usage: When True, loads saved model state from reload_dir at specified reload_epoch. Note that you might want to change some details in here for reload functionality, such as when you want to start again, etc, etc

            diff_model_training (bool, default=False)
                Purpose: Enables training with different base models for different agents
                Usage: Controls setup of multiple models, optimizers, and accelerators
                Context: Supports heterogeneous base model multi-agent training scenarios
            """
        self.args = config
        args = config
        self.config = config  # Alias for convenience

        # Store core components
        self.tokenizer = tokenizer
        self.policy = policy
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_heads = value_heads
        self.server_dict = server_dict
        self.dataset_name = dataset_name

        # Configure policy generation settings
        self.policy.generation_config.eos_token_id = None  # Disable EOS token for generation
        self.policy.generation_config.pad_token_id = None  # Disable padding token

        # Store optional components
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks

        # Training configuration
        self.agent_num = agent_num
        self.round_num = round_num
        self.min_length = min_length

        # Reward and penalty settings
        self.alpha = alpha
        self.penalty_reward_value = penalty_reward_value
        self.non_eos_penalty = non_eos_penalty
        self.non_box_penalty = non_box_penalty
        self.repeat_penalty = repeat_penalty

        # Consensus settings
        self.criteria_for_consensus_percentage = criteria_for_consensus_percentage
        self.criteria_for_consensus_reward_threshold = criteria_for_consensus_reward_threshold

        # Training control settings
        self.turn_based_training = turn_based_training
        self.turn_based_training_num = turn_based_training_num
        self.update_turn = update_turn
        self.update_initial_turn = update_initial_turn
        self.value_simplification = value_simplification

        # Multi-agent features
        self.reward_feedback = reward_feedback
        self.diff_model_training = diff_model_training
        self.another_policy = another_policy or []
        self.another_value_heads = another_value_heads or []
        self.another_policy_tokenizers = another_policy_tokenizers or []

        # Advanced training features
        self.summary = summary
        self.policy_separation = policy_separation
        self.collaboration_separation = collaboration_separation
        self.task_training = task_training
        self.no_reward_model = no_reward_model

        # Validation and penalty features
        self.score_penalty_for_nonvalid = score_penalty_for_nonvalid
        self.answer_detecting = answer_detecting

        # Reward rule settings
        self.rule_horizon = rule_horizon
        self.rule_agent_share = rule_agent_share
        self.rule_discount = rule_discount

        # Checkpoint/resume functionality
        self.reload = reload
        self.reload_dir = reload_dir

        # Determine answer format based on dataset
        if self.dataset_name in ["ANLI", "GSM8k"]:
            self.open_ended_answer = False
        elif self.dataset_name in ["FANTOM"]:
            self.open_ended_answer = True
        else:
            raise ValueError(f"Dataset name '{self.dataset_name}' not supported. "
                           "Supported datasets: ANLI, GSM8k, FANTOM")

        # Setup optimizers
        if optimizers[0] is not None:
            self.optimizer, self.lr_scheduler = optimizers
        else:
            raise ValueError("optimizers must be specified")

        # Handle checkpoint reloading
        if self.reload:
            if not self.reload_dir:
                raise ValueError("reload_dir must be specified when reload=True")
            last_update_num = extract_last_number_after_slash(self.reload_dir)
            checkpoint_path = f"{self.reload_dir}/checkpoint/checkpoint_update_{last_update_num}.pt"
            checkpoint = torch.load(checkpoint_path)
            self.updates_start_epoch = reload_epoch
            self.trial = checkpoint["trial"] + 1
        else:
            self.updates_start_epoch = 0
            self.trial = 1
        


        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        
        if not self.diff_model_training:
            accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
            self.accelerator = accelerator
            main_accelerator = self.accelerator
        else:
            self.accelerator = [Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) for _ in range(self.agent_num)]
            main_accelerator = self.accelerator[0]
        args.world_size = main_accelerator.num_processes

        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
`
        args.num_updates = args.total_episodes // args.batch_size

        time_tensor = torch.tensor(int(time.time()), device=main_accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + main_accelerator.process_index * 100003
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)

        #########
        # setup model, optimizer, and others
        #########
        # for module in [policy, ref_policy, value_model, reward_model]:
        disable_dropout_in_model(policy)
        if self.diff_model_training:
            for model in self.another_policy:
                disable_dropout_in_model(model)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_heads)
        # Keep first optimizer/scheduler as default for non-turn-based training
        self.optimizer = create_optimizer(self.args.learning_rate, self.model.parameters())
        self.lr_scheduler = create_scheduler(self.optimizer, int(args.num_updates)) 


        # Another model setup
        if self.diff_model_training:
            self.another_model = [
                PolicyAndValueWrapper(po, vh)
                for po, vh in zip(self.another_policy, self.another_value_heads)
            ]
            
            # Create multiple optimizers for each model for rounds
            self.another_optimizer = [create_optimizer(self.args.learning_rate, model.parameters()) for model in self.another_model ] 
            self.another_lr_scheduler = [create_scheduler(opt, int(args.num_updates)) for opt in self.another_optimizer]



        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks

        self.callback_handler = CallbackHandler(self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler)


        if self.diff_model_training:
            self.another_callback_handler = [
                CallbackHandler(
                    self.callbacks, model, tok, opt, sched
                )
                for model, tok, opt, sched in zip(
                    self.another_model,
                    self.another_policy_tokenizers,
                    self.another_optimizer,
                    self.another_lr_scheduler
                )
            ]


        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(main_accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(main_accelerator.state, "fsdp_plugin", None) is not None
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=CustomDataCollator(tokenizer, self.open_ended_answer),
            # collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)

        self.model, self.optimizer, self.dataloader = main_accelerator.prepare(self.model, self.optimizer, self.dataloader)            

        # Prepare with accelerators
        if self.diff_model_training:
            # Prepare main model with first accelerator
            # Prepare each additional model with its own accelerator
            for idx, (model, opt) in enumerate(zip(self.another_model, self.another_optimizer)):
                acc_idx = idx + 1  # Accelerator index (offset by 1 since main model uses index 0)
                self.another_model[idx], self.another_optimizer[idx] = self.accelerator[acc_idx].prepare(model, opt)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        # Prepare eval dataloader with first accelerator
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=CustomDataCollator(self.tokenizer, self.open_ended_answer),
            drop_last=True,
        )    
        self.eval_dataloader = main_accelerator.prepare(self.eval_dataloader)
        self.iter_dataloader = self.repeat_generator()




    def _save(self, model, output_dir: Optional[str] = None, state_dict=None, acc = None, adapter_save = None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, supported_classes) and adapter_save is None:
            if state_dict is None:
                state_dict = model.state_dict()
            if isinstance(acc.unwrap_model(model), supported_classes):
                acc.unwrap_model(model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        TRAINING_ARGS_NAME = "training_args.bin"

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def push_to_hub(self, **kwargs):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        self.backup_model = self.model
        main_accelerator = self.accelerator if not self.diff_model_training else self.accelerator[0]
        self.model = main_accelerator.unwrap_model(self.model).policy  # save only the policy
        super().push_to_hub(**kwargs)
        self.model = self.backup_model


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Save main model
        main_accelerator = self.accelerator if not self.diff_model_training else self.accelerator[0]
        if not _internal_call:
            self.backup_model = self.model
            self.model = main_accelerator.unwrap_model(self.model).policy

        
        state_dict = main_accelerator.get_state_dict(self.backup_model)
        policy_state_dict = OrderedDict(
            {k[len("policy."):]: v for k, v in state_dict.items() if k.startswith("policy.")}
        )

        if self.args.should_save:
            self._save(self.model, output_dir, state_dict=policy_state_dict, acc = main_accelerator)

        if not _internal_call:
            self.model = self.backup_model
        full_model = main_accelerator.unwrap_model(self.model)

        value_heads = [attr for attr in dir(full_model) if attr.startswith('value_heads_')]
            
        if value_heads:
            for head in value_heads:
                value_head_state_dict = OrderedDict(
                    {k[len(f"{head}."):]: v for k, v in state_dict.items() if k.startswith(f"{head}.")}
                )
                output_dir_head = f"{output_dir}/value_head_{head}"
                if self.args.should_save:
                    self._save(self.model, output_dir_head, state_dict=value_head_state_dict, acc = main_accelerator)

        if self.args.should_save:
            update = extract_last_number_after_slash(output_dir)
            checkpoint = {
                'update': update,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if hasattr(self, 'lr_scheduler') else None,
                'args': self.args,
                "trial": self.trial
            }
            checkpoint_dir = f"{output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_update_{update}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save additional models if they exist
        if hasattr(self, 'another_model') and self.diff_model_training:
            for idx, (model, optimizer, scheduler) in enumerate(zip(
                self.another_model,
                self.another_optimizer,
                self.another_lr_scheduler
            )):
                # Save in subdirectory
                model_dir = f"{output_dir}/model_{idx+1}"
                os.makedirs(model_dir, exist_ok=True)
                
                # Save policy
                if not _internal_call:
                    backup_model = model
                    model = self.accelerator[idx + 1].unwrap_model(model).policy

                model_state_dict = self.accelerator[idx + 1].get_state_dict(backup_model)
                model_policy_state_dict = OrderedDict(
                    {k[len("policy."):]: v for k, v in model_state_dict.items() if k.startswith("policy.")}
                )

                if self.args.should_save:
                    self._save(model, model_dir, state_dict=model_policy_state_dict, acc = self.accelerator[idx + 1], adapter_save = True)

                if not _internal_call:
                    model = backup_model
                model_full = self.accelerator[idx + 1].unwrap_model(model)

                # Save value heads
                value_heads = [attr for attr in dir(model_full) if attr.startswith('value_heads_')]
                
                if value_heads:
                    for head in value_heads:
                        value_head_state_dict = OrderedDict(
                            {k[len(f"{head}."):]: v for k, v in model_state_dict.items() if k.startswith(f"{head}.")}
                        )
                        output_dir_head = f"{model_dir}/value_head_{head}"
                        if self.args.should_save:
                            self._save(self.another_model, output_dir_head, state_dict=value_head_state_dict, acc = self.accelerator[idx + 1])

                # Save checkpoint
                if self.args.should_save:
                    checkpoint = {
                        'update': update,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'args': self.args,
                        "trial": self.trial
                    }
                    checkpoint_dir = f"{model_dir}/checkpoint"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_update_{update}.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint for model_{idx+1}: {checkpoint_path}")

                del model_full

        del full_model
        torch.cuda.empty_cache()

    def repeat_generator(self):
        while True:
            for batch in self.dataloader:
                yield batch
    
    
    ### For training, SUMMARY VERSION IS NOT COMPLETED YET SO I COMMENTED IT OUT -- actually during the training, I changed some part and it worked before so it doesn't need many updates for summary version for different model training. For now, I just comment it out. This is not included in the paper. But I also checked that previous version of summary works and have similar result with non summary version -- but there is a lot of room to improve it. 

    def train(self):
        args = self.args
        main_accelerator = self.accelerator if not self.diff_model_training else self.accelerator[0]
        
        model = self.model
        tokenizer = self.tokenizer
        if self.diff_model_training:
            another_model = self.another_model
            another_tokenizer = self.another_policy_tokenizers

        generation_config_dict = {} 
        for key in TEMPERATURE_DICT:
            generation_config_dict[key] = GenerationConfig(
                max_new_tokens=args.response_length if key != "gen_answer" else 15,
                min_new_tokens=args.response_length if key != "gen_answer" else 15,
                temperature=(TEMPERATURE_DICT[key] + 1e-7),
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )
        
        device = main_accelerator.device

        iter_dataloader = self.iter_dataloader
        training_turn = self.round_num -1

        main_accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        model.train()
        if self.diff_model_training:
            for another_model_ind in range(len(another_model)):
                another_model[another_model_ind].train()
        else:
            another_model = None

        for update in range(self.updates_start_epoch, args.num_updates):


            if self.turn_based_training:
                update_turn = update_turn_based_training(update, self.criteria_for_consensus_percentage, self.criteria_for_consensus_reward_threshold,
                                                       self.update_initial_turn, self.turn_based_training_num, self.round_num)
                
            main_optimizer = self.optimizer
            if self.diff_model_training:
                another_optimizer = self.another_optimizer

            model.train()
            if self.diff_model_training:
                for another_model_ind in range(len(another_model)):
                    another_model[another_model_ind].train()

            all_model_names = [model.module.policy.name_or_path]
            if self.diff_model_training:
                for another_model_ind in range(len(another_model)):
                    all_model_names.append(another_model[another_model_ind].module.policy.name_or_path)
            else:
                for agent in range(self.agent_num):
                    all_model_names.append(model.module.policy.name_or_path)


            stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps, self.round_num)
            approxkl_stats = torch.zeros(stats_shape, device=device)
            pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
            pg_loss_stats = torch.zeros(stats_shape, device=device)
            vf_loss_stats = torch.zeros(stats_shape, device=device)
            vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
            entropy_stats = torch.zeros(stats_shape, device=device)
            ratio_stats = torch.zeros(stats_shape, device=device)
            count_samples_stats = torch.zeros((args.num_mini_batches, self.round_num), device=device)
            if device == torch.device("cuda:0"):
                print("=============================")
                print(f"{update} update for training")

            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            
            with torch.no_grad():
                query_responses_turn_agent = {}
                query_responses_no_history_turn_agent = {}
                responses_turn_agent = {}
                sequence_lengths_turn_agent = {}
                context_length_turn_agent = {}
                logprobs_turn_agent = {}
                ref_logprobs_turn_agent = {}
                values_turn_agent = {}
                scores_turn_agent = {}
                correctnesses_turn_agent = {} 
                rewards_turn_agent = {}
                advantages_turn_agent = {}
                returns_turn_agent = {}
                sequence_lengths_p1_turn_agent = {}
                padding_mask_turn_agent = {}   
                padding_mask_p1_turn_agent = {}
                kl_turn_agent = {}
                non_score_reward_turn_agent = {} 
                unfinished_question_turn = []
                with ExitStack() as stack:
                    unwrapped_model = stack.enter_context(unwrap_model_for_generation(model, main_accelerator))
                    if self.diff_model_training:
                        unwrapped_other_models = [
                            stack.enter_context(unwrap_model_for_generation(m, self.accelerator[idx + 1]))
                            for idx, m in enumerate(another_model)
                        ]
                    question_text_set = tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)
                    answer_text_set = []
                    for answer in data["answers"]:
                        answer_text_set.append(extract_ans_from_response(self.dataset_name, tokenizer.decode(answer, skip_special_tokens=True)))

                    if self.open_ended_answer:
                        wrong_answer_text_set = []
                        for wrong_answer in data["wrong_answers"]:
                            wrong_answer_text_set.append(extract_ans_from_response(self.dataset_name, tokenizer.decode(wrong_answer, skip_special_tokens=True)))
                    agent_contexts_set = [[[{"role": "user", "content": formatting_question(self.dataset_name, all_model_names[agent], text)}] for agent in range(self.agent_num)] for text in question_text_set] # [ [agent context for text 1] [agent context for text 2]..]
                    agent_reward_set = [[-1 for agent in range(self.agent_num)] for text in question_text_set]

                    answers_all_agent_turn = [[] for i in range(self.round_num)]
                    rewards_all_agent_turn = [[] for i in range(self.round_num)]
                    finished_question = [-1 for i in range(len(question_text_set))]

                    unfinished_question_index = [index for index, value in enumerate(finished_question) if value == -1]
                    for turn in range(self.round_num):
                        unfinished_question_turn.append(unfinished_question_index)
                        if len(unfinished_question_index) == 0:
                            continue
                        if self.policy_separation and not self.collaboration_separation:
                            set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                f"policy_{turn}", self.diff_model_training)
                        elif self.policy_separation and self.collaboration_separation:
                            adapter_name = "policy" if turn == 0 else "col"
                            set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                adapter_name, self.diff_model_training)
                        else:
                            set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                "policy", self.diff_model_training)
                        answers_all_agent = [[] for i in range(self.agent_num)]
                        rewards_all_agent = [[] for i in range(self.agent_num)]
                        for agent in range(self.agent_num):
                            if self.diff_model_training:
                                current_model = unwrapped_model if agent == 0 else unwrapped_other_models[agent-1]
                                current_tokenizer = tokenizer if agent == 0 else another_tokenizer[agent-1]
                            else:
                                current_model = unwrapped_model
                                current_tokenizer = tokenizer


                            if turn != 0:            
                                agent_contexts_other_set  = [agent_contexts[:agent] + agent_contexts[agent + 1:] for agent_contexts in agent_contexts_set]
                                # if device == torch.device("cuda:0"):
                                    # print("agent contexts other set in device ", device, agent_contexts_other_set)

                                #### COMMENTED OUT -- SUMMARY VERSION 
                                # if self.summary and self.agent_num != 1:
                                #     summary_input_text = [[] for i in range(len(question_text_set))]
                                #     summary_result = ["_" for _ in range(len(question_text_set))]  # Initialize with "_"
                                #     for q in range(len(question_text_set)):
                                #         prefix_string = f'''There are other {self.agent_num - 1} agents who are trying to provide the answer to the following question: \n\n {question_text_set[q]} \n\n'''
                                #         for idx, ans in enumerate(agent_contexts_other_set[q]):
                                #             if self.reward_feedback:
                                #                 agent_response = ans[3 * turn - 2]["content"]
                                #                 agent_reward = ans[3 * turn - 1]["content"]
                                #             prefix_string = prefix_string + f"Agent {idx + 1} response: {agent_response}\n\n"
                                #             prefix_string += f"Agent {idx + 1} response's reward: {agent_reward}\n\n"
                                #             if self.reward_feedback:
                                #                 prefix_string = prefix_string + (
                                #                     f"Please summarize each agent's response individually, ensuring that the logic behind each response is included. "
                                #                     f"Only provide the summary, and please maintain the reward of each answer.\n "
                                #                     f"For example: Agent 1 - Summary: XXX, Reward: XXX.\n\n Agent 2 - Summary: XXX, Reward: XXX."
                                #                 )
                                #             else:
                                #                 prefix_string = prefix_string + f"Please summarize each agent's response individually, ensuring that the logic behind each response is included. Only provide the summary please. \n\n" +  f"For example: Agent 1 - Summary: XXX.\n\n Agent 2 - Summary: XXX. \n\n"
                                #             summary_input_text[q] = {"role": "user", "content": prefix_string}
                                #     summary_input_text = current_tokenizer.apply_chat_template(
                                #         summary_input_text, 
                                #         add_generation_prompt=True, 
                                #         return_tensors="pt", 
                                #         padding=True
                                #     ).to(device)

                                #     summary_input_text = current_tokenizer.apply_chat_template(
                                #         summary_input_text, 
                                #         add_generation_prompt=True, 
                                #         return_tensors="pt", 
                                #         padding=True
                                #     ).to(device)
                                    
                                #     context_length = summary_input_text.shape[1]
                                #     current_model.policy.set_adapter("ref")
            
                                #     for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                                #         small_batch_non_finished_index = [j for j in range(i, i + args.local_rollout_forward_batch_size) if j in unfinished_question_index]
                                #         summary_input_text_small_batch = summary_input_text[small_batch_non_finished_index]
                                #         if len(small_batch_non_finished_index) == 0:
                                #             continue
                                #         summary_response, _ = generate_only_response(
                                #             current_model.policy,
                                #             summary_input_text_small_batch,
                                #             current_tokenizer.pad_token_id,
                                #             generation_config_dict[current_tokenizer.init_kwargs["name_or_path"]],
                                #         )                                               
                                #         postprocessed_summary_response = summary_response
                                #         if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                                #             ending_tokens = ["<|end|>"]
                                #             truncation_tokens = [current_tokenizer.eos_token_id]
                                #             for ending_token in ending_tokens:
                                #                 encoded_end_token = current_tokenizer.encode(ending_token, add_special_tokens=False)
                                #                 if len(encoded_end_token) == 1:
                                #                     truncation_tokens.append(encoded_end_token[0])

                                #             if current_model.policy.config._name_or_path == "Qwen/Qwen2.5-3B-Instruct":
                                #                 truncation_tokens.append(4710)

                                #             postprocessed_summary_response = truncate_response(
                                #                 truncation_tokens,
                                #                 current_tokenizer.pad_token_id,
                                #                 summary_response
                                #             )
                                        
                                #         for idx, q in enumerate(small_batch_non_finished_index):
                                #             text = current_tokenizer.decode(postprocessed_summary_response[idx], skip_special_tokens=True)
                                #             summary_result[q] = text  # Save the generated summary for the correct index

                                #     for q in range(len(question_text_set)):
                                #         agent_contexts_set[q][agent].append(construct_message_multi_agent_summary(summary_result[q], question_text_set[q], self.reward_feedback))
                                #     del summary_input_text, summary_response, postprocessed_summary_response, summary_result                                    
                                # else:

                                #### COMMENTED OUT -- SUMMARY VERSION 

                                for q in range(len(question_text_set)):
                                    message = construct_message_multi_agent(agent_contexts_other_set[q], formatting_question(self.dataset_name, all_model_names[agent],  question_text_set[q]), turn, self.dataset_name, self.reward_feedback)
                                    agent_contexts_set[q][agent].append(message)
                                    del message 
                                del agent_contexts_other_set
                                
                            agent_context_set = combine_agent_contexts(agent_contexts_set, agent, turn, self.reward_feedback)



                            queries = current_tokenizer.apply_chat_template(agent_context_set, add_generation_prompt=True, return_tensors="pt", padding = True).to(device)

                            if (turn == 0) & (agent == 0):
                                raw_agent_context_set = []
                            if turn == 0:
                                raw_agent_context_set.append(agent_context_set.copy())

                            del agent_context_set
                            context_length = queries.shape[1]
                            context_length_turn_agent[(turn, agent)] = context_length
                            query_responses = [] 
                            query_responses_no_history = []
                            responses = [] 
                            postprocessed_responses = [] 
                            logprobs = [] 
                            ref_logprobs = [] 
                            values = [] 
                            scores = [] 
                            sequence_lengths = []
                            answers = []
                            scores_with_penalty = []
                            scores_for_training = []
                            scores_for_training_with_penalty = []
                            correctnesses_for_training = []



                            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                                queries = queries.to("cpu")
                                original_queries = current_tokenizer.apply_chat_template(
                                        raw_agent_context_set[agent], 
                                        add_generation_prompt=True, 
                                        return_tensors="pt", 
                                        padding=True
                                    ).to(device)

                                original_queries = original_queries.to("cpu")
                                small_batch_non_finished_index = [j for j in range(i, i + args.local_rollout_forward_batch_size) if j in unfinished_question_index]
                                if len(small_batch_non_finished_index) == 0:
                                    for q in range(i, min(i + args.local_rollout_forward_batch_size, queries.shape[0])):
                                        if q in small_batch_non_finished_index:
                                            idx = small_batch_non_finished_index.index(q)  # Find the index within the finished list
                                            text = current_tokenizer.decode(postprocessed_response[idx], skip_special_tokens=True)
                                            assistant_message = construct_assistant_message(text)
                                            agent_contexts_set[q][agent].append(assistant_message)
                                            del text, assistant_message
 
                                        else:
                                            agent_contexts_set[q][agent].append({'role': 'assistant', 'content': "Consensus has already been reached."})
                                            if self.reward_feedback:
                                                agent_contexts_set[q][agent].append({'role': 'user', 'content': f"No reward since consensus has already been reached." })
                                else:
                                    query = queries[small_batch_non_finished_index]
                                    answer_text_set_small_batch = [answer_text_set[j] for j in small_batch_non_finished_index]
                                    original_query = original_queries[small_batch_non_finished_index]
                                    query = query.to(device)
                                    original_query = original_query.to(device)
                                    if self.policy_separation and not self.collaboration_separation:
                                        set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                            f"policy_{turn}", self.diff_model_training)
                                    elif self.policy_separation and self.collaboration_separation:
                                        adapter_name = "policy" if turn == 0 else "col"
                                        set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                            adapter_name, self.diff_model_training)
                                    else:
                                        set_adapter_for_models_ppo(unwrapped_model, unwrapped_other_models if self.diff_model_training else None, 
                                                            "policy", self.diff_model_training)

                                    query_response, logits = generate(
                                        current_model.policy,
                                        query,
                                        current_tokenizer.pad_token_id,
                                        generation_config_dict[current_tokenizer.init_kwargs["name_or_path"]],
                                    )

                                    response = query_response[:, context_length:]

                                    all_logprob = F.log_softmax(logits, dim=-1)
                                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                                    del logits, all_logprob
                                    torch.cuda.empty_cache()
                                    current_model.policy.set_adapter("ref")
                                    assert check_initial_weights(current_model.policy, "ref") == True
                                    assert sorted(current_model.policy.active_adapters) == ["ref"]
                                    ref = True
                                    ref_output = forward(current_model.policy, ref, turn, query_response, current_tokenizer.pad_token_id, self.round_num, self.policy_separation, self.collaboration_separation, self.task_training)

                                    if self.policy_separation and not self.collaboration_separation:
                                        current_model.policy.set_adapter(f"policy_{turn}")
                                        assert sorted(current_model.policy.active_adapters) == [f"policy_{turn}"]
                                    elif self.policy_separation and self.collaboration_separation:
                                        if turn == 0:
                                            current_model.policy.set_adapter(f"policy")
                                            assert sorted(current_model.policy.active_adapters) == [f"policy"]
                                        else:
                                            current_model.policy.set_adapter(f"col")
                                            assert sorted(current_model.policy.active_adapters) == [f"col"]
                                    else:
                                        current_model.policy.set_adapter(f"policy")
                                        assert sorted(current_model.policy.active_adapters) == [f"policy"]

                                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                                    ref_logits /= args.temperature + 1e-7
                                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                                    del ref_output, ref_logits, ref_all_logprob
                                    torch.cuda.empty_cache()

                                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                                    postprocessed_response = response

                                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                                        ending_tokens = ["<|end|>"]
                                        
                                        truncation_tokens = [current_tokenizer.eos_token_id]
                                        for ending_token in ending_tokens:
                                            encoded_end_token = current_tokenizer.encode(ending_token, add_special_tokens=False)
                                            if len(encoded_end_token) == 1:
                                                truncation_tokens.append(encoded_end_token[0])
                                        if current_model.policy.config._name_or_path == "Qwen/Qwen2.5-3B-Instruct":
                                            truncation_tokens.append(4710)


                                        postprocessed_response = truncate_response(
                                            truncation_tokens,
                                            current_tokenizer.pad_token_id,
                                            response
                                        )
                                        if current_model.policy.config._name_or_path == "google/gemma-2-2b-it":
                                            postprocessed_response = truncate_gemma_repetitive(postprocessed_response, current_tokenizer.pad_token_id)


                                    for_score_query_response = torch.cat((original_query, postprocessed_response), 1)
                                    query_responses_no_history.append(for_score_query_response)

                                    sequence_length = first_true_indices(postprocessed_response == current_tokenizer.pad_token_id) - 1
                                
                                    postprocessed_responses.append(postprocessed_response)
                                    if self.diff_model_training:
                                        # Choose model based on agent index
                                        unwrapped_model_for_value = self.accelerator[agent].unwrap_model(model if agent == 0 else another_model[agent-1])
                                    else:
                                        # Always use main model
                                        unwrapped_model_for_value = main_accelerator.unwrap_model(model)


                                    if self.value_simplification:
                                        full_value = get_reward(
                                            unwrapped_model_for_value, ref, turn, for_score_query_response, current_tokenizer.pad_token_id, self.task_training
                                        )
                                        value = full_value[:, context_length_turn_agent[(0, agent)] - 1 : -1].squeeze(-1)

                                    else:
                                        full_value = get_reward(
                                            unwrapped_model_for_value, ref, turn, query_response, current_tokenizer.pad_token_id, self.task_training
                                        )
                                        value = full_value[:, context_length_turn_agent[(turn, agent)] - 1 : -1].squeeze(-1)




                                    if not self.open_ended_answer:
                                        modified_responses = []
                                        decoded_responses = [current_tokenizer.decode(postprocessed_response[q], skip_special_tokens=True).strip() for q in range(postprocessed_response.shape[0])]

                                        
                                        if self.answer_detecting: 
                                            if self.dataset_name in ["ANLI", "GSM8k"]:
                                                chat_for_identifying_answer = chat_for_answer([formatting_question(self.dataset_name, all_model_names[agent], question_text_set[i]) for i in small_batch_non_finished_index], postprocessed_response, current_tokenizer, self.dataset_name).to(device)
                                                chat_context_length = chat_for_identifying_answer.shape[1]
                                                retry = 5
                                                answer_best = []
                                                invalid_count_best = len(small_batch_non_finished_index) + 1
                                                
                                                while retry > 0:
                                                    current_model.policy.set_adapter("ref")
                                                    answer_gen, _  = generate(current_model.policy, chat_for_identifying_answer, current_tokenizer.pad_token_id,  generation_config_dict["gen_answer"],)
                                                    ### want to decode answer_gen for sanity check
                                                    

                                                    answer_gen = answer_gen[:, chat_context_length:]
                                                    #decode answer_gen
                                                    decoded_answer_gen = [current_tokenizer.decode(answer_gen[q], skip_special_tokens=True) for q in range(answer_gen.shape[0])]

                                                    answer = [
                                                        extract_ans_from_response(self.dataset_name, current_tokenizer.decode(answer_gen[q], skip_special_tokens=True))
                                                        for q in range(answer_gen.shape[0])
                                                    ]
                                                    if answer_best == []:
                                                        answer_best = answer.copy()
                                                    
                                                    for idx, aa in enumerate(answer_best):
                                                        if aa == "invalid":
                                                            answer_best[idx] = answer[idx]
                                                    
                                                    total_invalid_count = sum([1 for ans in answer if "invalid" == ans])
                                                    if total_invalid_count != 0:
                                                        retry -= 1
                                                    else:
                                                        break
                                                    # if device == torch.device("cuda:0"):
                                                    #     print("decoded_answer_gen:", decoded_answer_gen, "line 1409")
                                                    #     print("answer:", answer_best, "line 1410")                                            
                                                    del answer_gen
                                                answer = answer_best
                                                if device == torch.device("cuda:0"):
                                                    print(f"turn {turn} agent {agent} answer: {answer}, device {device} line 935")

                                                del chat_for_identifying_answer
                                        else:

                                            answer = [
                                                        extract_ans_from_response(self.dataset_name, decoded_responses[q])
                                                        for q in range(len(decoded_responses))
                                                    ]
                                            
                                            # if device == torch.device("cuda:0"):
                                            #     print("answer:", answer, "line 935")
                                            
            
                                            
                                        # Modify the responses by adding the boxed answer only if it doesn't already exist
                                        for q, decoded_response in enumerate(decoded_responses):
                                            final_answer_str = f"Answer: \\boxed{{{answer[q]}}}"
                                            modified_response = decoded_response
                                            if f"The answer is: {answer[q]}" in decoded_response:
                                                # Replace any occurrence of the unboxed answer with the new boxed answer
                                                modified_response = modified_response.replace(f"The answer is: {answer[q]}", final_answer_str)
                                            
                                            
                                            if f"The answer is: \\boxed{{{answer[q]}}}" in decoded_response:
                                                # Replace any occurrence of the unboxed answer with the new boxed answer
                                                modified_response = modified_response.replace(f"The answer is: \\boxed{{{answer[q]}}}", final_answer_str)
                                            
                                
                                            if f"Answer: {answer[q]}" in decoded_response:
                                                # Replace any occurrence of the unboxed answer with the new boxed answer
                                                modified_response = modified_response.replace(f"Answer: {answer[q]}", final_answer_str)
                                            
                                            length_to_check = int(1.5 * len(final_answer_str))
                                                                                    
                                            # Ensure that the response is at least as long as the portion we're checking
                                            if len(modified_response) >= length_to_check:
                                                # Check in the last 1.5 times the length of final_answer_str
                                                response_to_check = modified_response[-length_to_check:]
                                            else:
                                                # If the response is shorter, check the entire response
                                                response_to_check = modified_response
                                            
                                            # Append final_answer_str if it's not found in the response_to_check
                                            if answer[q] != "invalid":                                                 
                                                if final_answer_str not in response_to_check:
                                                    modified_response = re.sub(r"\\boxed{(.*?)}", r"\1", modified_response)
                                                    if modified_response.endswith(f"\n{answer[q]}"):
                                                        modified_response = modified_response.rstrip(f"\n{answer[q]}")
                                                    if modified_response.endswith("\n"):
                                                        modified_response = modified_response.rstrip("\n")
                                                    modified_response = f"{modified_response}" + "\n\n" + f"{final_answer_str}"
                                            
                                            modified_responses.append(modified_response)    
                                        del decoded_responses  
                                        encoded_tensors = [torch.tensor(current_tokenizer.encode(modified_response, add_special_tokens=False)) for modified_response in modified_responses]
                                    else:
                                        encoded_tensors = postprocessed_response
                                        modified_responses = [current_tokenizer.decode(encoded_tensor, skip_special_tokens=True) for encoded_tensor in encoded_tensors]
                                    # Determine the maximum length

                                    max_length = max([tensor.size(0) for tensor in encoded_tensors])
                                    encoded_end_token = current_tokenizer.encode("<|end|>", add_special_tokens=False)

                                    # Determine the end_token_id based on the length of the encoded token
                                    if len(encoded_end_token) == 1:
                                        end_token_id = encoded_end_token[0]
                                    else:
                                        end_token_id = current_tokenizer.eos_token_id

                                    # Pad all tensors to the maximum length
                                    padded_tensors = [
                                        torch.cat([tensor, torch.tensor([end_token_id] * (max_length - tensor.size(0)))])
                                        if tensor.size(0) < max_length
                                        else tensor
                                        for tensor in encoded_tensors
                                    ]

                                    # Stack the padded tensors into a single tensor
                                    postprocessed_response = torch.stack(padded_tensors)

                                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                                        # Encode the custom end token
                                        encoded_end_token = current_tokenizer.encode("<|end|>", add_special_tokens=False)

                                        # Determine the token list for truncation
                                        if len(encoded_end_token) == 1:
                                            truncation_tokens = [args.stop_token_id, encoded_end_token[0]]
                                        else:
                                            truncation_tokens = [args.stop_token_id]
                                        if current_model.policy.config._name_or_path == "Qwen/Qwen2.5-3B-Instruct":
                                            truncation_tokens.append(4710)

                                        # Apply truncation
                                        postprocessed_response = truncate_response(
                                            truncation_tokens,
                                            current_tokenizer.pad_token_id,
                                            postprocessed_response
                                        )

                                    postprocessed_response = postprocessed_response.to(device)
                                    
                                    del   encoded_tensors, padded_tensors
                                    query_responses.append(query_response)
                                   
                                    queries_for_score_batches = []

                                    for (ix, index) in enumerate(small_batch_non_finished_index):
                                        # Extract the raw agent context set for the current query
                                        current_raw_agent_context = raw_agent_context_set[agent][index][0]["content"]
                                        modified_response = modified_responses[ix]

                                        if isinstance(modified_response, torch.Tensor):
                                            modified_response = tokenizer.decode(modified_response, skip_special_tokens=True)
                                        
                                        # Remove the last token if it's <|im_end|>
                                        if modified_response.endswith("<|im_end|>"):
                                            modified_response = modified_response[:-len("<|im_end|>")].strip()

                                        # Remove the starting token if it's "model" or <|im_start|>assistant
                                        if modified_response.startswith("model"):
                                            modified_response = modified_response[len("model"):].strip()
                                        if modified_response.startswith("<|im_start|>"):
                                            modified_response = modified_response[len("<|im_start|>"):].strip()                                          
                                        if modified_response.startswith("assistant"):
                                            modified_response = modified_response[len("assistant"):].strip()
                                        if modified_response.startswith("<|im_start|>assistant"):
                                            modified_response = modified_response[len("<|im_start|>assistant"):].strip()
                                        if extract_ans_from_response(self.dataset_name, modified_response) != "invalid":
                                            modified_response = re.sub(r'[^a-zA-Z}]+$', '', modified_response)

                                        
                                        
                                        query_for_score = [
                                            {"role": 'user', "content": current_raw_agent_context},
                                            {"role": "assistant", "content": modified_response}
                                        ]
                                        
                                        queries_for_score_batches.append(query_for_score)


                                    correctness_for_training = calculate_reward_from_answer(self.dataset_name, queries_for_score_batches, answer_text_set_small_batch).to(device)
                                    if self.reward_feedback:
                                        score = reward_from_different_server(queries_for_score_batches, self.server_dict[current_tokenizer.init_kwargs["name_or_path"]], self.reward_feedback).to(device)
                                    else:                                       
                                        score = correctness_for_training.to(dtype=torch.float)
                                    score_for_training = score.clone()
                                    

                                    ## for_score_query_response,make answer_gen with added question - if answer is not providing reason, but just providing that answer without refering the question, then it should be penalized. For example, if you do natural language inference, at least they need to provide some reason, not simply saying the relationship between the question and the hypothesis. 
                                    ## we will ask it to language model to do it.    For example,   answer_gen, _  = generate(unwrapped_model.policy, XXX(it will be some prompt with for_score_query_response), tokenizer.pad_token_id,  generation_config_for_answer,)
                                
                                    if self.score_penalty_for_nonvalid:

                                        penalize = []
                                        chat_for_identifying_reasoning = chat_for_no_reasoning_check(postprocessed_response, current_tokenizer, self.dataset_name).to(device)

                                        chat_context_length = chat_for_identifying_reasoning.shape[1]
                                        retry = 5
                                        answer_best = ["invalid"] * len(small_batch_non_finished_index)  # Track the best answers
                                        invalid_count_best = len(small_batch_non_finished_index) + 1
                                        invalid_indices = list(range(len(small_batch_non_finished_index)))  # Track invalid indices

                                        while retry > 0 and len(invalid_indices) > 0:
                                            # Only re-check invalid parts
                                            chat_for_invalid = chat_for_identifying_reasoning[invalid_indices]
                                            
                                            # Generate answers for invalid parts
                                            current_model.policy.set_adapter("ref")
                                            answer_gen, _ = generate(current_model.policy, chat_for_invalid, current_tokenizer.pad_token_id, generation_config_dict["gen_answer"],)
                                            
                                            if device == torch.device("cuda:0"):
                                                answer_gen_decoded = [current_tokenizer.decode(answer_gen[q, chat_context_length:], skip_special_tokens=True) for q in range(answer_gen.shape[0])]
                                                postprocessed_response_decoded = [current_tokenizer.decode(postprocessed_response[q], skip_special_tokens=True) for q in invalid_indices]
                                            answer_gen = answer_gen[:, chat_context_length:]
                                            reasoning_answer = [
                                                extract_ans_from_response_reasoning(current_tokenizer.decode(answer_gen[q], skip_special_tokens=True))
                                                for q in range(answer_gen.shape[0])
                                            ]
                                            if device == torch.device("cuda:0"):
                                                print("answer_gen:", postprocessed_response_decoded, answer_gen_decoded, reasoning_answer)
                                            
                                            
                                            # Track new invalid indices and update valid answers
                                            new_invalid_indices = []
                                            for idx, ans in enumerate(reasoning_answer):
                                                global_idx = invalid_indices[idx]  # Map back to the original index
                                                if ans == -1 or ans == 1:
                                                    answer_best[global_idx] = ans  # Update the best answer for this index
                                                else:
                                                    new_invalid_indices.append(global_idx)  # Still invalid, track it for retry
                                                    
                                            invalid_count = len(new_invalid_indices)
                                            
                                            if invalid_count < invalid_count_best:
                                                invalid_count_best = invalid_count
                                            
                                            # Update invalid indices for the next retry
                                            invalid_indices = new_invalid_indices

                                            if invalid_count != 0:
                                                retry -= 1
                                            else:
                                                break
                                            del answer_gen

                                        penalize = [ans == "invalid" for ans in answer_best]

                                        score_with_penalty = score.clone()
                                        score_for_training_with_penalty = score_for_training.clone()
                                        for q in range(len(score_with_penalty)):
                                            if penalize[q] == 1:
                                                score_with_penalty[q] = 0
                                                score_for_training_with_penalty[q] = 0
                                        
                                        del answer_best, reasoning_answer, chat_for_identifying_reasoning

                                    if self.reward_feedback:
                                        for q in range(len(small_batch_non_finished_index)):
                                            agent_reward_set[small_batch_non_finished_index[q]][agent] = score[q].item()
                                    
                                    
                                            
                                    responses.append(response)
                                    logprobs.append(logprob)
                                    ref_logprobs.append(ref_logprob)
                                    values.append(value)
                                    sequence_lengths.append(sequence_length)
                                    scores.append(score)
                                    scores_for_training.append(score_for_training)
                                    correctnesses_for_training.append(correctness_for_training)

                                    if self.score_penalty_for_nonvalid:
                                        scores_with_penalty.append(score_with_penalty)
                                        scores_for_training_with_penalty.append(score_for_training_with_penalty)

                                    if not self.open_ended_answer:
                                        answers.append(answer)
                                    else:
                                        answer = [
                                            extract_ans_from_response(self.dataset_name, current_tokenizer.decode(postprocessed_response[q], skip_special_tokens=True))
                                            for q in range(postprocessed_response.shape[0])
                                        ]
                                        answers.append(answer)


                                    for q in range(i, min(i + args.local_rollout_forward_batch_size, queries.shape[0])):
                                        if q in small_batch_non_finished_index:
                                            idx = small_batch_non_finished_index.index(q)  # Find the index within the finished list
                                            try:
                                                text = current_tokenizer.decode(postprocessed_response[idx], skip_special_tokens=True)
                                            except:
                                                postprocessed_response[idx] = postprocessed_response[idx].to(torch.int)
                                                text = current_tokenizer.decode(postprocessed_response[idx], skip_special_tokens=True)
                                            assistant_message = construct_assistant_message(text)
                                            agent_contexts_set[q][agent].append(assistant_message)
                                            if self.reward_feedback:
                                                score_value = score[idx].item()

                                                if score_value < 0.3:
                                                    feedback = "Your answer is highly likely wrong."
                                                elif score_value < 0.6:
                                                    feedback = "Your answer might be wrong, or your reasoning needs to have a stronger argument."
                                                elif score_value < 0.8:
                                                    feedback = "Your answer seems right, but check your reasoning again, there might be some room for improvement."
                                                else:
                                                    feedback = "Your answer is likely right with high probability."

                                                agent_contexts_set[q][agent].append({'role': 'user', 'content': f"Reward from a verifier of your answer: {score_value:.3f} out of 1.0, which means {feedback}"})
                                            del text, assistant_message

                                        else:
                                            # Append a message indicating that consensus has already been approached
                                            agent_contexts_set[q][agent].append({'role': 'assistant', 'content': "Consensus has already been reached."})
                                            if self.reward_feedback:
                                                agent_contexts_set[q][agent].append({'role': 'user', 'content': f"No reward since consensus has already been reached." })

                                    del (query, original_query, query_response, response, logprob, 
                                            ref_logprob, value, score, sequence_length, postprocessed_response,
                                            for_score_query_response, full_value, unwrapped_model_for_value)
                            query_responses = torch.cat(query_responses, 0)
                            query_responses_no_history = torch.cat(query_responses_no_history, 0)
                            responses = torch.cat(responses, 0)
                            postprocessed_responses = torch.cat(postprocessed_responses, 0)
                            logprobs = torch.cat(logprobs, 0)
                            ref_logprobs = torch.cat(ref_logprobs, 0)
                            values = torch.cat(values, 0)
                            sequence_lengths = torch.cat(sequence_lengths, 0)
                            scores = torch.cat(scores, 0)
                            scores_for_training = torch.cat(scores_for_training, 0)
                            correctnesses_for_training = torch.cat(correctnesses_for_training, 0)

                            if self.score_penalty_for_nonvalid:
                                scores_with_penalty = torch.cat(scores_with_penalty, 0)
                                scores_for_training_with_penalty = torch.cat(scores_for_training_with_penalty, 0)
                            ## answers is list's list, but I wanna make it just a list
                            answers = [item for sublist in answers for item in sublist]
                            for ans in answers:
                                answers_all_agent[agent].append(ans)   
                            rewards_all_agent[agent].append(scores.clone())

                            ending_tokens = ["<|end|>"]
                           
                            sequence_len = torch.tensor([
                                first_true_indices(postprocessed_responses[q] == current_tokenizer.pad_token_id) - 1
                                for q in range(postprocessed_responses.size(0))
                            ])  
                            sufficient_length = (sequence_len >= self.min_length).to(device)
                            no_invalid_repetitions = torch.ones_like(sufficient_length, dtype=torch.bool, device = device)
                            for iii in range(postprocessed_responses.shape[0]):
                                rep_check_seq = postprocessed_responses[iii][:sequence_len[iii]]
                                if check_repeated_sequences(current_tokenizer, rep_check_seq):
                                    no_invalid_repetitions[iii] = False                                
                                

                            pattern_box = re.compile(r"\\boxed{.*?}")

                            # Decode each response and check if the pattern matches
                            contain_box_pattern = [
                                bool(pattern_box.search(current_tokenizer.decode(response.tolist())))
                                for response in postprocessed_responses
                            ]

                            # Create the penalty_condition based on the matches
                            contain_box_sequence = torch.tensor(contain_box_pattern, dtype=torch.bool, device=device)


                            if self.non_eos_penalty:
                                penalty_condition = sufficient_length & no_invalid_repetitions
                                
                                if self.task_training or (turn != 0):
                                    penalty_condition = penalty_condition & contain_box_sequence
                                    
                            penalty_value = torch.full_like(scores, self.penalty_reward_value)

                            scores = torch.where(
                                penalty_condition, 
                                scores_for_training_with_penalty if self.score_penalty_for_nonvalid else scores_for_training, 
                                penalty_value
                            )
                            sequence_lengths_p1 = sequence_lengths + 1
                            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                            values = torch.masked_fill(values, padding_mask_p1, 0)

                            query_responses_turn_agent[(turn, agent)] = query_responses.to("cpu").clone()
                            query_responses_no_history_turn_agent[(turn, agent)] = query_responses_no_history.to("cpu").clone()
                            responses_turn_agent[(turn, agent)] = responses.to("cpu").clone()
                            logprobs_turn_agent[(turn, agent)] = logprobs.to("cpu").clone()
                            ref_logprobs_turn_agent[(turn, agent)] = ref_logprobs.to("cpu").clone()
                            values_turn_agent[(turn, agent)] = values.to("cpu").clone()
                            sequence_lengths_turn_agent[(turn, agent)] = sequence_lengths.to("cpu").clone()
                            scores_turn_agent[(turn, agent)] = scores.to("cpu").clone()
                            correctnesses_turn_agent[(turn, agent)] = correctnesses_for_training.to("cpu").clone()
                            sequence_lengths_p1_turn_agent[(turn, agent)] = sequence_lengths_p1.to("cpu").clone()
                            padding_mask_turn_agent[(turn, agent)] = padding_mask.to("cpu").clone()
                            padding_mask_p1_turn_agent[(turn, agent)] = padding_mask_p1.to("cpu").clone()
                            del (query_responses, responses, postprocessed_responses, logprobs, ref_logprobs, 
                                values, sequence_lengths, scores, context_length, sequence_lengths_p1, 
                                padding_mask, padding_mask_p1,  queries, response_idxs, query_responses_no_history)                           
                        answers_all_agent_turn[turn] = answers_all_agent
                        rewards_all_agent_turn[turn] = rewards_all_agent
                        q_to_unfinished_index = {q: idx for idx, q in enumerate(i for i in range(len(finished_question)) if finished_question[i] == -1)}

                        for q in range(len(question_text_set)):
                            if finished_question[q] == -1:
                                q_index_in_unfinished = q_to_unfinished_index[q]
                                
                                consensus = check_agent_consensus(
                                    [answers_all_agent_turn[turn][agent][q_index_in_unfinished] for agent in range(self.agent_num)],
                                    rewards_all_agent,
                                    self.criteria_for_consensus_percentage,
                                    self.criteria_for_consensus_reward_threshold,
                                    q_index_in_unfinished
                                )
                                # if device == torch.device("cuda:0"):
                                #     print("line 1020: ", consensus)
                                if consensus and update > self.update_initial_turn:
                                    finished_question[q] = turn        

                        unfinished_question_index = [index for index, value in enumerate(finished_question) if value == -1]



                if update == 0:
                    # Use gather_object for non-tensor data
                    all_agent_contexts = gather_object(agent_contexts_set)
                    all_agent_rewards = gather_object(agent_reward_set)
                    all_answer_text_set = gather_object(answer_text_set)
                    all_answer_only = gather_object(answers_all_agent_turn)
                    all_correctnesses = {} 
                    for agent in range(self.agent_num):
                        for turn in range(self.round_num):
                            all_correctnesses[(turn, agent)] = gather_object(correctnesses_turn_agent[(turn, agent)])
                    all_questions = gather_object(question_text_set)
                    def get_unique_model_name(base_name, existing_names):
                        if base_name not in existing_names:
                            return base_name
                        
                        counter = 1
                        while f"{base_name}-{counter}" in existing_names:
                            counter += 1
                        return f"{base_name}-{counter}"

                    model_name = [self.tokenizer.init_kwargs["name_or_path"]]

                    if self.diff_model_training:
                        for another in another_tokenizer:
                            base_name = another.init_kwargs["name_or_path"]
                            unique_name = get_unique_model_name(base_name, model_name)
                            model_name.append(unique_name)
                    else:
                        base_name = self.tokenizer.init_kwargs["name_or_path"]
                        for j in range(self.agent_num - 1):
                            unique_name = get_unique_model_name(base_name, model_name)
                            model_name.append(unique_name)
                    if main_accelerator.is_main_process:
                        data = {} 
                        for idx, question in enumerate(all_agent_contexts):
                            data[all_questions[idx]] = {}
                            for agent in range(len(model_name)):
                                data[all_questions[idx]][model_name[agent]] = {}
                                print(int(len(all_agent_contexts[0][0])/3))
                                for turn in range(int(len(all_agent_contexts[0][0])/3)):
                                    data[all_questions[idx]][model_name[agent]][f"turn_{turn}"] = {}
                                    data[all_questions[idx]][model_name[agent]][f"turn_{turn}"]["response"] = all_agent_contexts[idx][agent][turn*3 + 1]["content"]
                                    data[all_questions[idx]][model_name[agent]][f"turn_{turn}"]["reward"] = extract_reward(all_agent_contexts[idx][agent][turn*3 + 2]["content"]) 
                                    data[all_questions[idx]][model_name[agent]][f"turn_{turn}"]["correct"] = all_correctnesses[(turn, agent)][idx].item()
                                    data[all_questions[idx]][model_name[agent]][f"turn_{turn}"]["answer"] = extract_ans_from_response(self.dataset_name, data[all_questions[idx]][model_name[agent]][f"turn_{turn}"]["response"] )
                            data[all_questions[idx]]["answer"] = all_answer_text_set[idx]
                            
                        with open(f"{args.output_dir}/data.json", "w") as f:
                            json.dump(data, f, indent=4)
                if self.diff_model_training:
                    del unwrapped_other_models
                torch.cuda.empty_cache()
                gc.collect()
                scores_tmp = {}  # Initialize a temporary dictionary to store the filtered scores
                for turn in range(self.round_num):
                    for agent in range(self.agent_num):
                        scores = score_rule(self.rule_horizon, self.rule_agent_share, self.rule_discount, scores_turn_agent, self.round_num, self.agent_num, finished_question, turn, agent, incentive= True, alpha=self.alpha, device=str(device))
                        bonues = bonus_rule(correctnesses_turn_agent, self.round_num, self.agent_num, finished_question, turn, agent, incentive= False, alpha=self.alpha, device=str(device))
                        # Filter out rewards that are -1 (indicating the turn was not approached)
                        scores_bonuses = scores + bonues
                        filtered_scores = scores_bonuses[scores != -1]
                        scores_tmp[(turn, agent)] = filtered_scores.to("cpu").clone()
                        del scores, bonues, scores_bonuses, filtered_scores
                # Update the original scores_turn_agent with the filtered scores
                scores_turn_agent = scores_tmp
                del scores_tmp
                for turn in range(self.round_num):
                    for agent in range(self.agent_num):
                        scores = scores_turn_agent[(turn, agent)].to(device)
                        if len(scores) == 0:
                            continue
                        responses = responses_turn_agent[(turn, agent)].to(device)
                        logprobs = logprobs_turn_agent[(turn, agent)].to(device)
                        ref_logprobs = ref_logprobs_turn_agent[(turn, agent)].to(device)
                        values = values_turn_agent[(turn, agent)].to(device)
                        sequence_lengths = sequence_lengths_turn_agent[(turn, agent)].to(device)
                        sequence_lengths_p1 = sequence_lengths_p1_turn_agent[(turn, agent)].to(device)
                        padding_mask = padding_mask_turn_agent[(turn, agent)].to(device)
                        padding_mask_p1 = padding_mask_p1_turn_agent[(turn, agent)].to(device)
                        kl = logprobs - ref_logprobs
                        non_score_reward = -args.kl_coef * kl
                        rewards = non_score_reward.clone()
                        actual_start = torch.arange(rewards.size(0), device=rewards.device)
                        actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                        actual_end = torch.clamp(actual_end, 0, rewards.size(1) - 1)  # Clamp indices to valid range

                        try:
                            rewards[actual_start, actual_end] += scores
                        except Exception as e:
                            print(f"Error updating rewards at indices: {e}")
                            raise

                        if args.whiten_rewards:
                            rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                            rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                        # 6. Compute advantages and returns
                        lastgaelam = 0
                        advantages_reversed = []
                        gen_length = responses.shape[1]

                        for t in reversed(range(gen_length)):
                            try:
                                # Safely get next values
                                if t < gen_length - 1:
                                    nextvalues = values[:, t + 1]
                                else:
                                    nextvalues = torch.zeros_like(values[:, 0], device=values.device)  # Create tensor of zeros with proper shape
                                # Ensure valid index range
                                assert 0 <= t < gen_length, f"Index t={t} is out of bounds for gen_length={gen_length}"

                                # Calculate delta with pre-checks for invalid values
                                delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                                assert not torch.isnan(delta).any(), f"NaN detected in delta at step {t}"
                                assert not torch.isinf(delta).any(), f"Infinity detected in delta at step {t}"

                                # Log delta stats
                                # print(f"delta - min: {delta.min()}, max: {delta.max()}, NaN count: {torch.isnan(delta).sum()}, Inf count: {torch.isinf(delta).sum()}")

                                # Update advantages
                                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                                assert not torch.isnan(lastgaelam).any(), f"NaN detected in lastgaelam at step {t}"
                                assert not torch.isinf(lastgaelam).any(), f"Infinity detected in lastgaelam at step {t}"
                                advantages_reversed.append(lastgaelam)


                            except Exception as e:
                                print(f"Unexpected exception at t={t}: {e}")
                                raise

                        # Reverse the advantages list to match the original order
                        advantages = torch.stack(advantages_reversed[::-1], axis = 1)

                        try:
                            returns = advantages + values
                        except Exception as e:
                            print(f"Error calculating returns: {e} in device {device}")
                            print(f"advantages shape: {advantages.shape}, values shape: {values.shape}")
                            raise
                        advantages = masked_whiten(advantages, ~padding_mask)
                        advantages = torch.masked_fill(advantages, padding_mask, 0)
                        torch.cuda.empty_cache()
                        ### need to change - scofres_turn_agent duplicated. 
                        rewards_turn_agent[(turn, agent)] = rewards.to("cpu").clone()
                        advantages_turn_agent[(turn, agent)] = advantages.to("cpu").clone()
                        returns_turn_agent[(turn, agent)] = returns.to("cpu").clone()
                        kl_turn_agent[(turn, agent)] = kl.to("cpu").clone()
                        non_score_reward_turn_agent[(turn, agent)] = non_score_reward.to("cpu").clone()
                        del (responses, logprobs, ref_logprobs, values, sequence_lengths, sequence_lengths_p1,
                             padding_mask, padding_mask_p1, scores, rewards, actual_start, 
                             actual_end, advantages, returns, kl, non_score_reward, lastgaelam,
                             advantages_reversed, gen_length, nextvalues, delta, )
                        torch.cuda.empty_cache()
                        gc.collect()
                del (ref_logprobs_turn_agent, sequence_lengths_turn_agent, sequence_lengths_p1_turn_agent)
                torch.cuda.empty_cache()
                gc.collect()
            grad_requires_setting(model, self.task_training, self.policy_separation, self.collaboration_separation)
            if self.diff_model_training:
                for another_model_ind in another_model:
                    grad_requires_setting(another_model_ind, self.task_training, self.policy_separation, self.collaboration_separation)

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                if device == torch.device("cuda:0"):
                    print("=============================")
                    print(f"ppo_epoch_idx number is : {ppo_epoch_idx}")
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps, self.round_num)
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with ExitStack() as stack:
                            stack.enter_context(main_accelerator.accumulate(model))
                            if self.diff_model_training:
                                for idx, another_model_ind in enumerate(another_model):
                                    stack.enter_context(self.accelerator[idx + 1].accumulate(another_model_ind))

                            # grad_check_lora(model, self.task_training)
                            loss_dict = {}
                            output_dict = {}
                            vpred_temp_dict = {}
                            logits_dict = {}
                            new_all_logprobs_dict = {}
                            new_logprobs_dict = {}
                            vpred_dict = {}
                            vpred_temp_dict = {}
                            vpredclipped_dict = {}
                            vf_losses1_dict = {}
                            vf_losses2_dict = {}
                            vf_loss_max_dict = {}
                            vf_loss_dict = {}
                            vf_clipfrac_dict = {}
                            logprobs_diff_dict = {}
                            ratio_dict = {}
                            pg_losses_dict = {}
                            pg_losses2_dict = {}
                            pg_loss_max_dict = {}
                            pg_loss_dict = {}
                            
                            for turn in range(self.round_num):
                                for agent in range(self.agent_num):
                                    if self.diff_model_training:
                                        current_model = model if agent == 0 else another_model[agent-1]
                                        current_tokenizer = tokenizer if agent == 0 else another_tokenizer[agent-1]
                                    else:
                                        current_model = model
                                        current_tokenizer = tokenizer
                                
                                    if (self.turn_based_training):
                                        # if update_turn is integer, check turn != update_turn. If update_turn is list, check turn not in update_turn
                                        if isinstance(update_turn, int):
                                            if (turn != update_turn):
                                                context_manager = torch.no_grad()
                                            else:
                                                context_manager = nullcontext()
                                        else:
                                            if turn not in update_turn:
                                                context_manager = torch.no_grad()
                                            else:
                                                context_manager = nullcontext()
                                    else:
                                        context_manager = nullcontext()

                                    with context_manager:
                                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                                        valid_indices = [idx for idx, val in enumerate(finished_question) if val == -1 or val >= turn]
                                        filtered_micro_batch_inds = [idx for idx in micro_batch_inds if idx in valid_indices]
                                        micro_batch_inds = [valid_indices.index(idx) for idx in filtered_micro_batch_inds]
                                        if len(micro_batch_inds) == 0:
                                            #### fake means that actually there is an error when micro_batch_inds is empty.  -- so I just used the bypass 
                                            micro_batch_inds = [i for i in range(args.per_device_train_batch_size)]
                                            fake = 1
                                            actual_turn = 0
                                        else:
                                            fake = 0
                                            actual_turn = turn
                                        count_samples_stats[minibatch_idx, turn]+= len(micro_batch_inds)
                                        query_responses = query_responses_turn_agent[(actual_turn, agent)]
                                        query_responses_no_history = query_responses_no_history_turn_agent[(actual_turn, agent)]
                                        responses = responses_turn_agent[(actual_turn, agent)]
                                        logprobs = logprobs_turn_agent[(actual_turn, agent)]
                                        values = values_turn_agent[(actual_turn, agent)]
                                        advantages = advantages_turn_agent[(actual_turn, agent)]
                                        returns = returns_turn_agent[(actual_turn, agent)]
                                        padding_mask = padding_mask_turn_agent[(actual_turn, agent)].clone().to(device)
                                        padding_mask_p1 = padding_mask_p1_turn_agent[(actual_turn, agent)].clone().to(device)
                                        context_length = context_length_turn_agent[(actual_turn, agent)]
                                        mb_return = returns[micro_batch_inds].to(device)
                                        mb_advantage = advantages[micro_batch_inds].to(device)
                                        mb_values = values[micro_batch_inds].to(device)
                                        mb_responses = responses[micro_batch_inds].to(device)
                                        mb_query_responses = query_responses[micro_batch_inds].to(device)
                                        mb_query_responses_no_history = query_responses_no_history[micro_batch_inds].to(device)
                                        mb_logprobs = logprobs[micro_batch_inds].to(device)
                                        ref = False
                                        output_dict[(turn, agent)] = forward(current_model.policy, ref, turn, mb_query_responses, current_tokenizer.pad_token_id, self.round_num, self.policy_separation, self.collaboration_separation, self.task_training)
                                        if self.value_simplification:
                                            vpred_temp_dict[(turn, agent)] = get_reward(
                                                current_model, ref, turn, mb_query_responses_no_history, current_tokenizer.pad_token_id, self.task_training
                                            )
                                        else:
                                            vpred_temp_dict[(turn, agent)] = get_reward(
                                                current_model, ref, turn, mb_query_responses, current_tokenizer.pad_token_id, self.task_training
                                            )

                                        logits_dict[(turn, agent)] = output_dict[(turn, agent)].logits[:, context_length - 1 : -1]
                                        logits_dict[(turn, agent)] /= args.temperature + 1e-7
                                        new_all_logprobs_dict[(turn, agent)] = F.log_softmax(logits_dict[(turn, agent)], dim=-1)
                                        new_logprobs_dict[(turn, agent)] = torch.gather(new_all_logprobs_dict[(turn, agent)], 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                                        new_logprobs_dict[(turn, agent)] = torch.masked_fill(
                                            new_logprobs_dict[(turn, agent)], padding_mask[micro_batch_inds], INVALID_LOGPROB
                                        )
                                        if self.value_simplification:
                                            vpred_dict[(turn, agent)] = vpred_temp_dict[(turn, agent)][:, context_length_turn_agent[(0, agent)] - 1 : -1].squeeze(-1)
                                        else:
                                            vpred_dict[(turn, agent)] = vpred_temp_dict[(turn, agent)][:, context_length_turn_agent[(turn, agent)] - 1 : -1].squeeze(-1)

                                        vpred_dict[(turn, agent)] = torch.masked_fill(vpred_dict[(turn, agent)], padding_mask_p1[micro_batch_inds], 0)
                                        vpredclipped_dict[(turn, agent)] = torch.clamp(
                                            vpred_dict[(turn, agent)],
                                            mb_values - args.cliprange_value,
                                            mb_values + args.cliprange_value,
                                        )

                                        vf_losses1_dict[(turn, agent)] = torch.square(vpred_dict[(turn, agent)] - mb_return)
                                        vf_losses2_dict[(turn, agent)] = torch.square(vpredclipped_dict[(turn, agent)] - mb_return)
                                        vf_loss_max_dict[(turn, agent)] = torch.max(vf_losses1_dict[(turn, agent)], vf_losses2_dict[(turn, agent)])
                                        vf_loss_dict[(turn, agent)] = 0.5 * masked_mean(vf_loss_max_dict[(turn, agent)], ~padding_mask_p1[micro_batch_inds])
                                        vf_clipfrac_dict[(turn, agent)] = masked_mean(
                                            (vf_losses2_dict[(turn, agent)] > vf_losses1_dict[(turn, agent)]).float(), ~padding_mask_p1[micro_batch_inds]
                                        )
                                        logprobs_diff_dict[(turn, agent)] = new_logprobs_dict[(turn, agent)] - mb_logprobs
                                        ratio_dict[(turn, agent)] = torch.exp(logprobs_diff_dict[(turn, agent)])
                                        pg_losses_dict[(turn, agent)] = -mb_advantage * ratio_dict[(turn, agent)]
                                        pg_losses2_dict[(turn, agent)] = -mb_advantage * torch.clamp(ratio_dict[(turn, agent)], 1.0 - args.cliprange, 1.0 + args.cliprange)
                                        pg_loss_max_dict[(turn, agent)] = torch.max(pg_losses_dict[(turn, agent)], pg_losses2_dict[(turn, agent)])
                                        pg_loss_dict[(turn, agent)] = masked_mean(pg_loss_max_dict[(turn, agent)], ~padding_mask[micro_batch_inds])
                                        

                                        loss_dict[(turn, agent)] = (pg_loss_dict[(turn, agent)] + args.vf_coef * vf_loss_dict[(turn, agent)]) * (1- fake)

                                                                                    
                                        with torch.no_grad():
                                            if not fake:
                                                pg_clipfrac = masked_mean(
                                                    (pg_losses2_dict[(turn, agent)] > pg_losses_dict[(turn, agent)]).float(), ~padding_mask[micro_batch_inds]
                                                )
                                                prob_dist = torch.nn.functional.softmax(logits_dict[(turn, agent)], dim=-1)
                                                entropy = torch.logsumexp(logits_dict[(turn, agent)], dim=-1) - torch.sum(prob_dist * logits_dict[(turn, agent)], dim=-1)
                                                approxkl = 0.5 * (logprobs_diff_dict[(turn, agent)]**2).mean()
                                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn] += approxkl / self.agent_num * len(micro_batch_inds)
                                                pg_clipfrac_stats[
                                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn
                                                ] += pg_clipfrac / self.agent_num * len(micro_batch_inds)
                                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn] += pg_loss_dict[(turn, agent)] / self.agent_num* len(micro_batch_inds)
                                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn] += vf_loss_dict[(turn, agent)] / self.agent_num* len(micro_batch_inds)
                                                vf_clipfrac_stats[
                                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx , turn
                                                ] += vf_clipfrac_dict[(turn, agent)] / self.agent_num* len(micro_batch_inds)
                                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn] += entropy.mean() / self.agent_num* len(micro_batch_inds)
                                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx, turn] += ratio_dict[(turn, agent)].mean() / self.agent_num* len(micro_batch_inds)
                                                del approxkl, entropy, pg_clipfrac, prob_dist
                                        del (context_length, query_responses, responses, logprobs, 
                                            values, advantages, returns, padding_mask, padding_mask_p1,
                                            mb_return, mb_advantage, mb_values, 
                                            mb_responses, mb_query_responses, mb_query_responses_no_history ,mb_logprobs, 
                                                micro_batch_end, micro_batch_inds)

                                        if not self.diff_model_training:
                                            current_accelerator = main_accelerator
                                            current_optimizer = main_optimizer
                                        else:
                                            if agent == 0:
                                                current_accelerator = main_accelerator
                                                current_optimizer = main_optimizer
                                            else:
                                                current_accelerator = self.accelerator[agent]
                                                current_optimizer = another_optimizer[agent-1]
                                        should_update = True
                                        if self.turn_based_training:
                                            if isinstance(update_turn, int):
                                                should_update = (turn == update_turn)
                                            elif isinstance(update_turn, list):
                                                should_update = (turn in update_turn)
                                        if should_update:
                                            current_accelerator.backward(loss_dict[(turn, agent)])
                                            current_optimizer.step()
                                            current_optimizer.zero_grad()
                                        else:
                                            current_optimizer.zero_grad()

                                        del (
                                            output_dict[(turn, agent)], vpred_temp_dict[(turn, agent)], logits_dict[(turn, agent)], new_all_logprobs_dict[(turn, agent)], new_logprobs_dict[(turn, agent)], vpred_dict[(turn, agent)], vpredclipped_dict[(turn, agent)],
                                            vf_losses1_dict[(turn, agent)], vf_losses2_dict[(turn, agent)], vf_loss_max_dict[(turn, agent)], vf_loss_dict[(turn, agent)], vf_clipfrac_dict[(turn, agent)], logprobs_diff_dict[(turn, agent)], ratio_dict[(turn, agent)], 
                                            pg_losses_dict[(turn, agent)], pg_losses2_dict[(turn, agent)], pg_loss_max_dict[(turn, agent)], pg_loss_dict[(turn, agent)], loss_dict[(turn, agent)]
                                        )
                                        # accelerator.wait_for_everyone()

 
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off

                    del (
                        output_dict, vpred_temp_dict, logits_dict, new_all_logprobs_dict, new_logprobs_dict, vpred_dict, vpredclipped_dict,
                        vf_losses1_dict, vf_losses2_dict, vf_loss_max_dict, vf_loss_dict, vf_clipfrac_dict, logprobs_diff_dict, ratio_dict, 
                        pg_losses_dict, pg_losses2_dict, pg_loss_max_dict, pg_loss_dict, loss_dict, 
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            if device == torch.device("cuda:0"):
                print("LOGGING START")
            with torch.no_grad():
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["episode"] = global_step
                metrics["update"] = update
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]

                # Initialize accumulators for weighted sum
                total_valid_samples = count_samples_stats.sum(dim=0)  # Sum across minibatches for each turn
                if self.open_ended_answer:
                    answer_input_forstat = [answer_text_set, wrong_answer_text_set]
                else:
                    answer_input_forstat = answer_text_set
                correctness_turn_list, change_statistics_turn_list = stat_all(answer_input_forstat, answers_all_agent_turn, finished_question, self.open_ended_answer)
                for turn in range(self.round_num):
                    if total_valid_samples[turn] > 0:
                        mean_kl = (
                            (approxkl_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_pg_clipfrac = (
                            (pg_clipfrac_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_pg_loss = (
                            (pg_loss_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_vf_loss = (
                            (vf_loss_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_vf_clipfrac = (
                            (vf_clipfrac_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_entropy = (
                            (entropy_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        mean_ratio = (
                            (ratio_stats[:, :, :, turn].sum(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        var_ratio = (
                            (ratio_stats[:, :, :, turn].var(dim=(0, 1, 2)) * count_samples_stats[:, turn].sum()).to("cuda") / total_valid_samples[turn]
                        )
                        # mean_non_score_reward = torch.stack([non_score_reward_turn_agent[(turn, agent)].sum(1).mean().to("cuda") for agent in range(self.agent_num)]).mean()

                        # rlhf_reward = mean_non_score_reward + scores_turn_agent[(turn, agent)].to("cuda").mean()

                        metrics[f"objective/kl_turn_{turn}"] = main_accelerator.gather(mean_kl).mean().item()
                        metrics[f"objective/entropy_turn_{turn}"] = main_accelerator.gather(mean_entropy).mean().item()
                        # metrics[f"objective/non_score_reward_turn_{turn}"] = main_accelerator.gather(mean_non_score_reward).mean().item()
                        # metrics[f"objective/rlhf_reward_turn_{turn}"] = main_accelerator.gather(rlhf_reward).mean().item()
                        metrics[f"objective/scores_turn_{turn}"] = main_accelerator.gather(torch.stack([scores_turn_agent[(turn, agent)].mean().to("cuda") for agent in range(self.agent_num)]).mean()).mean().item()

                        metrics[f"policy/approxkl_avg_turn_{turn}"] = main_accelerator.gather(mean_kl).mean().item()
                        metrics[f"policy/clipfrac_avg_turn_{turn}"] = main_accelerator.gather(mean_pg_clipfrac).mean().item()
                        metrics[f"loss/policy_avg_turn_{turn}"] = main_accelerator.gather(mean_pg_loss).mean().item()
                        metrics[f"loss/value_avg_turn_{turn}"] = main_accelerator.gather(mean_vf_loss).mean().item()
                        metrics[f"val/clipfrac_avg_turn_{turn}"] = main_accelerator.gather(mean_vf_clipfrac).mean().item()
                        metrics[f"policy/entropy_avg_turn_{turn}"] = main_accelerator.gather(mean_entropy).mean().item()
                        metrics[f"val/ratio_turn_{turn}"] = main_accelerator.gather(mean_ratio).mean().item()
                        metrics[f"val/ratio_var_turn_{turn}"] = main_accelerator.gather(var_ratio).mean().item()
                    else:
                        metrics[f"objective/kl_turn_{turn}"] = 0.0
                        metrics[f"objective/entropy_turn_{turn}"] = 0.0 
                        metrics[f"objective/non_score_reward_turn_{turn}"] = 0.0 
                        metrics[f"objective/rlhf_reward_turn_{turn}"] = 0.0 
                        metrics[f"objective/scores_turn_{turn}"] = 0.0 

                        # Set metrics to zero if there are no valid samples
                        metrics[f"policy/approxkl_avg_turn_{turn}"] = 0.0
                        metrics[f"policy/clipfrac_avg_turn_{turn}"] = 0.0
                        metrics[f"loss/policy_avg_turn_{turn}"] = 0.0
                        metrics[f"loss/value_avg_turn_{turn}"] = 0.0
                        metrics[f"val/clipfrac_avg_turn_{turn}"] = 0.0
                        metrics[f"policy/entropy_avg_turn_{turn}"] = 0.0
                        metrics[f"val/ratio_turn_{turn}"] = 0.0
                        metrics[f"val/ratio_var_turn_{turn}"] = 0.0


                    correctness_tensor = torch.tensor(
                            correctness_turn_list[turn],    
                            device="cuda"
                        )
                    correctness_tensor = correctness_tensor[correctness_tensor != -1]
                    correctness_sum = correctness_tensor.sum()
                    correctness_count = correctness_tensor.size(0) 
                    gathered_tensor = main_accelerator.gather(correctness_sum)
                    gathered_tensor[gathered_tensor > 10000] = 0
                    gathered_correctness_sum = gathered_tensor.sum().item()
                    gathered_correctness_count = main_accelerator.gather(torch.tensor(correctness_count, device="cuda")).sum().item()
                    if gathered_correctness_count > 0:
                        metrics[f"correctness_turn_{turn}"] = gathered_correctness_sum / gathered_correctness_count
                    else:
                        metrics[f"correctness_turn_{turn}"] = 0.0
                    if turn <= self.round_num - 2:
                        change_statistics = change_statistics_turn_list[turn]
                        change_tensor = torch.tensor(change_statistics, device="cuda")  # Shape: (3,)
                        gathered_change_tensor = main_accelerator.gather(change_tensor)  # This might flatten the tensor
                        num_gpus = gathered_change_tensor.size(0) // 3
                        gathered_change_tensor = gathered_change_tensor.view(num_gpus, 3)  # Now the shape should be (num_gpus, 3)
                        total_change_tensor = gathered_change_tensor.sum(dim=0)  # Shape: (3,)
                        total_count = total_change_tensor.sum().item()
                        if total_count > 0:
                            improve_count = total_change_tensor[0].item() / total_count
                            decline_count = total_change_tensor[1].item() / total_count
                            no_change_count = total_change_tensor[2].item() / total_count
                            
                            metrics[f"change_ratio_improve_turn_{turn}"] = improve_count
                            metrics[f"change_ratio_decline_turn_{turn}"] = decline_count
                            metrics[f"change_ratio_no_change_turn_{turn}"] = no_change_count
                        else:
                            metrics[f"change_ratio_improve_turn_{turn}"] = 0.0
                            metrics[f"change_ratio_decline_turn_{turn}"] = 0.0
                            metrics[f"change_ratio_no_change_turn_{turn}"] = 0.0

                finished_question_stats = {}
                for t in range(-1, self.round_num):
                    finished_question_tensor = torch.tensor(finished_question)
                    count_t = torch.sum(finished_question_tensor == t).to(main_accelerator.device).to(dtype=torch.int32)
                    length = torch.tensor(len(finished_question), device=main_accelerator.device)
                    gathered_count_t = main_accelerator.gather(count_t).sum().item()
                    gathered_length = main_accelerator.gather(length).sum().item()
                    finished_question_stats[f"finished_question_ratio_turn_{t}"] = gathered_count_t /gathered_length # Normalize by total number of questions
                
                metrics.update(finished_question_stats)

                # save metrics manually on 
                if main_accelerator.is_main_process:
                    metrics_file = f"{args.output_dir}/metrics.json"
                    
                    # Load existing metrics if the file exists
                    if os.path.exists(metrics_file):
                        with open(metrics_file, "r") as f:
                            existing_metrics = json.load(f)
                    else:
                        existing_metrics = []

                    existing_metrics.append(metrics)
                    with open(metrics_file, "w") as f:
                        json.dump(existing_metrics, f, indent=4)
                    #save metrics manually on wandb
                    graph_gen_turn(args.output_dir)
                    # graph_gen_turn_specific(args.output_dir)

                # print(metrics)
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
            del (query_responses_turn_agent,
                query_responses_no_history_turn_agent,
                 responses_turn_agent,
                 context_length_turn_agent, 
                 logprobs_turn_agent,
                 values_turn_agent,
                 scores_turn_agent,
                 rewards_turn_agent, 
                 advantages_turn_agent,
                 returns_turn_agent,
                 padding_mask_turn_agent,
                 padding_mask_p1_turn_agent,
                 non_score_reward_turn_agent,
                 kl_turn_agent,
                 mean_kl, mean_entropy, metrics)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and update  % self.sample_generations_freq == 0:
                print("====================")
                print(update, "updates, saving the model. ")
                # self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
                self.save_model(f"{self.args.output_dir}/{update}")