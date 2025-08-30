### only for the rebuttal, so I think there is some part that we can improve more. 

from datetime import datetime
from peft import LoraConfig, LoftQConfig

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
server_dict = {
    "80gb-new" : "http://172.31.40.64:8501/predict",
    "40gb" : 'http://172.31.46.250:8501/predict',
    "80gb-old" : "http://172.31.15.201:8501/predict",
    "microsoft/Phi-3-mini-128k-instruct" : "http://172.31.15.201:8501/predict",
    "meta-llama/Meta-Llama-3-8B-Instruct" : "http://172.31.40.64:8501/predict",
    'meta-llama/Llama-3.1-8B-Instruct': "http://172.31.15.201:8501/predict",
    "meta-llama/Llama-3.2-3B": "http://172.31.15.201:8501/predict",
    "Qwen/Qwen2.5-3B-Instruct": "http://172.31.40.64:8501/predict",
}

# =============================================================================
# BASIC SETTINGS
# =============================================================================
random_seed = 42
debug = False

# =============================================================================
# DATASET AND TASK CONFIGURATION
# =============================================================================
dataset_name = "GSM8k"
if dataset_name == "ANLI":
    max_input_length = 350
    max_output_length = 300
elif dataset_name == "GSM8k":
    max_input_length = 200
    max_output_length = 300

min_output_length = 50

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Available base models
base_model_dict = {
    "llama3_8B_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi3": "microsoft/Phi-3-mini-128k-instruct",
    "gemma2-2b": "google/gemma-2-2b-it",
    "llama3.2-1B": "meta-llama/Llama-3.2-1B",
    "llama3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3B": "meta-llama/Llama-3.2-3B",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
}

base_model_name = "phi3"
base_model = base_model_dict[base_model_name]
sft_model_path = base_model
stop_token = 'eos'

# PEFT Configuration
peft = True
quantization = True
if quantization:
    peft_config = LoraConfig(
        r=8,
        loftq_config = LoftQConfig(loftq_bits=4),
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
else:
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

special_tokens = {'additional_special_tokens': ["<<", ">>"]}

# =============================================================================
# MULTI-AGENT CONFIGURATION
# =============================================================================
agent_num = 2
round_num = 2  # Note: This is actually a turn-2 config despite some references to round 3
turn_based_training = True
policy_separation = True
collaboration_separation = True
task_training = False

# Validation logic for training modes
if (not policy_separation) and (not task_training):
    print("error: policy_separation and task_training can't be False at the same time")
    exit()
if collaboration_separation and (not policy_separation):
    print("error: collaboration_separation can't be True while policy_separation is False")
    exit()

# =============================================================================
# DIFFERENTIAL MODEL TRAINING
# =============================================================================
diff_model_training = False
another_policy_base_model_name = ["Qwen2.5-3B"]
another_policy_base_model = [base_model_dict[am] for am in another_policy_base_model_name]
another_policy_sft_model_path = another_policy_base_model

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# PPO Parameters
vf_coef = 0.1
kl_coef = 0.002
num_ppo_epochs = 4

# Batch and Generation Parameters
num_mini_batches = 1  # Alternative: 6
total_episodes = None
num_sample_generations = 100
gradient_accumulation_steps = 4
per_device_train_batch_size = 1
local_rollout_forward_batch_size = 3  # Alternative: 12

# Optimizer Parameters
learning_rate = 1e-5
adafactor = True  # default: AdamW optimizer
adam_beta1 = 0.9
adam_beta2 = 0.95
warmup_steps = 10  # will change it to 10% of the total steps

# Precision Settings
half_precision_backend = True
bf16 = True
fp16 = False
load_in_8bit = True

early_stopping = False

# =============================================================================
# REWARD AND PENALTY CONFIGURATION
# =============================================================================
# Reward model settings
no_reward_model = True
reward_model_dir = "reward_phi3_gsm8k_lr_5e-05/checkpoint-6000"
reward_baseline = 0.0

# Penalty settings
non_eos_penalty = True
non_box_penalty = True
penalty_reward_value = -10
score_penalty_for_nonvalid = False

# Feedback settings
summary = False
reward_feedback = False
value_simplification = True
answer_detecting = False

# =============================================================================
# MULTI-AGENT REWARD RULES
# =============================================================================
rule_horizon = "discounted_sum"
rule_agent_share = "all"
rule_discount = 0.3

# =============================================================================
# EVALUATION AND LOGGING
# =============================================================================
eval_batch_size = 4
evaluation_strategy = "steps"
eval_steps = 300
eval_accumulation_steps = 1

logging_strategy = "steps"
logging_steps = 10

test_len = 100

save_strategy = "steps"
save_steps = 100

# =============================================================================
# CONSENSUS CRITERIA
# =============================================================================
criteria_for_consensus_percentage = 1.1
criteria_for_consensus_reward_threshold = 1.1

# =============================================================================
# OUTPUT AND REPORTING
# =============================================================================
output_dir = "training_gsm8k_2_3_noverifier"
reload_dir = output_dir
report_to = "wandb"
deepspeed = "config/deepspeed_config/ds_config_ppo_v1.json"

# Generate timestamp for run name
now = datetime.now()
current_time = now.strftime("%y%m%d%H%M")[2:]

run_name = f"ppo_{base_model_name}_{dataset_name}_lr_{learning_rate}_batchsize_{per_device_train_batch_size * 8}_kl_{kl_coef}_date_{current_time}_multi_agent_diff_model_{another_policy_base_model_name[0]}"

# =============================================================================
# CONFIGURATION DICTIONARY
# =============================================================================
config = {
    # Server and Reload Configuration
    "server_dict": server_dict,
    "reload_dir": reload_dir,

    # Multi-Agent Configuration
    "agent_num": agent_num,
    "round_num": round_num,
    "turn_based_training": turn_based_training,
    "policy_separation": policy_separation,
    "collaboration_separation": collaboration_separation,
    "task_training": task_training,

    # Reward Rules
    "rule_horizon": rule_horizon,
    "rule_agent_share": rule_agent_share,
    "rule_discount": rule_discount,

    # Training Settings
    "value_simplification": value_simplification,
    "num_mini_batches": num_mini_batches,
    "total_episodes": total_episodes,
    "local_rollout_forward_batch_size": local_rollout_forward_batch_size,
    "num_sample_generations": num_sample_generations,
    "num_ppo_epochs": num_ppo_epochs,

    # PPO Parameters
    "vf_coef": vf_coef,
    "kl_coef": kl_coef,
    "per_device_train_batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,

    # Dataset and Model
    "dataset_name": dataset_name,
    "base_model": base_model,
    "sft_model_path": sft_model_path,
    "stop_token": stop_token,

    # Model Configuration
    "max_input_length": max_input_length,
    "min_output_length": min_output_length,
    "max_output_length": max_output_length,
    "peft": peft,
    "peft_config": peft_config,
    "special_tokens": special_tokens,

    # Reward Configuration
    "no_reward_model": no_reward_model,
    "reward_model_dir": reward_model_dir,
    "reward_baseline": reward_baseline,
    "penalty_reward_value": penalty_reward_value,
    "non_eos_penalty": non_eos_penalty,
    "non_box_penalty": non_box_penalty,
    "score_penalty_for_nonvalid": score_penalty_for_nonvalid,

    # Feedback Settings
    "summary": summary,
    "reward_feedback": reward_feedback,
    "answer_detecting": answer_detecting,

    # Optimizer Configuration
    "learning_rate": learning_rate,
    "adafactor": adafactor,
    "early_stopping": early_stopping,
    "half_precision_backend": half_precision_backend,
    "fp16": fp16,
    "bf16": bf16,
    "load_in_8bit": load_in_8bit,
    "adam_beta1": adam_beta1,
    "adam_beta2": adam_beta2,
    "warmup_steps": warmup_steps,

    # Evaluation Configuration
    "eval_batch_size": eval_batch_size,
    "evaluation_strategy": evaluation_strategy,
    "eval_steps": eval_steps,
    "eval_accumulation_steps": eval_accumulation_steps,

    # Logging Configuration
    "logging_strategy": logging_strategy,
    "logging_steps": logging_steps,

    # Saving Configuration
    "save_strategy": save_strategy,
    "save_steps": save_steps,

    # Test Configuration
    "test_len": test_len,

    # Consensus Criteria
    "criteria_for_consensus_percentage": criteria_for_consensus_percentage,
    "criteria_for_consensus_reward_threshold": criteria_for_consensus_reward_threshold,

    # Output and Reporting
    "output_dir": output_dir,
    "report_to": report_to,
    "deepspeed": deepspeed,
    "run_name": run_name,

    # Differential Model Training
    "diff_model_training": diff_model_training,
    "another_policy_base_model_name": another_policy_base_model_name,
    "another_policy_base_model": another_policy_base_model,
    "another_policy_sft_model_path": another_policy_sft_model_path,

    # Basic Settings
    "random_seed": random_seed,
    "debug": debug,
}
