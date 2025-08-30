from datetime import datetime
from peft import LoraConfig
random_seed = 42 
debug = False
dataset_name = "GSM8k"
GPU_VISIBLE = "0,1,2,3,4,5,6,7"
gpu_server_num = 1
if gpu_server_num == 0 or gpu_server_num == 1:
    batch_size_per_gpu = 200
if gpu_server_num == 2:
    batch_size_per_gpu = 100
base_model_name = "qwen2_5_3B_instruct"

base_model = "Qwen/Qwen2.5-3B-Instruct"
if base_model_name == "llama3_8B_instruct":
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
if base_model_name == "phi3":
    base_model = "microsoft/Phi-3-mini-128k-instruct"
if base_model_name == "llama3_2_3B":
    base_model = "meta-llama/Llama-3.2-3B-Instruct"
sft_model = "sft_phi3_finetune_ANLI_lr_5e-05"
if dataset_name == "ANLI":
    max_input_length = 350
    max_output_length = 100
if dataset_name == "GSM8k":
    max_input_length = 200
    max_output_length = 300
if dataset_name == "FANTOM":
    max_input_length = 1800
    max_output_length = 200
if dataset_name == "HOTPOT":
    max_input_length = 1800
    max_output_length = 300
    batch_size_per_gpu = 5
    # if gpu_server_num == 0 or gpu_server_num == 1:
    #     batch_size_per_gpu = 20
    # if gpu_server_num == 2:
    #     batch_size_per_gpu = 10
    # if gpu_server_num == 3 or gpu_server_num == 4 or gpu_server_num == 5:
    #     batch_size_per_gpu = 5
# 10 
peft = False
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
special_tokens = {'additional_special_tokens': ["<<", ">>"]}
train_batch_size = 64
num_train_epochs = 1
gradient_accumulation_steps = 2
num_gen_per_each_gpus = 128
if dataset_name == "ANLI":
    repeat_time = 5
elif dataset_name == "GSM8k":
    repeat_time = 100
elif dataset_name == "HOTPOT":
    repeat_time = 6
#default: AdamW optimizer
learning_rate = 5e-05
half_precision_backend = True
fp16 = True
adam_beta1 = 0.9
adam_beta2 = 0.95
warmup_steps = 10
eval_batch_size = 8
evaluation_strategy = "steps"
eval_steps = 10
eval_accumulation_steps = 1
logging_strategy = "steps"
logging_steps = 5
save_strategy = "epoch"
save_steps = 30
now = datetime.now()
current_time = now.strftime("%y%m%d%H%M")[2:]
output_dir =  f"reward_gen_{base_model_name}_finetune_{dataset_name}_lr_{learning_rate}"
report_to =  "wandb"
deepspeed =  "config/deepspeed_config/ds_config.json"
run_name =  f"model/reward_gen/{base_model_name}_{dataset_name}_lr_{learning_rate}_epochs_{num_train_epochs}_batchsize_{train_batch_size}_date_{current_time}"
config = {
    "random_seed": random_seed,
    "base_model_name": base_model_name,
    "debug": debug,
    "GPU_VISIBLE": GPU_VISIBLE,
    "gpu_server_num": gpu_server_num,
    "dataset_name": dataset_name,
    "sft_model": sft_model, 
    "base_model": base_model,
    "max_input_length": max_input_length,
    "max_output_length": max_output_length, 
    "peft": peft,
    "peft_config": peft_config,
    "special_tokens": special_tokens,
    "repeat_time": repeat_time,
    "num_gen_per_each_gpus": num_gen_per_each_gpus,
    "train_batch_size": train_batch_size,
    "batch_size_per_gpu": batch_size_per_gpu,
    "num_train_epochs": num_train_epochs,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "half_precision_backend": half_precision_backend,
    "fp16": fp16,
    "adam_beta1": adam_beta1,
    "adam_beta2": adam_beta2,
    "warmup_steps": warmup_steps,
    "eval_batch_size": eval_batch_size,
    "evaluation_strategy": evaluation_strategy,
    "eval_steps": eval_steps,
    "eval_accumulation_steps": eval_accumulation_steps,
    "logging_strategy": logging_strategy,
    "logging_steps": logging_steps,
    "save_strategy": save_strategy,
    "save_steps": save_steps,
    
    "output_dir": output_dir,
    "report_to": report_to,
    "deepspeed": deepspeed,
    "run_name": run_name
}