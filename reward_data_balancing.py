# Standard library imports
import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

# Local imports
from TOKEN import HF_TOKEN
from utils.dataset import DatasetFactory_reward_training, extract_ans_from_response
from utils.utils_general import (
    set_seed,
    load_config_from_python,
    setup_distributed,
    cleanup_distributed,
    load_safetensors_part,
    DownloadableFile,
    unzip_file,
    check_built,
    mark_built,
    download_and_check_hash,
    build_data
)


# torchrun --nproc_per_node=8 reward_generation_data.py --config_file config/reward_data_gen_config/config_rewardgen_llama32_GSM8k_0.py >output_reward_gen.log 2>&1

# Constants
WRONG_ANSWER_CAUTIONS = '''
And please you MUST not say that you are intentionally making a wrong answer. Do not provide a right answer. Make a wrong answer, while you should not mention that you are intentionally generating wrong answer. Also, do not only provide the answer, you need some reasoning.
'''

# Configuration constants
DESIRED_ANSWERS_PER_TYPE = 10  # Number of correct and incorrect answers desired per question


# Global argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
args = parser.parse_args()
config = load_config_from_python(args.config_file)
globals().update(config)


def clean_endtoken(response: torch.Tensor, base_model_name: str, tokenizer) -> torch.Tensor:
    """
    Clean the end token from model response for specific model types.

    Args:
        response: The generated token sequence from the model
        base_model_name: Name of the base model being used
        tokenizer: The tokenizer used for decoding

    Returns:
        The cleaned response tensor with end tokens removed
    """
    if base_model_name == "phi3":
        # Find the index of the first occurrence of token 32007
        try:
            index_32007 = (response == 32007).nonzero(as_tuple=True)[0][0]  # First occurrence
        except IndexError:
            index_32007 = None  # 32007 is not found in the response

        # If 32007 is found, replace everything after it with the pad token
        if index_32007 is not None:
            response[index_32007 + 1:] = tokenizer.pad_token_id  # Set tokens after 32007 to pad_token_id
    return response

def extract_ans_gsm8k(answer: str, eos=None) -> int:
    """
    Extract the numerical answer from GSM8K format response.

    Args:
        answer: The model response containing the answer in #### format
        eos: End of sequence token (optional, not used in current implementation)

    Returns:
        The extracted numerical answer, or 0 if no valid answer found
    """
    pattern = r"\n#### (-?\d+)"
    match = re.search(pattern, answer)
    # Extract and convert to integer
    if match:
        result = int(match.group(1))
        return result
    else:
        return 0

def question_prompt(s: str) -> str:
    return f'Question: {s}'

def format_chat_template(q: str) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nProvide a detailed reasoning for your solution. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way."}]
    return row_json

def format_chat_template_for_over80(q: str, random_answer: List[str]) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a wrong answer which seems right. You should not make a calculation mistake, but make some logical mistake. Here is a wrong example answer: " + ", ".join(random_answer) + "\n\n Now, make a wrong answer without no calculation mistake. If question is very easy so that it is not easy to make wrong answer, make very creatvie wrong answer. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. " + WRONG_ANSWER_CAUTIONS}]
    return row_json

def format_chat_template_for_over95(q: str, random_answer: List[str]) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a completely wrong answer. You should not make a calculation mistake, but make some logical mistake. Here is a wrong example answer: " + ", ".join(random_answer) + "\n\n Now, make a wrong answer without no calculation mistake. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. " + WRONG_ANSWER_CAUTIONS}]
    return row_json

def format_chat_template_for_100(q: str) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a completely wrong answer. You should not make a calculation mistake, but make some logical mistake. If question is very easy so that it is not easy to make wrong answer, make very creatvie wrong answer. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. Remember, do not write answer first. " + WRONG_ANSWER_CAUTIONS}]
    return row_json

def format_chat_template_for_over80_phi3(q: str, random_answer: List[str]) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a wrong answer which seems right. You should not make a calculation mistake, but make some logical mistake. Here is a wrong example answer: " + ", ".join(random_answer) + "\n\n Now, make a wrong answer without no calculation mistake. If question is very easy so that it is not easy to make wrong answer, make very creatvie wrong answer. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. Write reasoning first, and then provide the answer. Remember, do not write answer first. " + WRONG_ANSWER_CAUTIONS}]
    return row_json

def format_chat_template_for_over95_phi3(q: str, random_answer: List[str]) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a completely wrong answer. You should not make a calculation mistake, but make some logical mistake. Here is a wrong example answer: " + ", ".join(random_answer) + "\n\n Now, make a wrong answer without no calculation mistake. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. Write reason first, and then provide the answer. Remember, do not write answer first. " + WRONG_ANSWER_CAUTIONS}]
    return row_json

def format_chat_template_for_100_phi3(q: str) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nPlease make a completely wrong answer. You should not make a calculation mistake, but make some logical mistake. If question is very easy so that it is not easy to make wrong answer, make very creatvie wrong answer. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. Write reasoning first, and then provide the answer. Remember, do not write answer first. " + WRONG_ANSWER_CAUTIONS}]
    return row_json




def format_chat_template_for_below20(q: str, random_answer: List[str]) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": "Question: " + q + "\n\nI provide you a possible sample answer (which is guaranteed to be right). However, do not copy them to make your own answer, but note that I already provided an answer. Example answers are as follows: " + ", ".join(random_answer) + " At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. Write reasoning first, and then provide the answer. Remember, do not write answer first."}]
    return row_json

def format_chat_templalte_for_mimic(q: str, random_answer: str, your_style: str) -> List[Dict[str, str]]:
    row_json = [{"role": "user", "content": f'''
        **Task:**
        Please do **not** correct any errors in the original answer; your focus is purely on rephrasing.

        ---

        ### Assistant's Answer:
        <|start|>
        {random_answer}
        <|end|>

        ---

        ### Your Task:
        Reword the above content in **your style** provided below. Pay attention to the **format** and the **writing style** specified. Remember, you should not modify the correctness of the original answer.

        ---

        ### Example of Your Style:
        <|start|>
        {your_style}
        <|end|>

        ---

        Begin rewriting the content, making sure your rephrased version reflects your unique style while preserving the original meaning, even if it contains mistakes. Only provide solution and answer. Do not rewrite the task or question. Again, do not write TASK or QUESTION.

    '''}]
    return row_json

def clean_response(response: str) -> str:
    """
    Clean and normalize the model response by removing unwanted prefixes and suffixes.

    Args:
        response: The raw response string from the model

    Returns:
        The cleaned response string with prefixes and artifacts removed
    """
    # Remove "Reasoning:" if it starts the response
    if response.startswith("Reasoning:"):
        response = response[10:]
    # Remove "<|start|>" if it starts the response
    if response.startswith("<|start|>"):
        response = response[len("<|start|>"):]
    
    response = response.strip()
    
    # Split the response into sentences and check the first one
    sentences = response.split('.')
    if sentences[0].strip().startswith("Alright,"):
        # Remove the first sentence
        sentences = sentences[1:]
    
    # Join the remaining sentences and strip the final response
    response = '.'.join(sentences).strip()
    
    # Find the index of the phrase "### Your Task: "
    task_index = response.find("### Your Task: ")
    
    # If the phrase exists, truncate everything after it
    if task_index != -1:
        response = response[:task_index]
    
    # Strip the response to remove extra whitespace
    response = response.strip()
    
    # Handle the ending `-` or `--` (flexibly removing all trailing hyphens)
    while response.endswith('-'):
        response = response[:-1].strip()

    response = response.strip()
    return response

def avoiding(response: str) -> bool:
    if "rephrase" in response.lower():
        return True
    if "rewrite" in response.lower():
        return True
    if "reword" in response.lower():
        return True
    if "###" in response.lower():
        return True
    else:
        return False
    

def main():
    alpha = 1.1
    setup_distributed()
    set_seed(random_seed)    
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,  trust_remote_code=True, quantization_config=bnb_config)

    model = DDP(model, device_ids = [local_rank])



    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = DatasetFactory_reward_training.create_dataset(dataset_name, tokenizer, 1, gpu_server_num, max_input_length, debug )
    sampler = DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)


    if dataset_name == "GSM8k":
        if gpu_server_num == 0 or gpu_server_num == 1:
            num_repeats = 16
            trial_num_limit = 4

        elif gpu_server_num == 2:
            num_repeats = 8 # Number of answers to generate for each question
            trial_num_limit = 8
    elif dataset_name == "HOTPOT": 
        if gpu_server_num == 0 or gpu_server_num == 1:
            num_repeats = 16
            trial_num_limit = 6
        elif gpu_server_num == 2:
            num_repeats = 8
            trial_num_limit = 6
                   

    QAC = {}
    passed_number = 0
    with open(f'QAC_{dataset_name}_{base_model_name}_{gpu_server_num}.json', 'r') as f:
        config = json.load(f)

    if dataset_name == "GSM8k":
        with open(f'QAC_{dataset_name}_balanced_phi3.json', 'r') as f:
            ref = json.load(f)
    else:
        ref = {}
    cnt = 0
    for batch in tqdm(data_loader):
        input_id = batch['input_ids'][0]
        # decode input_id 
        question = tokenizer.decode(input_id, skip_special_tokens=True)
        question = question.strip()
        question = question.split("Question:")[1]
        question = question[:question.find("\n\nProvide")]
        question = question.strip()

        answer = batch['answer'][0]
        answer = tokenizer.decode(answer, skip_special_tokens=True)
        if base_model_name == "llama3_2_3B":
            answer = answer.split("assistant\n\n", 1)[1]
        
        QAC[question] = {
                        "original_answer": answer,
                        "generated_answers": [],
                        "rewards": []
                    }
        trial_num = 0 
        weird = 0
        if dataset_name == "GSM8k":
            if question not in ref.keys():
                passed_number += 1
                print(local_rank, "passed-number", passed_number, "the question is", question)
                continue
            if "rewards" not in ref[question].keys():
                passed_number += 1
                print(local_rank, "passed-number", passed_number, "the question is", question)
                continue
        if question not in config.keys():
            passed_number += 1
            print(local_rank, "passed-number", passed_number, "the question is", question)
            continue

        # batch is size 1 tensor, and I want to decode it. 
        reward_cnt = config[question]['rewards'].count(1)
        wrong_cnt = config[question]['rewards'].count(0) 
        ### if refs[question] does not have ["rewards"], continue 

        if reward_cnt < DESIRED_ANSWERS_PER_TYPE:
            config[question]["generated_answers"].append('Answer: ' + answer)
            config[question]["rewards"].append(1)
            right_answer_number = reward_cnt + 1 
            current_new_right_number = 0
            while current_new_right_number < DESIRED_ANSWERS_PER_TYPE - right_answer_number:
                trial_num += 1
                if trial_num > trial_num_limit:
                    weird = 1
                    break

                one_index_ref = [i for i, e in enumerate(ref[question]["rewards"]) if e == 1]
                selected_one_index_ref = random.sample(one_index_ref, 1)
                selected_exampler_answer_ref = ref[question]["generated_answers"][selected_one_index_ref[0]]
                len_of_question = len(config.keys())
                random_indice_for_style = random.randint(0, len_of_question-1)
                style_ref = config[list(config.keys())[random_indice_for_style]]["generated_answers"][0]
                # Generate num_repeats of row_json
                if trial_num < alpha * trial_num_limit:
                    row_json_list = [format_chat_template_for_below20(
                                    question, 
                                    config[question]["generated_answers"][random.sample(
                                            [i for i, e in enumerate(ref[question]["rewards"]) if e == 1], 1)[0]]) 
                                for _ in range(num_repeats)]
                else:
                    row_json_list = [format_chat_templalte_for_mimic(
                                        question, 
                                        ref[question]["generated_answers"][random.sample(
                                            [i for i, e in enumerate(ref[question]["rewards"]) if e == 1], 1)[0]], 
                                        config[list(config.keys())[random.randint(0, len(config.keys()) - 1)]]
                                        ["generated_answers"][0]) 
                                    for _ in range(num_repeats)]

                # Convert each row_json to chat template using the tokenizer
                input_ids_list = [
                    tokenizer.apply_chat_template(
                        row_json, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                    for row_json in row_json_list
                ]
                # Find the maximum length in the input_ids_list to pad all tensors to this length
                max_len = max(input_id.size(1) for input_id in input_ids_list)
                input_ids_list_padded = [
                    F.pad(input_id, (max_len - input_id.size(1), 0), value=tokenizer.pad_token_id) for input_id in input_ids_list
                ]

                # Stack the padded tensors
                input_ids = torch.cat(input_ids_list_padded, dim=0).to(local_rank)

                random_temperature = random.uniform(0.5, 1)

                outputs = model.module.generate(
                    input_ids,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=random_temperature,
                    top_p=0.9)
                outputs = outputs.cpu()
        
                for i in range(num_repeats):
                    response = outputs[i][input_ids.shape[-1]:]
                    response = clean_endtoken(response, base_model_name, tokenizer)
                    response = tokenizer.decode(response, skip_special_tokens=True)
                    response = clean_response(response)
                    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
                    matches = boxed_pattern.findall(response)
                    if avoiding(response):
                        pred_ans = "invalid"
                        continue
                    if not matches:  # No matches for `\boxed{}` means invalid
                        pred_ans = "invalid"
                        continue
                    else:                
                        pred_ans = extract_ans_from_response("GSM8k", response)
                        answer_str = f"Answer: \\boxed{{{pred_ans}}}"
                        if response.endswith(answer_str):
                            # Remove the previous `Answer: \\boxed{}` and any newlines before it
                            response = response[:-(len(answer_str))].rstrip()


                        # Now add the formatted string with exactly two newlines
                        response += f"\n\n{answer_str}"

                    reward = 1 if pred_ans == extract_ans_gsm8k(answer) else 0
                    if reward == 1:
                        config[question]["generated_answers"].append(response)
                        config[question]["rewards"].append(reward)
                        current_new_right_number += 1
                del response, outputs, matches, boxed_pattern, input_ids
                torch.cuda.empty_cache()
        trial_num = 0 
        if wrong_cnt < DESIRED_ANSWERS_PER_TYPE:
            wrong_answer_number = wrong_cnt
            current_new_wrong_number = 0
            while current_new_wrong_number < DESIRED_ANSWERS_PER_TYPE - wrong_answer_number:
                trial_num += 1
                if trial_num > trial_num_limit:
                    weird = 1
                    break

                if trial_num < alpha * trial_num_limit:
                    if wrong_cnt + current_new_wrong_number == 0:
                        row_json_list = [format_chat_template_for_100(question) for _ in range(num_repeats)]
                    elif wrong_cnt + current_new_wrong_number < 0.2 * DESIRED_ANSWERS_PER_TYPE:
                        row_json_list = [format_chat_template_for_over95(question, config[question]["generated_answers"][random.sample(
                                            [i for i, e in enumerate(ref[question]["rewards"]) if e == 0], 1)[0]]) 
                                        for _ in range(num_repeats)]
                    else:
                        row_json_list = [format_chat_template_for_over80(
                                    question, 
                                    config[question]["generated_answers"][random.sample(
                                            [i for i, e in enumerate(ref[question]["rewards"]) if e == 0], 1)[0]]) 
                                for _ in range(num_repeats)]
                else:
                    row_json_list = [format_chat_templalte_for_mimic(question, 
                                    ref[question]["generated_answers"][random.sample(
                                        [i for i, e in enumerate(ref[question]["rewards"]) if e == 0], 1)[0]], 
                                    config[list(config.keys())[random.randint(0, len(config.keys())-1)]]
                                    ["generated_answers"][0]) 
                                for _ in range(num_repeats)]

                # Convert each row_json to chat template using the tokenizer
                input_ids_list = [
                    tokenizer.apply_chat_template(
                        row_json, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                    for row_json in row_json_list
                ]

                # Find the maximum length in the input_ids_list to pad all tensors to this length
                max_len = max(input_id.size(1) for input_id in input_ids_list)

                # Pad each tensor to match the max length
                input_ids_list_padded = [
                    F.pad(input_id, (max_len - input_id.size(1), 0), value=tokenizer.pad_token_id) for input_id in input_ids_list
                ]


                # Stack the padded tensors
                input_ids = torch.cat(input_ids_list_padded, dim=0).to(local_rank)

                random_temperature = random.uniform(0.5, 1)

                outputs = model.module.generate(
                    input_ids,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=random_temperature,
                    top_p=0.9)
                outputs = outputs.cpu()
                for i in range(num_repeats):
                    response_before = outputs[i]
                    response = outputs[i][input_ids.shape[-1]:]
                    response = clean_endtoken(response, base_model_name, tokenizer)


                    response = tokenizer.decode(response, skip_special_tokens=True)
                    response = clean_response(response)
                    

                    response = response.strip()
                    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
                    matches = boxed_pattern.findall(response)
                    if avoiding(response):
                        pred_ans = "invalid"
                        continue
                    if not matches:  # No matches for `\boxed{}` means invalid
                        pred_ans = "invalid"
                        continue
                    else:                
                        pred_ans = extract_ans_from_response("GSM8k", response)
                        answer_str = f"Answer: \\boxed{{{pred_ans}}}"
                        if response.endswith(answer_str):
                            # Remove the previous `Answer: \\boxed{}` and any newlines before it
                            response = response[:-(len(answer_str))].rstrip()
                        if response.startswith("<|start|>"):
                            response = response[len("<|start|>"):]

                        # Now add the formatted string with exactly two newlines
                        response += f"\n\n{answer_str}"

                    reward = 1 if pred_ans == extract_ans_gsm8k(answer) else 0

                    if reward == 0 and isinstance(pred_ans, int):
                        config[question]["generated_answers"].append(response)
                        config[question]["rewards"].append(reward)
                        current_new_wrong_number += 1
                del response, outputs, matches, boxed_pattern, input_ids
                torch.cuda.empty_cache()

        if weird == 1:
            passed_number += 1
            print(local_rank, "passed-number", passed_number, "the question is", question)
            del QAC[question]
        else:
            # check config's answer order
            generated_answers = config[question]['generated_answers']
            rewards = config[question]['rewards']
            # select DESIRED_ANSWERS_PER_TYPE of 0 and 1 indices for reward
            zero_index = [i for i, e in enumerate(rewards) if e == 0]
            one_index = [i for i, e in enumerate(rewards) if e == 1]
            selected_zero_index = random.sample(zero_index, DESIRED_ANSWERS_PER_TYPE)
            selected_one_index = random.sample(one_index, DESIRED_ANSWERS_PER_TYPE)
            for i in selected_zero_index:
                QAC[question]["generated_answers"].append(generated_answers[i])
                QAC[question]["rewards"].append(rewards[i])
            for i in selected_one_index:
                QAC[question]["generated_answers"].append(generated_answers[i])
                QAC[question]["rewards"].append(rewards[i])



        if cnt % 10 == 0:
            #save QAC
            with open(f'QAC_filtered_GSM8k__{local_rank}_{gpu_server_num}_{base_model_name}.json', 'w') as f:
                json.dump(QAC, f, indent=4)
            
        cnt += 1


    with open(f'QAC_filtered_GSM8k__{local_rank}_{gpu_server_num}_{base_model_name}.json', 'w') as f:
        json.dump(QAC, f, indent=4)
    

    cleanup_distributed()

if __name__ == "__main__":
    main()
