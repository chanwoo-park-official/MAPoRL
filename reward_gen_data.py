# Standard library imports
import argparse
import datetime
import hashlib
import json
import os
import random
import re


# Third-party imports
import pandas as pd
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Type hints
from typing import Any, Dict, List, Optional, Tuple, Union

# Local imports
from utils.dataset import DatasetFactory_reward_training, extract_ans_from_response
from utils.utils_general import (
    set_seed, load_config_from_python, setup_distributed, cleanup_distributed,
    load_safetensors_part, DownloadableFile, unzip_file, check_built, mark_built,
    download_and_check_hash, build_data
)

'''
torchrun --nproc_per_node=8 reward_gen_data.py --config_file config/reward_data_gen_config/config_rewardgen_phi3_GSM8k.py >output_reward_gen.log 2>&1
'''


def extract_ans_gsm8k(answer: str, eos=None):
    """
    Extract the final numerical answer from GSM8k dataset responses.

    The GSM8k dataset uses the format "#### <number>" to indicate the final answer.

    Args:
        answer (str): The model's response text
        eos: Unused parameter (kept for compatibility)

    Returns:
        int: The extracted numerical answer, or 0 if no valid answer found
    """
    pattern = r"\n#### (-?\d+)"
    match = re.search(pattern, answer)
    if match:
        return int(match.group(1))
    return 0

def extract_correctness_hotpot(answer: str, eos=None):
    """
    Extract correctness decision from HotpotQA evaluation responses.

    where 0 indicates incorrect and 1 indicates correct.

    Args:
        answer (str): The evaluation response text
        eos: Unused parameter (kept for compatibility)

    Returns:
        int: 1 for correct, 0 for incorrect, -1 for invalid/no decision found
    """
    answer = answer.strip()
    boxed_pattern = re.compile(r'\\decision\{(.*?)\}')
    matches = boxed_pattern.findall(answer)
    if matches:
        try:
            x = int(matches[-1].strip())
            return x if x in [0, 1] else -1
        except ValueError:
            return -1
    else:
        if f"decision{{{0}}}" in answer:
            return 0
        elif f"decision{{{1}}}" in answer:
            return 1
        return -1


def chat_for_answer(questions, answers, tokenizer, dataset_name, base_model_name,
                    answers_right=None, answers_wrong=None):
    """
    Create chat templates for different datasets to generate evaluation prompts.

    This function creates appropriate chat prompts for evaluating model responses
    based on the dataset type (ANLI, GSM8k, HOTPOT, FANTOM).

    Args:
        questions: Input questions/prompts
        answers: Model-generated answers to evaluate
        tokenizer: Tokenizer for the model
        dataset_name (str): Name of the dataset ("ANLI", "GSM8k", "HOTPOT", "FANTOM")
        base_model_name (str): Name of the base model being used
        answers_right: Correct reference answers (used for HOTPOT and FANTOM)
        answers_wrong: Incorrect reference answers (used for FANTOM)

    Returns:
        torch.Tensor: Tokenized chat prompts ready for model input
    """
    if dataset_name in ["ANLI", "GSM8k", "HOTPOT"]:
        chat_list = [[] for i in range(len(questions))]
        for i in range(len(questions)):
            if dataset_name in ["ANLI", "GSM8k"]:
                chat_list[i] = [{"role": "user", "content": questions[i]}, {"role": "assistant", "content": tokenizer.decode(answers[i], skip_special_tokens=True)}]
                if dataset_name == "ANLI":  
                    chat_list[i].append([{"role": "user", "content": ''' Could you provide your answer within \\boxed{} (e.g., \\boxed{entailment}, \\boxed{neutral}, or \\boxed{contradiction}) based on your previous response? Please do not include any additional information or reasoning, just provide your final answer. NO NEED FOR REASONING. This process is only for grading your previous answer.  '''
                                    }])
                elif dataset_name == "GSM8k":
                    chat_list[i].append([{"role": "user", "content": ''' Could you provide your answer within \\boxed{} (e.g., \\boxed{XXnumeric value}) based on your previous response? Please do not include any additional information or reasoning, just provide your final answer. NO NEED FOR REASONING. This process is only for grading your previous answer.  '''
                                    }])
            elif dataset_name == "HOTPOT":
                if base_model_name == "llama3_2_3B":
                    q= tokenizer.decode(questions[i], skip_special_tokens=True).split("user\n\n", 1)[1]
                    gta = tokenizer.decode(answers_right[i], skip_special_tokens=True).split("assistant\n\n", 1)[1]
                else:
                    q = tokenizer.decode(questions[i], skip_special_tokens=True)
                    gta =  tokenizer.decode(answers_right[i], skip_special_tokens=True)
                prompt =  f'''
                        Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user question, based on the reference answer provided. 
                        Your evaluation should consider correctness. 
                        You will be given a reference answer and the assistant's answer. 
                        You evaluation should focus on the assistant's answer. 
                        Determine the assistant's solution's correctness by comparing the assistant's answer with the reference answer.
                        Be as objective as possible. 
                        Please provide very short explanation of your judging and provide your final decision (0 = incorrect, 1 = correct) and follow the format \\decision{{{0}}} or \\decision{{{1}}}.

                        <|Start of the Question|>
                        {q}
                        <|End of the Question|>

                        <|Start of the Assitant's answer|>
                        Answer: \\boxed{{{extract_ans_from_response("HOTPOT", tokenizer.decode(answers[i], skip_special_tokens=True)).strip()}}}
                        <|End of the Assitant's answer|>

                        <|Start of the Reference Answer|>
                        {gta}
                        <|End of the Reference Answer|>
                        
                        Again, please provide your final decision (0 = incorrect, 1 = correct, 2 = not sure) and follow the format \\decision{{{0}}}, \\decision{{{1}}} or \\decision{{{2}}}, with super short reasoning.
                        '''
                chat_list[i] = [{"role": "user", "content":prompt }]
    elif dataset_name == "FANTOM":
        chat_list = [[] for i in range(len(answers))]
        task_explanation = """
        Your task is to determine which option best aligns with the provided original text, focusing specifically on the concept of unawareness. The original text may describe situations where a person is unaware of something. You will be given two options that paraphrase or reframe the original text, and your goal is to choose the option that most closely captures the meaning of the original, with particular attention to whether the subject is unaware of the information.

        Output: Either "\\boxed{Option 1}" or "\\boxed{Option 2}"
        Focus: Unawareness (i.e., lack of knowledge, not being involved in or informed about a situation)
        When selecting the correct option, prioritize the one that better conveys the subject's unawareness, as implied in the original text. If one option incorrectly implies knowledge or understanding that the original text does not convey, it is likely the wrong choice.

        """
        example1 = """

            EXAMPLE 1: 

            Original Text: "Dominick is unaware of the suggestion made by Orlando and Mia about how job openings are often filled before they are made public. He wasn’t involved when this topic came up."
            
            Option 1: "Dominick does not know or is unaware about Orlando and Mia's suggestion regarding how often job openings are filled before they are publicly announced. This is because he was not involved in the conversation when this aspect was discussed."
            
            Option 2: "Dominick believes that Orlando and Mia's suggestion that job openings are often filled before they are publicly announced is true. They highlighted that many vacancies might not even get advertised and are filled through networking and internal referrals."
            
            Answer: \\boxed{Option 1}
            
            Explanation: Since the original text states that Dominick is unaware of the suggestion, Option 1 matches more closely. Option 2 introduces the idea that Dominick believes in the suggestion, which is not mentioned in the original text.

            """
    
        example2 =    """
        
            Example 2:

            Original Text: "Christine believes that McKenzie's suggestions for overcoming a breakup center on personal growth, changing one's environment or routine, and focusing on self-improvement and rebuilding."

            Option 1: "Christine is unaware about Mckenzie's suggestions for overcoming a breakup as she was not involved in the conversation when this was discussed."

            Option 2: "Christine believes that Mckenzie's suggestions for overcoming a breakup involve focusing on self-growth and betterment, changing one's scenery or routine, and adapting, growing, and rebuilding oneself."

            Answer: \\boxed{Option 2}

            Explanation: Since the original text emphasizes Christine’s belief in McKenzie's suggestions, Option 2 is the correct choice. Option 1 introduces the idea of Christine being unaware, which contradicts the original text. Therefore, Option 2 better aligns with Christine's awareness and understanding of McKenzie's suggestions.

            
                """
 
    
        for i in range(len(answers)):
            # chat_list[i] = [{"role": "user", "content" : task_explanation + example1 + example2  + "PROBLEM: \n\n Original Text:" 
            #                  + tokenizer.decode(answers[i], skip_special_tokens=True) + "\n\nOption 1: " + tokenizer.decode(answers_right[i], skip_special_tokens=True) + "\n\nOption 2: " + tokenizer.decode(answers_wrong[i], skip_special_tokens=True)}]
            chat_list[i] = task_explanation + example1 + example2  + "PROBLEM: \n\n Original Text:" + tokenizer.decode(answers[i], skip_special_tokens=True) + "\n\nOption 1: " + tokenizer.decode(answers_right[i], skip_special_tokens=True) + "\n\nOption 2: " + tokenizer.decode(answers_wrong[i], skip_special_tokens=True) + " \n\n Answer: \\boxed"

    return tokenizer.apply_chat_template(chat_list, padding= True, add_generation_prompt = True, return_tensors = "pt")

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



def setup_model_and_tokenizer(base_model, dataset_name, local_rank):
    """
    Set up the model and tokenizer based on dataset requirements.

    Args:
        base_model (str): Path or name of the base model
        dataset_name (str): Name of the dataset being processed
        local_rank (int): Local rank for distributed processing

    Returns:
        tuple: (model, tokenizer) ready for inference
    """
    if dataset_name in ["FANTOM", "HOTPOT"]:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = model.to(local_rank)

    model = DDP(model, device_ids=[local_rank])
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def setup_dataset_and_loader(dataset_name, tokenizer, repeat_time, gpu_server_num,
                           max_input_length, debug, batch_size_per_gpu):
    """
    Set up the dataset and data loader for distributed processing.

    Args:
        dataset_name (str): Name of the dataset
        tokenizer: Model tokenizer
        repeat_time: Number of times to repeat the dataset
        gpu_server_num: Number of GPU servers
        max_input_length: Maximum input length
        debug: Debug mode flag
        batch_size_per_gpu: Batch size per GPU

    Returns:
        DataLoader: Configured data loader for distributed processing
    """
    dataset = DatasetFactory_reward_training.create_dataset(
        dataset_name, tokenizer, repeat_time, gpu_server_num,
        max_input_length, debug
    )
    sampler = DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_per_gpu, sampler=sampler
    )
    return data_loader


def calculate_reward(dataset_name, pred_answer, answer, rewards_list=None):
    """
    Calculate reward based on dataset type and prediction accuracy.

    Args:
        dataset_name (str): Name of the dataset
        pred_answer: Predicted answer from model
        answer: Ground truth answer
        rewards_list: Pre-computed rewards list (for FANTOM dataset)

    Returns:
        int: Reward value (1 for correct, 0 for incorrect, -1 for invalid)
    """
    if dataset_name in ["ANLI", "GSM8k"]:
        if pred_answer == answer:
            return 1
        elif pred_answer == "invalid":
            return -1
        else:
            return 0
    elif dataset_name == "FANTOM":
        return int(rewards_list.item())
    elif dataset_name == "HOTPOT":
        return pred_answer
    return 0


def save_results(QAC, dataset_name, local_rank, gpu_server_num, base_model_name):
    """
    Save the QAC results to a JSON file.

    Args:
        QAC (dict): Question-Answer-Correctness dictionary
        dataset_name (str): Name of the dataset
        local_rank (int): Local rank for distributed processing
        gpu_server_num (int): Number of GPU servers
        base_model_name (str): Name of the base model
    """
    filename = f'QAC_{dataset_name}_{local_rank}_{gpu_server_num}_{base_model_name}.json'
    with open(filename, 'w') as f:
        json.dump(QAC, f, indent=4)


def process_hotpot_evaluation(questions, outputs, answers, model, tokenizer,
                             local_rank, temperature, base_model_name):
    """
    Process HOTPOT dataset evaluation by generating correctness judgments.

    Args:
        questions: Input questions
        outputs: Model-generated answers
        answers: Ground truth answers
        model: Language model for evaluation
        tokenizer: Model tokenizer
        local_rank: Local rank for distributed processing
        temperature: Sampling temperature
        base_model_name: Name of the base model

    Returns:
        list: Correctness labels for each answer
    """
    chat_for_answer_HOTPOT = chat_for_answer(
        questions, outputs, tokenizer, "HOTPOT", base_model_name, answers
    )
    input_for_answer = chat_for_answer_HOTPOT.to(local_rank)
    attention_for_answer = (input_for_answer != tokenizer.pad_token_id).long().to(local_rank)
    chat_context_length = input_for_answer.shape[-1]

    answer_gen = model.module.generate(
        input_ids=input_for_answer,
        attention_mask=attention_for_answer,
        max_new_tokens=70,
        do_sample=True,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=0.9
    )
    answer_gen = answer_gen[:, chat_context_length:].cpu()
    torch.cuda.empty_cache()

    reward_label = [
        extract_correctness_hotpot(tokenizer.decode(answer_gen[q], skip_special_tokens=True))
        for q in range(answer_gen.shape[0])
    ]

    # Clean up memory
    del chat_for_answer_HOTPOT, input_for_answer, attention_for_answer, answer_gen

    return reward_label


def initialize_qac_entry(question, dataset_name, answer, answers_wrong=None):
    """
    Initialize a QAC entry based on dataset type.

    Args:
        question (str): The question text
        dataset_name (str): Name of the dataset
        answer: Ground truth answer
        answers_wrong: Wrong answers (for FANTOM dataset)

    Returns:
        dict: Initialized QAC entry
    """
    if dataset_name == "ANLI":
        return {
            "original_answer": answer,
            "generated_answers": [],
            "generated_label": [],
            "rewards": [],
            "failed_to_generate": [],
            "answer_want": []
        }
    elif dataset_name == "FANTOM":
        return {
            "original_answer": answer,
            "original_wrong_answer": extract_ans_from_response(
                dataset_name, answers_wrong
            ),
            "generated_answers": [],
            "generated_label": [],
            "rewards": []
        }
    elif dataset_name in ["GSM8k", "HOTPOT"]:
        return {
            "original_answer": answer,
            "generated_answers": [],
            "generated_label": [],
            "rewards": [],
            "failed_to_generate": []
        }
    return {}


def process_response_formatting(response, dataset_name):
    """
    Format response based on dataset requirements and extract predicted answer.

    Args:
        response (str): Raw model response
        dataset_name (str): Name of the dataset

    Returns:
        tuple: (formatted_response, pred_answer)
    """
    pred_answer = None
    if dataset_name in ["GSM8k", "HOTPOT"]:
        boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
        matches = boxed_pattern.findall(response)

        if not matches:
            pred_answer = "invalid"
        else:
            pred_answer = extract_ans_from_response(dataset_name, response)
            answer_str = f"Answer: \\boxed{{{pred_answer}}}"

            if response.endswith(answer_str):
                response = response[:-(len(answer_str))].rstrip()
            response += f"\n\n{answer_str}"

    return response, pred_answer


def process_single_output(i, output, questions, answers, answer_wants, answers_wrong,
                         rewards_list, reward_label, QAC, tokenizer, dataset_name,
                         base_model_name):
    """
    Process a single output from the model and update QAC dictionary.

    Args:
        i (int): Index of the current output
        output: Model-generated output tensor
        questions: List of questions
        answers: List of ground truth answers
        answer_wants: Wanted answers (for ANLI)
        answers_wrong: Wrong answers (for FANTOM)
        rewards_list: Pre-computed rewards (for FANTOM)
        reward_label: Evaluation labels (for HOTPOT)
        QAC (dict): Question-Answer-Correctness dictionary to update
        tokenizer: Model tokenizer
        dataset_name (str): Name of the dataset
        base_model_name (str): Name of the base model
    """
    question = tokenizer.decode(questions[i], skip_special_tokens=True)
    if base_model_name == "llama3_2_3B":
        question = question.split("user\n\n", 1)[1]

    # Extract ground truth answer
    if dataset_name != "GSM8k":
        answer = extract_ans_from_response(
            dataset_name, tokenizer.decode(answers[i], skip_special_tokens=True)
        )
    else:
        answer = extract_ans_gsm8k(tokenizer.decode(answers[i], skip_special_tokens=True))

    # Initialize QAC entry if new question
    if question not in QAC:
        wrong_answer_text = None
        if answers_wrong is not None:
            wrong_answer_text = tokenizer.decode(answers_wrong[i], skip_special_tokens=True)
        QAC[question] = initialize_qac_entry(
            question, dataset_name, answer, wrong_answer_text
        )

    # Process model response
    response = tokenizer.decode(output, skip_special_tokens=True).rstrip()
    response, pred_answer = process_response_formatting(response, dataset_name)

    if dataset_name not in ["GSM8k", "HOTPOT"]:
        pred_answer = extract_ans_from_response(dataset_name, response)

    # Handle dataset-specific logic
    if dataset_name == "ANLI" and answer_wants is not None:
        answer_want = extract_ans_from_response(
            dataset_name, tokenizer.decode(answer_wants[i], skip_special_tokens=True)
        )
        QAC[question]["answer_want"].append(answer_want)

    # Calculate reward
    reward = calculate_reward(dataset_name, pred_answer, answer, rewards_list[i] if rewards_list is not None else None)
    if dataset_name == "HOTPOT" and reward_label is not None:
        reward = reward_label[i]

    # Update QAC with results
    QAC[question]["generated_answers"].append(response)
    QAC[question]["rewards"].append(reward)

    if dataset_name in ["ANLI", "GSM8k"]:
        QAC[question]["generated_label"].append(pred_answer)

    if dataset_name == "ANLI":
        failed = 1 if pred_answer not in ["entailment", "contradiction", "neutral"] else 0
        QAC[question]["failed_to_generate"].append(failed)


def main():
    """
    Main function to generate reward training data using distributed processing.

    This function sets up distributed training, loads the model and tokenizer,
    processes the dataset, generates responses, and saves the results to JSON files.
    The function handles different datasets (ANLI, GSM8k, HOTPOT, FANTOM) with
    appropriate evaluation and reward calculation logic.

    The generated data is saved periodically and at the end of processing.
    """
    setup_distributed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to the configuration file')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (-1: not distributed)')
    args = parser.parse_args()
    config = load_config_from_python(args.config_file)
    globals().update(config)
    set_seed(random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(base_model, dataset_name, local_rank)

    # Setup dataset and data loader
    data_loader = setup_dataset_and_loader(
        dataset_name, tokenizer, repeat_time, gpu_server_num,
        max_input_length, debug, batch_size_per_gpu
    )

        model.eval()
    QAC = {}
    cnt = 0

    for batch in tqdm(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(local_rank)
        attention_mask = batch["attention_mask"].to(local_rank)
        questions = batch["question"]
        answers = batch["answer"]

        # Extract dataset-specific information
        answer_wants = batch.get("answer_want") if dataset_name == "ANLI" else None
        answers_wrong = batch.get("answer_want") if dataset_name == "FANTOM" else None

        # Generate responses
        random_temperature = random.uniform(1.3, 1.5)
        with torch.no_grad():
            outputs = model.module.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_output_length,
                do_sample=True,
                tokenizer=tokenizer,
                temperature=random_temperature,
                top_p=0.9
            )
            outputs = outputs[:, input_ids.shape[-1]:].cpu()

        # Clean up GPU memory
        input_ids.cpu()
        attention_mask.cpu()
        torch.cuda.empty_cache()

        # Special handling for HOTPOT dataset
        reward_label = None
        if dataset_name == "HOTPOT":
            reward_label = process_hotpot_evaluation(
                questions, outputs, answers, model, tokenizer,
                local_rank, random_temperature, base_model_name
            )

        # Extract rewards list for FANTOM
        rewards_list = batch.get("reward") if dataset_name == "FANTOM" else None

        # Process each output in the batch
        for i, output in enumerate(outputs):
            process_single_output(
                i, output, questions, answers, answer_wants, answers_wrong,
                rewards_list, reward_label, QAC, tokenizer, dataset_name,
                base_model_name
            )

        # Save intermediate results every 10 batches
        if cnt % 10 == 0:
            save_results(QAC, dataset_name, local_rank, gpu_server_num, base_model_name)
        cnt += 1

    # Save final results
    save_results(QAC, dataset_name, local_rank, gpu_server_num, base_model_name)
    cleanup_distributed()

if __name__ == "__main__":
    main()
