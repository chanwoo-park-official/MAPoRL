import ast
import io
import os
import re
import sys
import json
import requests
from collections import Counter
from typing import Optional
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from peft import get_peft_model
import torch
from utils.utils_general import list_folders

# Constants
INVALID_ANSWER = -1
INVALID_ANSWER_STR = str(INVALID_ANSWER)
MAX_INPUT_LENGTH_MATH = 250
TRAIN_DATASET_SIZE_MATH = 100000
TEST_DATASET_SIZE_MATH = 1000
DEBUG_DATASET_SIZE = 2000
def convert_and_sort_folders(folder_names):
    print(folder_names)
    try:
        folder_numbers = [int(name) for name in folder_names if name.isdigit()]
        folder_numbers.sort()
        print(folder_numbers)
        return folder_numbers
    except ValueError as e:
        return str(e)

def create_invalid_entry(input_ids, tokenizer):
    """Create a standardized invalid entry for dataset processing."""
    return {
        "input_ids": input_ids,
        "lengths": len(input_ids),
        "answer": tokenizer.encode(INVALID_ANSWER_STR)
    }


def safe_execute_code(code_string, function_name='simple_math_problem'):
    """
    Safely execute code and return the result of a specific function.

    Args:
        code_string: The code to execute
        function_name: Name of the function to call from the executed code

    Returns:
        The result of calling the function, or INVALID_ANSWER if execution fails
    """
    try:
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        namespace = {}
        exec(code_string, namespace)
        result = namespace[function_name]()
    except Exception:
        return INVALID_ANSWER
    finally:
        sys.stdout = original_stdout
    return result


def extract_info_from_ppo(input_string):
    # Define regular expressions for extracting the values
    agent_regex = r'agent_(\d+)'
    round_regex = r'round_(\d+)'
    policy_separation_regex = r'policy_separation_(True|False)'

    # Search for the values in the string
    agent_match = re.search(agent_regex, input_string)
    round_match = re.search(round_regex, input_string)
    policy_separation_match = re.search(policy_separation_regex, input_string)

    # Check if all values are found
    if agent_match and round_match and policy_separation_match:
        agent_number = int(agent_match.group(1))
        round_number = int(round_match.group(1))
        policy_separation = policy_separation_match.group(1) == 'True'

        return agent_number, round_number, policy_separation
    else:
        return False, False, False


def update_state_dict_keys_ppo2(state_dict):
    updated_state_dict = {}
    
    for key in state_dict.keys():
        # Patterns for matching keys with specified structures
        patterns = [
            (re.compile(r'base_model\.model\.model\.layers\.(\d+)\.mlp\.down_proj\.lora_A\.weight'), 
             r'base_model.model.model.layers.\1.mlp.down_proj.lora_A.default.weight'),
            (re.compile(r'base_model\.model\.model\.layers\.(\d+)\.mlp\.down_proj\.lora_B\.weight'), 
             r'base_model.model.model.layers.\1.mlp.down_proj.lora_B.default.weight'),
            (re.compile(r'base_model\.model\.model\.layers\.(\d+)\.self_attn\.o_proj\.lora_A\.weight'), 
             r'base_model.model.model.layers.\1.self_attn.o_proj.lora_A.default.weight'),
            (re.compile(r'base_model\.model\.model\.layers\.(\d+)\.self_attn\.o_proj\.lora_B\.weight'), 
             r'base_model.model.model.layers.\1.self_attn.o_proj.lora_B.default.weight')
        ]
        
        new_key = key
        for pattern, replacement in patterns:
            new_key = re.sub(pattern, replacement, new_key)
        
        updated_state_dict[new_key] = state_dict[key]
        
    return updated_state_dict


def update_external_model_lora(model, state_dict, yyy):
    model_state_dict = model.state_dict()
    for key, value in state_dict.items():
        # Identifying the type of weight and corresponding indices
        if "mlp.down_proj.lora_A.weight" in key:
            xxx_index = key.split('.')[4]
            new_key = f'base_model.model.model.layers.{xxx_index}.mlp.down_proj.lora_A.{yyy}.weight'
        elif "mlp.down_proj.lora_B.weight" in key:
            xxx_index = key.split('.')[4]
            new_key = f'base_model.model.model.layers.{xxx_index}.mlp.down_proj.lora_B.{yyy}.weight'
        elif "self_attn.o_proj.lora_A.weight" in key:
            xxx_index = key.split('.')[4]
            new_key = f'base_model.model.model.layers.{xxx_index}.self_attn.o_proj.lora_A.{yyy}.weight'
        elif "self_attn.o_proj.lora_B.weight" in key:
            xxx_index = key.split('.')[4]
            new_key = f'base_model.model.model.layers.{xxx_index}.self_attn.o_proj.lora_B.{yyy}.weight'
        else:
            continue

        if new_key in model_state_dict:
            model_state_dict[new_key].copy_(value)
        else:
            print(f"Warning: Key {new_key} not found in the model's state dict.")

    # Load the updated state dictionary into the model
    model.load_state_dict(model_state_dict)




def step_config(mode, ppo_model):
    if mode == "ppo":
        steps = convert_and_sort_folders(list_folders(ppo_model))+ ["original_model"] 
    elif mode == "sft":
        steps = ["sft-ed", "original_model"]

    steps = [step for step in steps if not step == 0]

    return steps

def model_path_for_step(mode, step, base_model, sft_model, ppo_model):
    if step == "original_model":
        model_path = base_model
        return model_path
    if mode == "sft" and step == "sft-ed":
        model_path = sft_model
    elif mode == "ppo":
        model_path = f"{ppo_model}/{step}"
    
    return model_path

def model_for_step(mode, step, base_model, sft_model, peft, peft_config, local_rank):
    if mode == "sft" and step =="original_model":
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,  trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(sft_model, torch_dtype=torch.bfloat16,  trust_remote_code=True)
        if base_model == "microsoft/Phi-3-mini-128k-instruct":
            if peft:
                model = get_peft_model(model, peft_config, adapter_name = "ref")      
                print("ref is ready")              
    return model    


def construct_message_multi_agent(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer. \n\n The original question is as follows:\n {}.".format(question)}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to this question? \n\n The original question is as follows: \n {}. \n You should not copy other agents' answer, but just use information from other agents.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}

def initialize_policy_weights(policy_model, state_dict, round_num):
    layer_keys = [
        ('self_attn.o_proj.lora_A.value_{turn}.weight', 'self_attn.o_proj.lora_A.default.weight'),
        ('self_attn.o_proj.lora_B.value_{turn}.weight', 'self_attn.o_proj.lora_B.default.weight'),
        ('mlp.down_proj.lora_A.value_{turn}.weight', 'mlp.down_proj.lora_A.default.weight'),
        ('mlp.down_proj.lora_B.value_{turn}.weight', 'mlp.down_proj.lora_B.default.weight')
    ]
    num_layers = policy_model.config.num_hidden_layers

    for i in range(num_layers):
        for turn in range(round_num):
            for policy_key_template, value_key_template in layer_keys:
                policy_key = policy_key_template.format(turn=turn)
                value_key = value_key_template

                policy_weight_path = f'policy_model.base_model.model.model.layers[{i}].{policy_key}'
                value_weight_path = f'state_dict["model.base_model.model.model.layers.{i}.{value_key}"]'

                # Evaluate the paths to get the weights
                policy_weight = eval(policy_weight_path)
                value_weight = eval(value_weight_path)

                # Copy the weights from value_model to policy_model
                policy_weight.data.copy_(value_weight.data)


def extract_ans_from_response(answer: str) -> any:
    """Extract numerical answer from response string."""
    answer = answer.strip()
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    matches = boxed_pattern.findall(answer)
    if matches:
        answer = matches[-1].strip()

    answer = re.sub(r'[^0-9\-\.]', '', answer)
    try:
        return int(answer)
    except ValueError:
        return "invalid"


def extract_ans_from_response_with_str(answer: str) -> str:
    """Extract answer from response as string."""
    answer = answer.strip()
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    matches = boxed_pattern.findall(answer)
    if matches:
        answer = matches[-1].strip()
        return answer
    else:
        return "invalid"
def extract_reward(text):
    """
    Extracts the reward value from a given string.

    :param text: Input string containing the reward information.
    :return: Extracted reward as a float, or None if not found.
    """
    match = re.search(r"Reward from a verifier of your answer:\s*([\d\.]+)", text)
    return float(match.group(1)) if match else None

def update_turn_based_training(update, criteria_for_consensus_percentage, criteria_for_consensus_reward_threshold,
                              update_initial_turn, turn_based_training_num, round_num):
    """
    Update the turn based training.

    This function is used to update the turn based training. -- so for ease of training,
    we use turn based training so that we freeze other turn's model parameters.

    Args:
        update: Current update number
        criteria_for_consensus_percentage: Consensus percentage threshold
        criteria_for_consensus_reward_threshold: Consensus reward threshold
        update_initial_turn: Initial turn update number
        turn_based_training_num: Number of turns for training
        round_num: Total number of rounds

    Returns:
        int or list: Turn number(s) to update
    """

    if criteria_for_consensus_percentage >=1 or criteria_for_consensus_reward_threshold >=1:
        k = (update-update_initial_turn) // turn_based_training_num  # Find the appropriate k value
        turn_update_turn = round_num -2 - (k % (round_num-1 ))
        if -1<=update<=update_initial_turn- 1:
            turn_update_turn = 0
        return turn_update_turn + 1
    else:
        k = (update-update_initial_turn) // turn_based_training_num  # Find the appropriate k value
        return [i for i in range(1, round_num)]


def stat_all(answer_input_forstat, answers_all_agent_turn, finished_question, open_ended_answer):
    """
    Calculate statistics for all agent turns.

    Args:
        answer_input_forstat: Ground truth answers (or tuple for open-ended)
        answers_all_agent_turn: List of agent answers for each turn
        finished_question: List indicating which questions are finished
        open_ended_answer: Whether this is open-ended question type

    Returns:
        tuple: (correctness_turn_list, count_turn_list) - Statistics for each turn
    """
    from sentence_transformers import SentenceTransformer
    from collections import Counter
    import torch

    if open_ended_answer:
        groundtruth_answers = answer_input_forstat[0]
        groundtruth_wrong_answers = answer_input_forstat[1]
        sentence_embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
        answer_turn_list = [[[] for j in range(len(answers_all_agent_turn[0]))] for i in range(len(answers_all_agent_turn))]
    else:
        groundtruth_answers = answer_input_forstat
        answer_turn_list = answers_all_agent_turn

    count_turn_list = []  # Will store the difference between turn and turn+1 correctness

    correctness_turn_list = [[] for i in range(len(answers_all_agent_turn))]
    for turn in range(len(answers_all_agent_turn)):
        correctness = []

        # Step 1: Filter the valid questions based on the condition
        valid_indices = [idx for idx in range(len(finished_question)) if finished_question[idx] >= turn or finished_question[idx] == -1]

        for q in range(len(groundtruth_answers)):
            groundtruth = groundtruth_answers[q]
            if open_ended_answer:
                groundtruth_wrong = groundtruth_wrong_answers[q]
                groundtruth_TF_list = [groundtruth, groundtruth_wrong]
                embeddings_groundtruth = sentence_embedding_model.encode(groundtruth_TF_list)


            # Only consider answers if the question has not been finished before or at the current turn
            if finished_question[q] >= turn or finished_question[q] == -1:
                # Map the current q to the correct index in valid_indices
                q_index_in_valid = valid_indices.index(q)

                if len(answers_all_agent_turn) == 1:
                    current_turn_answers = [answers[q_index_in_valid] for answers in answers_all_agent_turn[turn] if answers]
                    if not open_ended_answer:
                        if current_turn_answers and current_turn_answers[0] == groundtruth:
                            correctness.append(1)
                        else:
                            correctness.append(0)
                    else:
                        if current_turn_answers:
                            agent_ans_list = [current_turn_answers[0]]
                            embeddings_agent = sentence_embedding_model.encode(agent_ans_list)
                            similarities = sentence_embedding_model.similarity(embeddings_groundtruth, embeddings_agent)
                            if similarities[0] >= similarities[1]:
                                correctness.append(1)
                        else:
                            correctness.append(0)


                else:
                    current_turn_answers = [answers[q_index_in_valid] for answers in answers_all_agent_turn[turn] if answers]
                    if not current_turn_answers:  # No answers provided
                        correctness.append(-2)
                    else:
                        if open_ended_answer:
                            embeddings_agent = sentence_embedding_model.encode(current_turn_answers)
                            similarities = sentence_embedding_model.similarity(embeddings_groundtruth, embeddings_agent)
                            ans_extracted = ['correct' if similarities[i, 0] > similarities[i, 1] else 'wrong' for i in range(len(current_turn_answers))]
                            for i in range(len(current_turn_answers)):
                                answer_turn_list[turn][i].append(ans_extracted[i])
                            answer_counts = Counter(ans_extracted)
                        else:
                            answer_counts = Counter(current_turn_answers)


                        max_count = max(answer_counts.values())

                        # Find all answers that have the maximum count
                        majority_votes = [ans for ans, count in answer_counts.items() if count == max_count]
                        if "invalid" in majority_votes and len(majority_votes) > 1:
                            majority_votes.remove("invalid")

                        # Check if the groundtruth is in the majority votes
                        if open_ended_answer:
                            if 'correct' in majority_votes:
                                correctness.append(1 / len(majority_votes))

                            else:
                                correctness.append(0)

                        else:
                            if groundtruth in majority_votes:
                                correctness.append(1 / len(majority_votes))  # Correct, but with possible ties

                            else:
                                correctness.append(0)  # Incorrect

            else:
                correctness.append(-1)  # Mark as finished for this turn, no need to evaluate
        correctness_turn_list[turn] = correctness


    for turn in range(len(answers_all_agent_turn) - 1):
        diff = []
        turn_correct = correctness_turn_list[turn]
        turnp1_correct = correctness_turn_list[turn + 1]

        for q in range(len(groundtruth_answers)):
            # Calculate the difference between turn and turn + 1 correctness
            if turn_correct[q] == -1 or turnp1_correct[q] == -1:
                diff.append(-2)  # No comparison needed
            else:
                if turn_correct[q] < turnp1_correct[q]:
                    diff.append(1)  # Improvement
                elif turn_correct[q] > turnp1_correct[q]:
                    diff.append(-1)  # Decline
                else:
                    diff.append(0)  # No change

        improve_count = diff.count(1)
        decline_count = diff.count(-1)
        no_change_count = diff.count(0)

        count_turn_list.append([improve_count, decline_count, no_change_count])
    if open_ended_answer:
        del sentence_embedding_model
        torch.cuda.empty_cache()

    return correctness_turn_list, count_turn_list


def check_repeated_sequences(tokenizer, sequence, min_seq_length=1, max_seq_length=20, max_repeats=3):
    """
    Check if any subsequence of length 1-11 is repeated more than 3 times,
    excluding sequences of numeric tokens (0-9).

    Args:
        tokenizer: The tokenizer to use for decoding
        sequence: torch.Tensor of shape (sequence_length,)
        min_seq_length: minimum length of subsequence to check
        max_seq_length: maximum length of subsequence to check
        max_repeats: maximum number of allowed repetitions

    Returns:
        bool: True if invalid repetition is found, False otherwise
    """
    seq_len = len(sequence)

    # Convert sequence to list for easier processing
    seq_list = sequence.tolist()

    for length in range(min_seq_length, min(max_seq_length + 1, seq_len + 1)):
        for start in range(seq_len - length + 1):
            # Get the subsequence
            subseq = seq_list[start:start + length]

            # Skip if all tokens in subsequence are numeric (0-9 in token space)
            decoded_subseq = tokenizer.decode(subseq)
            if decoded_subseq.strip().isdigit():
                continue

            # Count repetitions
            repeat_count = 1
            current_pos = start + length

            while current_pos + length <= seq_len:
                next_subseq = seq_list[current_pos:current_pos + length]
                if next_subseq == subseq:
                    repeat_count += 1
                    if repeat_count > max_repeats:
                        return True
                    current_pos += length
                else:
                    break

    return False


def reward_from_different_server(query_responses, ip, reward_feedback = True):
    # Convert the tensor to a list for JSON serialization
    # query_responses_list = query_responses.tolist()
    # json_post = {'query_responses': queries_for_score_batches}


    if reward_feedback:
        json_post = {'query_responses': query_responses}

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
    else:
        return torch.zeros(len(query_responses))


def grad_requires_setting(model, task_training, policy_separation, collaboration_separation):
    for name, param in model.named_parameters():
        if task_training:
            if ("lora" in name) and ("ref" not in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            if policy_separation:
                if collaboration_separation:
                    if ("lora" in name) and ("ref" not in name) and (("value" in name) or ("col" in name)):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if ("lora" in name) and ("ref" not in name) and (("value" in name) or ("policy_0" not in name)):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


def calculate_reward_from_answer(dataset_name, queries_for_score_batches, answer_text_set_small_batch):
    """
    Calculate rewards based on answer correctness for given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'GSM8k', 'ANLI')
        queries_for_score_batches: List of query batches to score
        answer_text_set_small_batch: Ground truth answers

    Returns:
        torch.Tensor: Binary reward tensor (1 for correct, 0 for incorrect)
    """
    from utils.dataset import extract_ans_from_response  # Import here to avoid circular imports

    answer_from_your_queries = [
        extract_ans_from_response(dataset_name, queries_for_score_batch[1]["content"])
        for queries_for_score_batch in queries_for_score_batches
    ]
    return torch.tensor(
        [1 if answer_from_your_queries[i] == answer_text_set_small_batch[i] else 0
         for i in range(len(answer_from_your_queries))],
        dtype=torch.long
    )


def check_agent_consensus(answers_all_agent, rewards_all_agent, criteria_for_consensus_percentage,
                         criteria_for_consensus_reward_threshold, q_index_in_unfinished):
    """
    Check if agents have reached consensus on answers.

    Args:
        answers_all_agent: List of answers from all agents
        rewards_all_agent: List of rewards for all agents
        criteria_for_consensus_percentage: Minimum percentage for consensus
        criteria_for_consensus_reward_threshold: Minimum reward threshold
        q_index_in_unfinished: Index of question in unfinished list

    Returns:
        bool: True if consensus reached, False otherwise
    """
    rewards_all_agent_new = [rewards_all_agent[agent][0][q_index_in_unfinished]
                           for agent in range(len(answers_all_agent))]

    if len(answers_all_agent) == 0:
        return False

    answer_counter = Counter(answers_all_agent)
    majority_answer, majority_count = answer_counter.most_common(1)[0]
    majority_percentage = majority_count / len(answers_all_agent)

    if majority_answer == "invalid":
        return False

    if majority_percentage <= criteria_for_consensus_percentage:
        return False

    # Filter rewards for majority answer
    majority_rewards = [rewards_all_agent_new[i] for i in range(len(answers_all_agent))
                       if answers_all_agent[i] == majority_answer]
    avg_reward = torch.tensor(majority_rewards).mean().item()

    return avg_reward > criteria_for_consensus_reward_threshold
