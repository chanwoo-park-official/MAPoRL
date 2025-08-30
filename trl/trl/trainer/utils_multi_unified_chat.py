"""
Chat utilities for PPO training.

This module contains utility functions for formatting chat conversations
and preparing data for language model interactions.
"""

from typing import List
import json
import requests
import torch


def chat_for_answer(questions: List[str], answers, tokenizer, dataset_name: str):
    """
    Format questions and answers into chat format for model evaluation.

    Creates a chat conversation format with user questions and assistant answers,
    adding specific instructions based on the dataset type for answer formatting.

    Args:
        questions: List of question strings
        answers: Tensor of answer token sequences
        tokenizer: The tokenizer to use for decoding and chat template application
        dataset_name: Name of the dataset ("ANLI" or "GSM8k")

    Returns:
        Formatted chat template ready for model input

    Raises:
        ValueError: If dataset_name is not supported
    """
    chat_list = [[] for i in range(len(questions))]
    for i in range(len(questions)):
        chat_list[i] = [
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": tokenizer.decode(answers[i], skip_special_tokens=True)}
        ]

        if dataset_name == "ANLI":
            chat_list[i].append({
                "role": "user",
                "content": ''' Could you provide your answer within \\boxed{} (e.g., \\boxed{entailment}, \\boxed{neutral}, or \\boxed{contradiction}) based on your previous response? Please do not include any additional information or reasoning, just provide your final answer. NO NEED FOR REASONING. This process is only for grading your previous answer.  '''
            })
        elif dataset_name == "GSM8k":
            chat_list[i].append({
                "role": "user",
                "content": ''' Could you provide your answer within \\boxed{} based on your previous response? Please do not include any additional information or reasoning, just provide your final answer. NO NEED FOR REASONING. This process is only for grading your previous answer.  '''
            })
        else:
            raise ValueError("Dataset name not predefined or open-ended-answer required Dataset")

    return tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, return_tensors="pt", padding=True)


def chat_for_no_reasoning_check(answers, tokenizer, dataset_name: str):
    """
    Format answers into chat format for reasoning analysis.

    Creates a chat conversation that prompts the model to analyze whether
    reasoning is provided in the given answers, specifically for ANLI dataset.

    Args:
        answers: Tensor of answer token sequences to analyze
        tokenizer: The tokenizer to use for decoding and chat template application
        dataset_name: Name of the dataset (currently only supports "ANLI")

    Returns:
        Formatted chat template ready for model input

    Raises:
        ValueError: If dataset_name is not supported or reasoning check is not needed
    """
    # Initialize chat_list to hold conversation pairs
    chat_list = [[] for _ in range(len(answers))]

    # Iterate through the questions and answers
    for i in range(len(answers)):
        # Add user question and assistant response to chat list
        answer_decoded = tokenizer.decode(answers[i], skip_special_tokens=True)

        if dataset_name == "ANLI":
            chat_list[i] = [
                {
                    "role": "user",
                    "content": f"""

    You are given a statement about the relationship between a hypothesis and a premise. Your task is to first classify whether **reasoning** is provided using the format \\boxed{{Reason Provided}} or \\boxed{{No Reason Provided}}. Afterward, explain why you classified it that way.

    ### **Hints for Classifying Reasoning:**

    1. **Reason Provided:**
    Reasoning is provided when the statement includes **specific details** or a **clear explanation** that justifies how or why the premise supports (or contradicts) the hypothesis. The explanation should provide a logical connection or evidence for why the relationship holds.

    **Important**: Simply saying "directly supports," "entails," or "contradicts" without further explanation **does not count as reasoning**. The statement must **explain how or why** the premise leads to the conclusion.

    2. **No Reason Provided:**
    Reasoning is **not** provided if the statement only classifies the relationship (e.g., "entailment," "contradictory") or uses terms like "directly supports" **without explaining how or why** the relationship holds. The model should only consider reasoning as provided if there is a clear logical explanation connecting the premise and hypothesis.

    ---

    ### Example 1
    **Statement**: "The premise directly supports the hypothesis."
    **Correct Classification**: \\boxed{{No Reason Provided}}
    **Why**: While the statement says "directly supports," it does not explain *how* or *why* the premise supports the hypothesis. Without further justification or details, it cannot be considered reasoning.

    ### **Example 2:**
    - **Statement**: "The premise does not provide information about the platforms Minecraft can be played on, making the relationship neutral."
    - **Classification**: \\boxed{{Reason Provided}}
    - **Why**: This statement provides reasoning by explaining that the premise lacks specific information about platforms, which justifies the neutral classification.

    ### **Example 3:**
    - **Statement**: "The hypothesis contradicts the premise."
    - **Classification**: \\boxed{{No Reason Provided}}
    - **Why**: The statement does not explain *why* the hypothesis and premise contradict each other; it merely states the classification.


    **Question:**
    - **Statement**:
    {answer_decoded}

    Proceed classification and give why.
    """
                }
            ]
        else:
            raise ValueError(
                "Dataset name not predefined or open-ended-answer required Dataset "
                "or we do not need reasoning check"
            )

    # Apply tokenizer chat template and return the result
    return tokenizer.apply_chat_template(
        chat_list,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    )


def combine_agent_contexts(agent_contexts_set_list, agent, turn, reward_feedback=True):
    """
    Combine agent contexts from multiple turns into a single context.

    Args:
        agent_contexts_set_list: List of agent context sets
        agent: Agent index to combine contexts for
        turn: Current turn number
        reward_feedback: Whether to include reward feedback in context

    Returns:
        List: Combined agent contexts
    """
    new_agent_contexts_set = []
    cnt = 0
    for j, agent_contexts in enumerate(agent_contexts_set_list):
        new_agent_contexts = []

        # Add the first element (index 0) to the new list as it doesn't follow the pattern
        new_agent_contexts.append(agent_contexts[agent][0])
        # Loop through the turns and combine relevant entries
        if reward_feedback:
            for i in range(turn):
                # Add the element at 3*i + 1
                new_agent_contexts.append(agent_contexts[agent][3*i + 1])

                # Concatenate the elements at 3*i + 2 and 3*i + 3
                combined_content = {"role": "user", "content": agent_contexts[agent][3*i + 2]['content'] + "\n\n" + agent_contexts[agent][3*i + 3]['content']}

                # Add the combined content to the new list
                new_agent_contexts.append(combined_content)
        else:
            for i in range(turn):
                new_agent_contexts.append(agent_contexts[agent][2*i + 1])
                new_agent_contexts.append(agent_contexts[agent][2*i + 2])

        # Add the new list to the set
        new_agent_contexts_set.append(new_agent_contexts)
        cnt += 1
    return new_agent_contexts_set


def formatting_question(dataset_name, model_name, question):
    if dataset_name == "GSM8k":
        if model_name == "microsoft/Phi-3-mini-128k-instruct":
            output = "Question: " + question + "\n\nProvide a reasoning for your solution. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way."
        elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            output =   "Question: " + question  + "\n\n Let's think step by step. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way. "
        elif model_name == "Qwen/Qwen2.5-3B-Instruct":
            output =  "Question: " + question + "\n\nProvide a short but precise reasoning for your solution. At the end, you MUST write the answer in the following format:\n\nAnswer: \\boxed{XX}\n\nPlease ensure that the final answer is always formatted this way."
        else:
            output =  "Question: " + question + "\n\nProvide a short but precise reasoning for your solution. At the end, you MUST write the answer in the following format:\n\nAnswer: \\boxed{XX}\n\nPlease ensure that the final answer is always formatted this way."
    elif dataset_name == "ANLI":
        output =  question
    else:
        ValueError("The dataset name is not supported")
    return output


def construct_message_multi_agent(agents, question, turn, dataset_name, reward_feedback=False):
    # Check if there are any agents
    if dataset_name == "FANTOM":
        index = question.rfind('\n\n Question:')
        if index != -1:
            # Return the substring starting from that index
            question_for_input = question[index+3:]  # +2 to remove the leading newline characters
        else:
            # If the sequence is not found, return the original text
            question_for_input =  question
    else:
        question_for_input = question

    if len(agents) == 0:
        return {
            "role": "user",
            "content": (
                f"Here, 'reward' refers to the likelihood that each suggested answer is correct, as evaluated by a verifier. "
                f"The reward value ranges between 0 and 1, with values closer to 1 indicating a higher probability of correctness. "
                f"Although the reward might not be perfect, it comes from a highly reliable verifier and is typically a good indicator of accuracy. "
                f"You should focus on providing the reasoning and the final answer based on this context, without explicitly discussing the reward. "
                f"Ensure that your reasoning is clear and includes any key steps or thought processes that justify your conclusion.\n\n"
                f"Once again, the question is:\n {question_for_input}. "
            )
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    # Determine the index based on the turn and reward_feedback flag
    if reward_feedback:
        idx = 3 * turn - 2
    else:
        idx = 2 * turn - 1

    # Loop through each agent's response
    agent_num = 1
    for agent in agents:
        agent_response = agent[idx]["content"]

        if reward_feedback:
            try:
                agent_reward = agent[idx + 1]["content"]
                # print("line1592: ", agent_reward, "\n\n")
                agent_reward = agent_reward.replace(
                    "Reward from a verifier of your answer:",
                    f"Reward associated with agent {agent_num}'s solution calculated by verifier: "
                )
                agent_reward = agent_reward.replace(
                    "your",
                    f"agent {agent_num}'s"
                )
                agent_reward = agent_reward.replace(
                    "Your",
                    f"Agent {agent_num}'s"
                )
            except Exception as e:
                print("\n\n", agents, "\n\n")
                print("Error:", str(e))

            response = f"\n\n Agent {agent_num} solution: ```{agent_response}```"
            response += f"\n\n {agent_reward}"
        else:
            response = f"\n\n Agent {agent_num} solution: ```{agent_response}```"

        prefix_string += response
        agent_num = agent_num + 1
    if reward_feedback:
        prefix_string += (
            f"Here, each reward represents the probability that a suggested answer is correct, as evaluated by a verifier. "
            f"The reward value is between 0 and 1, with values closer to 1 indicating a higher likelihood of correctness. "
            f"While these rewards offer useful context, they are not always perfect, though generally quite reliable.\n\n"
            f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
            f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
            f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
            f"Once again, the question is: {question_for_input}"
        )
    else:
        prefix_string += (
            f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
            f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
            f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
            f"Once again, the question is: {question_for_input}"
        )

    return {"role": "user", "content": prefix_string}


def construct_message_multi_agent_eval(agents, question, turn, dataset_name, reload, reward_feedback=False):
    # Check if there are any agents
    if dataset_name == "FANTOM":
        index = question.rfind('\n\n Question:')
        if index != -1:
            # Return the substring starting from that index
            question_for_input = question[index+3:]  # +2 to remove the leading newline characters
        else:
            # If the sequence is not found, return the original text
            question_for_input =  question
    else:
        question_for_input = question

    if reload:
        question_for_input= " "

    if len(agents) == 0:
        return {
            "role": "user",
            "content": (
                f"Here, 'reward' refers to the likelihood that each suggested answer is correct, as evaluated by a verifier. "
                f"The reward value ranges between 0 and 1, with values closer to 1 indicating a higher probability of correctness. "
                f"Although the reward might not be perfect, it comes from a highly reliable verifier and is typically a good indicator of accuracy. "
                f"You should focus on providing the reasoning and the final answer based on this context, without explicitly discussing the reward. "
                f"Ensure that your reasoning is clear and includes any key steps or thought processes that justify your conclusion.\n\n"
                f"Once again, the question is:\n {question_for_input}. "
            )
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    # Determine the index based on the turn and reward_feedback flag
    if reward_feedback:
        idx = 3 * turn - 2
    else:
        idx = 2 * turn - 1

    # Loop through each agent's response
    agent_num = 1
    for agent in agents:
        agent_response = agent[idx]["content"]

        if reward_feedback:
            try:
                agent_reward = agent[idx + 1]["content"]
                # print("line1592: ", agent_reward, "\n\n")
                agent_reward = agent_reward.replace(
                    "Reward from a verifier of your answer:",
                    f"Reward associated with agent {agent_num}'s solution calculated by verifier: "
                )
                agent_reward = agent_reward.replace(
                    "your",
                    f"agent {agent_num}'s"
                )
                agent_reward = agent_reward.replace(
                    "Your",
                    f"Agent {agent_num}'s"
                )
            except Exception as e:
                print("\n\n", agents, "\n\n")
                print("Error:", str(e))

            response = f"\n\n Agent {agent_num} solution: ```{agent_response}```"
            response += f"\n\n {agent_reward}"
        else:
            response = f"\n\n Agent {agent_num} solution: ```{agent_response}```"

        prefix_string += response
        agent_num = agent_num + 1
    if reward_feedback:
        if reload:
            prefix_string += (
                f"Here, each reward represents the probability that a suggested answer is correct, as evaluated by a verifier. "
                f"The reward value is between 0 and 1, with values closer to 1 indicating a higher likelihood of correctness. "
                f"While these rewards offer useful context, they are not always perfect, though generally quite reliable.\n\n"
                f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
                f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
                f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
                f"Once again, the question is: {question_for_input}"
            )
        else:
            prefix_string += (
                f"Here, each reward represents the probability that a suggested answer is correct, as evaluated by a verifier. "
                f"The reward value is between 0 and 1, with values closer to 1 indicating a higher likelihood of correctness. "
                f"While these rewards offer useful context, they are not always perfect, though generally quite reliable.\n\n"
                f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
                f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
                f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
            )
    else:
        prefix_string += (
            f"Focus on providing a well-reasoned response that not only considers your own previous solution but also takes into account answers from other agents. "
            f"If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided. Also, internaly think about the reward of your and other agents' answer."
            f"Ensure that your explanation is well justifing your final answer. Please maintain your answer with very simple reasoning.\n\n"
            f"Once again, the question is: {question_for_input}"
        )

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}


def message_formatter(sample, dataset_name):
    if dataset_name == "ANLI":
        msg = [{
            "role": "user",
            "content": (
                "Premise: " + sample["premise"] + "\n\n"
                "Hypothesis: " + sample["hypothesis"] + "\n\n"
                "Please determine the relationship between the premise and hypothesis. Choose one of the following: 'entailment', 'neutral', or 'contradiction'. "
                "Start with a concise reasoning for your choice, and conclude with your final answer. You do not need to restate the premise and hypothesis."
            )
        }]
    return msg
