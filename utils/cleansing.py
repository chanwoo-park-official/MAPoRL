import json
import re
from typing import Dict, Any, List


def combine_configs(configs_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """Combine multiple configuration dictionaries into one."""
    combined = {}

    for config_key, config in configs_dict.items():
        for key, value in config.items():
            if key not in combined:
                combined[key] = value
            else:
                combined[key]['generated_answers'] += value['generated_answers']
                combined[key]['rewards'] += value['rewards']

    return combined


def answer_cleansing(ans: str) -> str:
    """Clean and format the answer string."""
    from .utils_cooperateLLM import extract_ans_from_response

    extract_original_answer = extract_ans_from_response(ans)
    answer_original_str = f"Answer: \\boxed{{{extract_original_answer}}}"
    ans = ans.replace(answer_original_str, "")
    ans = ans.strip()

    if "---\n" in ans:
        ans = ans.split("---\n")[0]
        ans = ans.strip()
    if "Question:" in ans:
        ans = ans.split("Question:")[0]
        ans = ans.strip()
    if "## Your Task" in ans:
        ans = ans.split("## Your Task")[0]
        ans = ans.strip()

    banned_expression = ["\\begin", "**Task"]
    replace_expression = ["```", "'''"]
    prev_save_expresstion = ["Question:", "---\n", "## Your Task", "rewritten", "rewritting", "rephras", "rework"]

    for banned in banned_expression:
        if banned in ans.lower():
            ans = " "
    for replace in replace_expression:
        if replace in ans.lower():
            ans = ans.replace(replace, "")
    for prev_save in prev_save_expresstion:
        if prev_save in ans.lower():
            ans = ans.split(prev_save)[0]
            ans = ans.strip()

    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    matches = boxed_pattern.findall(ans)
    ans = ans.strip()

    if len(matches) >= 2:
        ans = " "
    elif len(matches) == 1:
        ans0 = matches[0]
        ans1 = extract_original_answer
        answer_0_str = f"Answer: \\boxed{{{ans0}}}"
        answer_1_str = f"\n\nAnswer: \\boxed{{{ans1}}}"
        pattern = r"^\d+[a-zA-Z]+\d+$"
        ans = ans.strip()

        while ans.endswith("`"):
            ans = ans[:-1]
        while ans.endswith("'"):
            ans = ans[:-1]
        while ans.startswith("`"):
            ans = ans[1:]
        while ans.startswith("'"):
            ans = ans[1:]
        ans = ans.strip()

        if re.match(pattern, ans0):
            ans = " "
        if ans.endswith(answer_0_str):
            ans = ans.replace(answer_0_str, "")

    if len(ans) == 0:
        ans = " "
    else:
        ans = ans.strip()
        ans = ans + "\n\n" + answer_original_str
    ans = ans.strip()
    if ans == answer_original_str:
        ans = " "
    return ans
