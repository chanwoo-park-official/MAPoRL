import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as HFDataset
import random
from tqdm import tqdm
import itertools
from typing import List, Dict, Optional, Tuple, Any
from utils.utils_general import set_seed, json_loader, DownloadableFile, unzip_file, check_built, mark_built, download_and_check_hash, build_data

# Constants
DEFAULT_MAX_LENGTH = 1000
DEFAULT_RANDOM_SEED = 42
DEBUG_DATASET_SIZE = 100
DEFAULT_TEST_LEN = 100
CRITICAL_VALUE = 1  # Used in reward dataset balancing
MIN_ANSWERS = 30   # Minimum answers per question for cleansing

# TODO: Consider making dataset loading more flexible to handle varying dataset locations
class BaseDataset(Dataset):
    """Base dataset class for loading and processing question-answer data.

    This class provides a foundation for dataset implementations with common functionality
    for tokenization and data preparation for language model training.
    """
    def __init__(self, tokenizer: Any, split: str, max_length: int = DEFAULT_MAX_LENGTH, debug: bool = False, eval_mode: bool = False) -> None:
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.debug = debug
        self.eval_mode = eval_mode
        self.qa_list, self.a_list = self.load_data()

    def load_data(self) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def __len__(self) -> int:
        return len(self.qa_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        txt = self.qa_list[idx]
        encodings_dict = self.tokenizer.apply_chat_template(txt, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True, add_generation_prompt = self.eval_mode)
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        if self.eval_mode:
            txt_a = self.a_list[idx]
            ## extract the answer.text
            txt_a = [{'role': 'assistant', 'content': txt_a[0]["content"].split("####", 1)[-1].strip()}]
            encodings_dict_a = self.tokenizer.apply_chat_template(txt_a, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True)
            input_ids_a = torch.tensor(encodings_dict_a["input_ids"])
            attn_masks_a = torch.tensor(encodings_dict_a["attention_mask"])

            return {
                        "input_ids": input_ids,
                        "attention_mask": attn_masks,
                        "labels": input_ids,
                        "answers": input_ids_a,
                    }
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }

class BaseDataset_reward(Dataset):
    def __init__(self, tokenizer: Any, split: str, max_length: int = DEFAULT_MAX_LENGTH, test_len: int = DEFAULT_TEST_LEN, debug: bool = False, shuffle: bool = True) -> None:
        set_seed(DEFAULT_RANDOM_SEED)
        self.input_ids = []
        self.attn_masks = []
        self.rewards = []
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.debug = debug
        self.test_len = test_len
        self.qar_list = self.load_data()
        self.indices = list(range(len(self.qar_list)))
        self.shuffle = shuffle

        for qar in tqdm(self.qar_list):
            qa, reward = qar["qa"], qar["reward"] 
            #assume that we have one shot, but if not, we might change this part, with getting the question as well.
            encodings_dict_qa = tokenizer.apply_chat_template(qa, truncation=True, max_length=max_length, padding="max_length", return_dict=True)
            self.input_ids.append(torch.tensor(encodings_dict_qa["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict_qa["attention_mask"]))
            self.rewards.append(torch.tensor(reward, dtype = torch.float32))
        if shuffle:
            self.shuffle_data()
    
    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def __len__(self):
        return len(self.qar_list)
    
    def __getitem__(self, idx):
        shuffled_idx = self.indices[idx]
        return (
            self.input_ids[idx],
            self.attn_masks[idx],
            self.rewards[idx]
        )
    
    def shuffle_data(self):
        random.shuffle(self.indices)

class OMIDataset(BaseDataset):
    def load_data(self):
        if self.split == "train":
            dataset_path = "data/OpenMathInstruct-1_train.json"
        if self.split == "test":
            dataset_path = "data/OpenMathInstruct-1_validation.json"
        set_seed(42)
        qa_list = []
        QAC = json_loader(dataset_path)

        if self.debug:
            QAC = dict(itertools.islice(QAC.items(), 100))
        
        for key, value in tqdm(QAC.items()):
            if len(value["correct_generated_solution"]) == 0:
                continue
            ans_index = random.randint(0, len(value["correct_generated_solution"]) - 1)
            qa_list.append([
                {"role": "system", "content": "You are Meta AI, a sophisticated and energetic AI Assistant. You excel at solving mathematical problems."},
                {"role": "user", "content": "Question: " + key + "\n\n Let's think step by step. Make sure to put the answer (and only answer) inside \\boxed{}."},
                {"role": "assistant", "content": value["correct_generated_solution"][ans_index]}
            ])
        return qa_list

class GSM8kDataset(BaseDataset):
    def load_data(self):
        qa_list = []
        a_list = []
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        if self.debug:
            dataset = dataset.select(range(500))
        if self.eval_mode:
            for sample in dataset:
                qa_list.append([
                    {"role": "user", "content": "Question: " + sample["question"] + "\n\n Let's think step by step. Make sure to put the answer (and only answer) inside \\boxed{}."}])
                a_list.append( [{"role": "assistant", "content": "Answer: " + sample["answer"]}])
        else:
            for sample in dataset:
                qa_list.append([
                    {"role": "user", "content": "Question: " + sample["question"] + "\n\n Let's think step by step. Make sure to put the answer (and only answer) inside \\boxed{}."},
                    {"role": "assistant", "content": "Answer: " + sample["answer"]}
                ])
        return qa_list, a_list

class ANLIDataset(BaseDataset):
    def load_data(self):
        qa_list = []
        a_list = []
        dataset = load_dataset('anli', split=f"{self.split}_r3")
        label = ['entailment', 'neutral', 'contradiction']
        if self.debug:
            dataset = dataset.select(range(100))
        
        if self.eval_mode:
            for sample in dataset:
                qa_list.append([
                    {"role": "user", "content": "Premise: " + sample["premise"] + "\n\n Hypothesis: " + sample["hypothesis"] + "\n\n Please choose the relationship between the given premise and hypothesis as one of the following: 'entailment', 'neutral', or 'contradiction'. Provide reasoning for your choice. You do not need to repeat premise and hypothesis in your response. Please **ALWAYS write your choice inside a box when you mention your choice** (e.g \\boxed{entailment}, \\boxed{neutral}, and \\boxed{contradiction}. When you mention your choice, you SHOULD write your choice in \\boxed{} so double check you used \\boxed{}"}])
                a_list.append([{"role": "assistant", "content": sample["reason"] + "\n\n \\boxed{" + label[sample['label']] + "}"}])
        else:
            for sample in dataset:
                qa_list.append([
                    {"role": "user", "content": "Premise: " + sample["premise"] + "\n\n Hypothesis: " + sample["hypothesis"] + "\n\n Please choose the relationship between the given premise and hypothesis as one of the following: 'entailment', 'neutral', or 'contradiction'. Provide reasoning for your choice. You do not need to repeat premise and hypothesis in your response. Please **ALWAYS write your choice inside a box when you mention your choice** (e.g \\boxed{entailment}, \\boxed{neutral}, and \\boxed{contradiction}. When you mention your choice, you SHOULD write your choice in \\boxed{} so double check you used \\boxed{}"},
                    {"role": "assistant", "content": sample["reason"] +  "\n\n \\boxed{" + label[sample['label']] + "}"}])
        return qa_list, a_list

class GSM8kDataset_reward(BaseDataset_reward):
    def load_data(self):
        qar_list = []
        dataset_path = "data/GSM8k_dataset_trial/QAC_GSM8k_balanced_phi3.json"
        QAC = json_loader(dataset_path)
        if self.split == "test":
            QAC = dict(itertools.islice(QAC.items(), len(QAC) - self.test_len, len(QAC)))
        elif self.debug:
            QAC = dict(itertools.islice(QAC.items(), 100))
        elif self.split == "train":
            QAC = dict(itertools.islice(QAC.items(), len(QAC) -self.test_len))
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
        for (key, value) in tqdm(QAC.items()):
            if "generated_answers" not in value:
                continue
            for ans_index in range(len(value["generated_answers"])):
                qar_dict = {}                
                qar_dict["q"] = [{"role": "user", "content": "Question: " + key + "\n\nProvide a reasoning for your solution. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way." }]
                qar_dict["qa"] = [{"role": "user", "content": "Question: " + key + "\n\nProvide a reasoning for your solution. At the end, you MUST write the answer in the following format:```\n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way." },
                {"role": "assistant", "content": value["generated_answers"][ans_index]}]
                qar_dict["answer"] = value["generated_answers"][ans_index]
                qar_dict["reward"] = float(value["rewards"][ans_index])
                qar_list.append(qar_dict)
        return qar_list

class ANLIDataset_reward(BaseDataset_reward):
    def load_data(self):
        qar_list = []
        dataset_path = "data/QAC_ANLI_combined_filtered.json"
        QAC = json_loader(dataset_path)
        if self.split == "test":
            QAC = dict(itertools.islice(QAC.items(), len(QAC) - self.test_len, len(QAC)))
        elif self.debug:
            QAC = dict(itertools.islice(QAC.items(), 4500))
        elif self.split == "train":
            QAC = dict(itertools.islice(QAC.items(), len(QAC) -self.test_len))
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
        for (key, value) in tqdm(QAC.items()):
            for ans_index in range(len(value["generated_answers"])):
                qar_dict = {}
                qar_dict["q"] = [{"role": "user", "content": key }]
                qar_dict["qa"] = [{"role": "user", "content": key} ,
                {"role": "assistant", "content": value["generated_answers"][ans_index]}]
                qar_dict["answer"] = value["generated_answers"][ans_index]
                qar_dict["reward"] = float(value["rewards"][ans_index])
                qar_list.append(qar_dict)
        return qar_list

class FANTOMDataset_reward(BaseDataset_reward):
    def load_data(self):
        qar_list = []
        if self.split == "train":
            dataset_path = "data/QAC_FANTOM_combined.json"
        elif self.split == "test":
            dataset_path = "data/QAC_FANTOM_combined_test.json"            
        QAC = json_loader(dataset_path)
        if self.debug:
            QAC = dict(itertools.islice(QAC.items(), 100))
        
        for (key, value) in tqdm(QAC.items()):
            one_ind = 0
            zero_ind = 0
            critical_value = CRITICAL_VALUE
            for ans_index in range(len(value["generated_answers"])):
                if float(value["rewards"][ans_index]) == 1.0 and one_ind <= critical_value - 1:                    
                    qar_dict = {}
                    qar_dict["q"] = [{"role": "user", "content": key }]
                    qar_dict["qa"] = [{"role": "user", "content": key} ,
                    {"role": "assistant", "content": value["generated_answers"][ans_index]}]
                    qar_dict["answer"] = value["generated_answers"][ans_index]
                    qar_dict["reward"] = float(value["rewards"][ans_index])
                    qar_list.append(qar_dict)
                    one_ind += 1
                elif float(value["rewards"][ans_index]) == 0.0 and zero_ind <= critical_value - 1:
                    qar_dict = {}
                    qar_dict["q"] = [{"role": "user", "content": key }]
                    qar_dict["qa"] = [{"role": "user", "content": key} ,
                    {"role": "assistant", "content": value["generated_answers"][ans_index]}]
                    qar_dict["answer"] = value["generated_answers"][ans_index]
                    qar_dict["reward"] = float(value["rewards"][ans_index])
                    qar_list.append(qar_dict)
                    
                if one_ind >=critical_value and zero_ind >= critical_value:
                    break
        return qar_list


class DatasetFactory:
    """Factory class for creating dataset instances based on dataset name."""

    @staticmethod
    def create_dataset(dataset_name: str, tokenizer: Any, split: str, max_length: int = DEFAULT_MAX_LENGTH, debug: bool = False, eval_mode: bool = False) -> BaseDataset:
        if dataset_name == 'OMI':
            return OMIDataset(tokenizer, split, max_length, debug, eval_mode)
        elif dataset_name == 'GSM8k':
            return GSM8kDataset(tokenizer, split, max_length, debug, eval_mode)
        elif dataset_name == 'ANLI':
            return ANLIDataset(tokenizer, split, max_length, debug, eval_mode)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

class DatasetFactory_reward:
    @staticmethod
    def create_dataset(dataset_name, tokenizer, split, max_length=1000, test_len = 100, debug=False):
        if dataset_name == 'GSM8k':
            return GSM8kDataset_reward(tokenizer, split, max_length, test_len, debug)
        elif dataset_name == 'ANLI':
            return ANLIDataset_reward(tokenizer, split, max_length, test_len, debug)
        elif dataset_name == "FANTOM":
            return FANTOMDataset_reward(tokenizer, split, max_length, test_len, debug)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

class DatasetFactory_PPO:
    @staticmethod
    def create_dataset(dataset_name, tokenizer, split, max_length=1000, debug=False):
        if dataset_name == 'GSM8k':
            return GSM8kDataset_PPO(tokenizer, split, max_length, debug)
        elif dataset_name == 'ANLI':
            return ANLIDataset_PPO(tokenizer, split, max_length, debug)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")

class BaseDataset_PPO(Dataset):
    def __init__(self, tokenizer, split, max_length=1000, debug=False):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.debug = debug
        self.dataset = None
        self.prepare_dataset()
        
    def prepare_dataset(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class GSM8kDataset_PPO(BaseDataset_PPO):
    def prepare_dataset(self):
        self.dataset = load_dataset('gsm8k', 'main', split=self.split)
        if self.debug:
            self.dataset = self.dataset.select(range(100))
        self.dataset = self.dataset.rename_column("question", "query")
        self.dataset = self.dataset.remove_columns(["answer"])
        self.dataset = self.dataset.map(self.tokenize)
    
    def tokenize(self, sample):
        sample["input_ids"] = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Question: " + sample["query"] + "\n\n Let's think step by step. Make sure to put the answer (and only answer) inside \\boxed{}. "}],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=True
        )
        sample["lengths"] = len(sample["input_ids"])
        return sample

class ANLIDataset_PPO(BaseDataset_PPO):
    def prepare_dataset(self):
        self.dataset = load_dataset('anli', split=f"{self.split}_r3")
        self.dataset = self.dataset.map(self.tokenize)
    
    def tokenize(self, sample):
        sample["input_ids"] = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Premise: " + sample["premise"] + "\n\n Hypothesis: " + sample["hypothesis"] + "\n\n Please choose the relationship between the given premise and hypothesis as one of the following: 'entailment', 'neutral', or 'contradiction'. Provide reasoning for your choice. You do not need to repeat premise and hypothesis in your response. At the end, write the final answer inside a box (e.g \\boxed{entailment}, \\boxed{neutral}, and \\boxed{contradiction}."}],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=True
        )
        return sample


class DataCollator_reward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0].unsqueeze(0) for f in data], dim=0)
        batch["attention_mask"] = torch.cat([f[1].unsqueeze(0) for f in data], dim=0)
        batch["rewards"] = torch.cat([f[2].unsqueeze(0) for f in data], dim=0)
        batch["labels"] = batch["rewards"]
        return batch


def DataCollator_PPO_GSM8k(data):
    batch = {}
    batch["query"] = [f["query"] for f in data]
    batch["input_ids"] = [torch.tensor(f['input_ids']) for f in data]
    return batch

def DataCollator_PPO_ANLI(data):
    batch = {}
    batch["premise"] = [f["premise"] for f in data]
    batch["hypothesis"] = [f["hypothesis"] for f in data]
    batch["input_ids"] = [torch.tensor(f['input_ids']) for f in data]
    return batch

def nshot_chats_GSM8k(nshot_data: List[Dict], n: int, question: str) -> Dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. Make sure to put the answer (and only answer) inside \\boxed{}."})

    return chats



def extract_ans_from_response(dataset_name: str, answer: str, eos: Optional[str] = None) -> str:
    """Extracts the final answer from a model response based on dataset-specific formatting.

    Args:
        dataset_name: Name of the dataset (e.g., 'GSM8k', 'ANLI')
        answer: Raw model response string
        eos: Optional end-of-sequence token to split on

    Returns:
        Extracted and cleaned answer string
    """
    if eos:
        answer = answer.split(eos)[0].strip()
    answer = answer.strip()
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    matches = boxed_pattern.findall(answer)
    if matches:
        answer = matches[-1].strip()
    
    if dataset_name == "GSM8k":
        answer = re.sub(r'[^0-9\-\.]', '', answer)
        # for remove_char in [',', '$', '%', 'g']:
        #     answer = answer.replace(remove_char, '')
        try:
            return int(answer)
        except ValueError:
            return "invalid"

    elif dataset_name == "ANLI":
        # Remove everything except alphabets and convert to lowercase
        answer = re.sub(r'[^a-zA-Z]', '', answer).lower()
        # if answer contain entailment, neutral, or contradiction -> answer = entailment, neutral, or contradiction. No need to exactly the same
        if 'entailment' in answer:
            return 'entailment'
        elif 'neutral' in answer:
            return 'neutral'
        elif 'contradiction' in answer:
            return 'contradiction'
        else:
            return "invalid"
        
    elif dataset_name == "FANTOM" or dataset_name == "HOTPOT":
        return answer

    
    elif dataset_name == "MATH":
        return answer
    
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")




def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def extract_ans_from_response_reasoning(answer: str, eos: Optional[str] = None) -> int:
    if eos:
        answer = answer.split(eos)[0].strip()
    answer = answer.strip()
    boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')
    matches = boxed_pattern.findall(answer)
    if matches:
        answer = matches[0].strip()
    
    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')
        if answer == "No Reason Provided".strip():
            return -1
        elif answer == "Reason Provided".strip():
            return 1
        else:
            return "invalid"

def check_max_token_length(qar_list, tokenizer):
    question_lengths = []
    qa_lengths = []

    for qar in tqdm(qar_list):
        q, qa = qar["q"], qar["qa"]
        q_encodings = tokenizer.apply_chat_template(q, truncation=False, padding=False, return_dict=True)
        qa_encodings = tokenizer.apply_chat_template(qa, truncation=False, padding=False, return_dict=True)
        question_lengths.append(len(q_encodings["input_ids"]))
        qa_lengths.append(len(qa_encodings["input_ids"]))
    
    max_q_length = max(question_lengths)
    max_qa_length = max(qa_lengths)
    mean_q_length = sum(question_lengths) / len(question_lengths)
    mean_qa_length = sum(qa_lengths) / len(qa_lengths)

    print(f"Max question length: {max_q_length}")
    print(f"Max QA length: {max_qa_length}")
    print(f"Mean question length: {mean_q_length}")
    print(f"Mean QA length: {mean_qa_length}")

    return max_q_length, max_qa_length


class BaseDataset_reward_training(Dataset):
    def __init__(self, tokenizer, repeat_time, gpu_server_num, max_length=1000, debug=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.repeat_time = repeat_time
        self.gpu_server_num = gpu_server_num
        self.debug = debug
        self.qa_list, self.q_list, self.a_list, self.answer_want, self.reward_list = self.load_data()
    
    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def __len__(self):
        return len(self.qa_list)
    
    def __getitem__(self, idx):
        txt = self.qa_list[idx]
        encodings_dict = self.tokenizer.apply_chat_template(txt, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True, add_generation_prompt = True)
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        txt_a = self.a_list[idx]
        encodings_dict_a = self.tokenizer.apply_chat_template(txt_a, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True)
        input_ids_a = torch.tensor(encodings_dict_a["input_ids"])
        txt_q = self.q_list[idx]
        encodings_dict_q = self.tokenizer.apply_chat_template(txt_q, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True)
        input_ids_q = torch.tensor(encodings_dict_q["input_ids"])
        if self.answer_want != []:
            txt_answer_want = self.answer_want[idx]
            encodings_dict_answer_want = self.tokenizer.apply_chat_template(txt_answer_want, truncation=True, max_length=self.max_length, padding="max_length", return_dict=True)
            input_ids_answer_want = torch.tensor(encodings_dict_answer_want["input_ids"])
        else:
            input_ids_answer_want = input_ids_a
        
        if self.reward_list != []:
            reward = self.reward_list[idx]
        else:
            reward = 1
            
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
            "question": input_ids_q,
            "answer": input_ids_a,
            "answer_want": input_ids_answer_want,
            "reward": torch.tensor(reward, dtype = torch.float32)
        }
    


class DatasetFactory_reward_training:
    @staticmethod
    def create_dataset(dataset_name, tokenizer, repeat_time, gpu_server_num, max_length=1000, debug=False):
        if dataset_name == 'GSM8k':
            return GSM8kDataset_reward_training(tokenizer, repeat_time, gpu_server_num, max_length, debug)
        elif dataset_name == 'ANLI':
            return ANLIDataset_reward_training(tokenizer, repeat_time, gpu_server_num, max_length, debug)
        elif dataset_name == "FANTOM":
            return FANTOMDataset_reward_training(tokenizer, repeat_time, gpu_server_num, max_length, debug)
        elif dataset_name == "HOTPOT":
            return HOTPOTDataset_reward_training(tokenizer, repeat_time, gpu_server_num, max_length, debug)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")



class GSM8kDataset_reward_training(BaseDataset_reward_training):
    def load_data(self):
        qa_list = []
        q_list = []
        a_list = []
        __ = []
        dataset = load_dataset('gsm8k', 'main', split="train")
        ## need to devide this
        
        ### need to change this part
        length_dataset = len(dataset)
        # importance = [1/80, 1/80, 1/150, 1/600, 1/600, 1/600]
        importance = [1/80, 1/80, 1/150]

        num = [int(length_dataset * importance[i] / sum(importance)) for i in range(len(importance))]
        num = [0] + num
        # cumulative num 
        num = [sum(num[:i+1]) for i in range(len(importance) + 1)]
        dataset = dataset.select(range(num[self.gpu_server_num], num[self.gpu_server_num + 1]))
        if self.debug:
            dataset = dataset.select(range(100))
        for sample in dataset:
            for _ in range(self.repeat_time):
                q_list.append([
                    {"role": "user", "content": "Question: " + sample["question"] + "\n\nProvide a short but precise reasoning for your solution. At the end, you MUST write the answer in the following format:\n\nAnswer: \\boxed{XX}\n\nPlease ensure that the final answer is always formatted this way."}])
                qa_list.append([
                    {"role": "user", "content": "Question: " + sample["question"] + "\n\nProvide a short but precise reasoning for your solution. At the end, you MUST write the answer in the following format:\n\nAnswer: \\boxed{XX}\n\nPlease ensure that the final answer is always formatted this way."}])
                a_list.append( [{"role": "assistant", "content": sample["answer"]}])
        return qa_list, q_list, a_list, __, __


class FANTOMDataset_reward_training(BaseDataset_reward_training):
    def load_data(self):
        with open("/home/ubuntu/cooperate-LLM/train_data_FANTOM.json", "r") as f:
            train_data = json.load(f)
        with open("/home/ubuntu/cooperate-LLM/test_data_FANTOM.json", "r") as f:
            test_data = json.load(f)
            
        def list_to_dict(data):
            keys = data[0].keys()  # Get keys from the first dictionary
            return {key: [d[key] for d in data] for key in keys}
        
        train_data_dict = list_to_dict(train_data)
        test_data_dict = list_to_dict(test_data)

        # Create Hugging Face datasets using the structured data
        train_dataset_hf = HFDataset.from_dict(train_data_dict)
        test_dataset_hf = HFDataset.from_dict(test_data_dict)        
        if self.gpu_server_num == 0:
            dataset = train_dataset_hf.select(range(0, len(train_dataset_hf)//3))
        elif self.gpu_server_num == 1:
            dataset = train_dataset_hf.select(range(len(train_dataset_hf)//3, 2*len(train_dataset_hf)//3))
        elif self.gpu_server_num == 2:
            dataset = train_dataset_hf.select(range(2*len(train_dataset_hf)//3, len(train_dataset_hf)))
        elif self.gpu_server_num == 3:
            dataset = test_dataset_hf
        
        qa_list = []
        q_list = []
        a_list = []
        wrong_a_list = []
        reward_list = []

        if self.debug:
            dataset = dataset.select(range(100))
        for sample in dataset:
            for _ in range(self.repeat_time):
                if _%2 == 1:
                    ans = sample["correct_answer"]
                else:
                    ans = sample["wrong_answer"]
                q_list.append([
                    {"role": "user", "content": "Conversation: " + sample["document"] + "\n\n Question: " + sample["question"] + "Answer the given question based on the conversation. When you answer, think carefully. "}])
                if "unaware" in ans or "not know" in ans or "not involved" in ans or "not aware" in ans:
                    qa_list.append([
                        {"role": "user", "content": "Pharaphrase the below sentences: \n\n " + ans}
                    ])
                else:
                    qa_list.append([
                        {"role": "user", "content": "Conversation: " + sample["document"] + "\n\n Question: " + sample["question"] + "Based on the conversation, respond thoughtfully to the question. \n Example Answer for this question: \n " + ans + " \n\n Please make sure your answer resembles the following example, but please do not copy it exactly. Especially, if the answer has terminology related to 'unaware' or not knowing the context, please maintain that."}])
                a_list.append([{"role": "assistant", "content": "Answer: \\boxed{" + sample["correct_answer"] + "}"}])
                wrong_a_list.append([{"role": "assistant", "content": "Answer: \\boxed{" + sample["wrong_answer"] + "}"}])
                reward_list.append(_%2)
        return qa_list, q_list, a_list, wrong_a_list, reward_list
    
    

class HOTPOTDataset_reward_training(BaseDataset_reward_training):
    def load_data(self):
        qa_list = []
        q_list = []
        a_list = []
        __ = []
        dataset = load_dataset("hotpotqa/hotpot_qa", 'distractor', trust_remote_code=True)['train']
        ## need to devide this
        
        ### need to change this part
        length_dataset = len(dataset)
        # 1/12 1/15 1/25 
        # importance = [1/80, 1/80, 1/150, 1/600, 1/600, 1/600]
        importance = [1/25, 1/25, 1/25, 1/25, 1/25, 1/25]

        num = [int(length_dataset * importance[i] / sum(importance)) for i in range(len(importance))]
        num = [0] + num
        # cumulative num 
        num = [sum(num[:i+1]) for i in range(len(importance) + 1)]
        dataset = dataset.select(range(num[self.gpu_server_num], num[self.gpu_server_num + 1]))
        if self.debug:
            dataset = dataset.select(range(400))
        formatted_dataset = []
        for d in dataset:
            sentences = d['context']['sentences']
            paragraph = [' '.join(s) for s in sentences]
            all_paragraph = '\n'.join(paragraph)
            if len(all_paragraph.split(' ')) >1000 :
                continue
            formatted_dataset.append({
                "paragraph": all_paragraph,
                "question": d['question'],
                "answer": d['answer'],
            })
        print("length", len(formatted_dataset))
        for sample in formatted_dataset:
            for _ in range(self.repeat_time):
                q_list.append([
                    {"role": "user", "content": "Paragraph: " + sample["paragraph"] + "\nQuestion: " + sample["question"] + "\n\nProvide a concise reasoning for your solution. At the end, you MUST write the answer in the following format:```XXX fill out your reason XXX \n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way."}])
                qa_list.append([
                    {"role": "user", "content": "Paragraph: " + sample["paragraph"] + "\nQuestion: " + sample["question"] + "\n\nProvide a concise reasoning for your solution. At the end, you MUST write the answer in the following format:```XXX fill out your reason XXX \n\nAnswer: \\boxed{ XXX }'''\n\nPlease ensure that the final answer is always formatted this way."}])
                a_list.append( [{"role": "assistant", "content": "Answer: \\boxed{" + sample["answer"] + "}"}])
        return qa_list, q_list, a_list, __, __


class ANLIDataset_reward_training(BaseDataset_reward_training):
    def load_data(self):
        qa_list = []
        q_list = []
        a_list = []
        answer_want = []
        ___ = []
        dataset = load_dataset('anli', split="train_r3")
        if self.gpu_server_num == 0:
            dataset = dataset.select(range(0, 37672))
        elif self.gpu_server_num == 1:
            dataset = dataset.select(range(37672, 53417))
        elif self.gpu_server_num == 2:
            dataset = dataset.select(range(53417, 68892))
        elif self.gpu_server_num == 3:
            dataset = dataset.select(range(68892, 84648))
        elif self.gpu_server_num == 4:
            dataset = dataset.select(range(84648, 100459))

        label = ['entailment', 'neutral', 'contradiction']
        if self.debug:
            dataset = dataset.select(range(100))
        
        for sample in dataset:
            for _ in range(self.repeat_time):
                for label_element in label:
                    q_list.append([
                        {"role": "user", "content": "Premise: " + sample["premise"] + "\n\n Hypothesis: " + sample["hypothesis"] + "\n\n Please choose the relationship between the given premise and hypothesis as one of the following: 'entailment', 'neutral', or 'contradiction'. Provide reasoning for your choice. You do not need to repeat premise and hypothesis in your response. At the end, write the final answer inside a box (e.g \\boxed{entailment}, \\boxed{neutral}, and \\boxed{contradiction}."}])
                    
                    qa_list.append([
                        {"role": "user", "content": "Premise: " + sample["premise"] + "\n\n Hypothesis: " + sample["hypothesis"] + "\n\n Please choose the relationship between the given premise and hypothesis as one of the following: 'entailment', 'neutral', or 'contradiction'. Provide reasoning for your choice. You do not need to repeat premise and hypothesis in your response. Please generate the reasoning assuming that the relationship between the given premise and hypothesis is " + label_element + ". You should strongly believe that the relationship is " + label_element + ". Also, do not mention that I asked you to generate the reasoning for the relationship between the given premise and hypothesis as " + label_element + ". Also, not start with declaring your opinion. Just provide the reasoning for the relationship between the given premise and hypothesis as " + label_element + ". You do not need to repeat premise and hypothesis in your response. At the end, write the final answer inside a box (e.g \\boxed{entailment}, \\boxed{neutral}, and \\boxed{contradiction}. "  }             
                        ])
                    a_list.append([{"role": "assistant", "content": sample["reason"] + "\n\n \\boxed{" + label[sample['label']] + "}"}])
                    answer_want.append([{"role": "user", "content": "\\boxed{" + label_element+ "}"}])
        return qa_list, q_list, a_list, answer_want, ___


def cleanse_gsm8k_data(
    input_files: Optional[List[str]] = None,
    reference_file: str = "QAC_GSM8k_combined_filtered.json",
    output_file: str = "QAC_GSM8k_cleansed.json",
    min_answers: int = MIN_ANSWERS
) -> Dict[str, Dict]:
    """
    Cleanse GSM8K data by combining multiple config files, cleaning answers,
    and ensuring minimum number of answers per question.

    Args:
        input_files: List of paths to input JSON config files. If None, uses default pattern
        reference_file: Path to reference JSON file for additional answers (default: QAC_GSM8k_combined_filtered.json)
        output_file: Path to save the cleansed data (default: QAC_GSM8k_cleansed.json)
        min_answers: Minimum number of answers to maintain per question (default: 30)

    Returns:
        Dictionary containing the cleansed configuration
    """
    import os
    import json
    import random
    from .cleansing import combine_configs, answer_cleansing

    # Set default input files if not provided
    if input_files is None:
        input_files = [
            "QAC_filtered_GSM8k__0.json",
            "QAC_filtered_GSM8k__1.json",
            "QAC_filtered_GSM8k__2.json",
            "QAC_filtered_GSM8k__3.json",
            "QAC_filtered_GSM8k__4.json",
            "QAC_filtered_GSM8k__5.json",
            "QAC_filtered_GSM8k__6.json",
            "QAC_filtered_GSM8k__7.json"
        ]

    # Load all input configurations
    configs = {}
    for i, file_path in enumerate(input_files):
        try:
            with open(file_path, 'r') as f:
                config_key = f'config_{i}'
                configs[config_key] = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
            continue

    # Combine configurations
    combined_config = combine_configs(configs)

    # Load reference data
    try:
        with open(reference_file, 'r') as f:
            ref = json.load(f)
    except FileNotFoundError:
        print(f"Reference file not found: {reference_file}")
        ref = {}
    except json.JSONDecodeError:
        print(f"Error decoding reference JSON: {reference_file}")
        ref = {}

    # Process each question
    new_config = {}
    for question in combined_config.keys():
        new_config[question] = {}
        gen_anss = combined_config[question]["generated_answers"]
        new_gen_anss = []
        new_gen_rewards = []

        if len(gen_anss) != len(combined_config[question]["rewards"]):
            print(f"Length mismatch for question: {question}")
            continue

        # Clean existing answers
        for i, gen_ans in enumerate(gen_anss):
            new_gen = answer_cleansing(answer_cleansing(gen_ans))
            if len(new_gen.strip()) != 0:
                new_gen_anss.append(new_gen)
                new_gen_rewards.append(combined_config[question]["rewards"][i])

        # Add answers from reference if needed
        if len(new_gen_anss) < min_answers:
            if question in ref:
                try:
                    len_ref = len(ref[question]["generated_answers"])
                    num_to_add = min_answers - len(new_gen_anss)
                    random_idx = random.sample(range(len_ref), num_to_add)
                    for idx in random_idx:
                        new_gen_anss.append(ref[question]["generated_answers"][idx])
                        new_gen_rewards.append(ref[question]["rewards"][idx])
                except (KeyError, ValueError) as e:
                    print(f"Error processing reference for question: {question} - {e}")

        new_config[question]["generated_answers"] = new_gen_anss
        new_config[question]["rewards"] = new_gen_rewards

    # Save the cleansed configuration
    with open(output_file, 'w') as f:
        json.dump(new_config, f, indent=4)

    print(f"Processed {len(new_config)} questions")
    return new_config