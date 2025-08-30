import ast
import random
import numpy as np
import torch
import json
import importlib.util
import sys
import io
import boto3
import yaml
import torch.distributed as dist
from safetensors.torch import load_file
import torch.nn as nn
import os
import datetime
import hashlib
import tarfile
import zipfile
import re
import pandas as pd
import requests
from tqdm import tqdm
BYTES_TO_GB = 1024**3

def extract_last_number_after_slash(path: str) -> int:
    """
    Extract the last number that appears after a slash in a path.

    Args:
        path: File or directory path containing numbers after slashes

    Returns:
        The last number found after a slash

    Raises:
        ValueError: If no numbers are found after slashes in the path
    """
    numbers = re.findall(r'/(\d+)', path)

    if numbers:
        return int(numbers[-1])
    else:
        raise ValueError(f"No numbers found after slashes in the path: {path}")




def find_latest_folder(base_dir: str) -> Optional[str]:
    """
    Find the latest (highest numbered) folder in a directory.

    Args:
        base_dir: Base directory to search for numbered subdirectories

    Returns:
        Path to the latest folder, or None if no numeric directories found
    """
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    numeric_dirs = []
    for d in subdirs:
        if d.isdigit():  # Check if the directory name is a number
            numeric_dirs.append((int(d), d))
        else:
            # For directory names that might include non-numeric characters
            match = re.search(r'\d+', d)
            if match:
                numeric_dirs.append((int(match.group()), d))

    # Sort directories based on the numeric value
    numeric_dirs.sort(key=lambda x: x[0], reverse=True)

    if numeric_dirs:
        return os.path.join(base_dir, numeric_dirs[0][1])
    else:
        return None


def parse_list(string: str) -> list:
    """
    Safely parse a string representation of a list using ast.literal_eval.

    Args:
        string: String representation of a list (e.g., "[1, 2, 3]")

    Returns:
        The parsed list object

    Raises:
        ValueError: If the string cannot be safely evaluated as a list
    """
    return ast.literal_eval(string)  # Safely evaluate the string        
def print_gpu_memory_usage() -> None:
    """
    Print current GPU memory usage for all available GPUs.

    Displays allocated and cached memory for each GPU device.
    """
    print("GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / BYTES_TO_GB:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / BYTES_TO_GB:.2f} GB")

def unzip_file(file_path, directory='.'):
    if file_path.endswith(".zip"):
        target_location = os.path.join(directory, os.path.splitext(os.path.basename(file_path))[0])
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_location)
    elif file_path.endswith(".tar.gz"):
        target_location = os.path.join(directory, os.path.basename(file_path).split(".")[0])
        with tarfile.open(file_path) as tar:
            tar.extractall(target_location)

    return target_location


def is_effectively_integer(num: float) -> bool:
    """
    Check if a number is effectively an integer.

    Args:
        num: The number to check (int or float)

    Returns:
        True if the number is an integer or a float that represents an integer
    """
    if isinstance(num, int):
        return True
    return isinstance(num, float) and num.is_integer()


def mark_built(path, version_string="1.0"):
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp plus a version description string.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)



def download_and_check_hash(url, filename, expected_hash, version, directory='data', chunk_size=1024 * 1024 * 10):
    # Download the file
    response = requests.get(url, stream=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
    except:
        print("Couldn't get content-length from response headers, using chunk_size instead")
        total_size = chunk_size
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    data = b''
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        data += chunk
    progress_bar.close()

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to disk
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(data)

    # Calculate the hash of the downloaded data
    sha256_hash = hashlib.sha256(data).hexdigest()

    # Compare the calculated hash to the expected hash
    if sha256_hash != expected_hash:
        print('@@@ Downloaded file hash does not match expected hash!')
        raise RuntimeError

    return file_path



def build_data(resource, directory='data'):
    # check whether the file already exists
    if resource.filename.endswith('.tar.gz'):
        resource_dir = os.path.splitext(os.path.splitext(os.path.basename(resource.filename))[0])[0]
    else:
        resource_dir = os.path.splitext(os.path.basename(resource.filename))[0]
    file_path = os.path.join(directory, resource_dir)

    built = check_built(file_path, resource.version)

    if not built:
        # Download the file
        file_path = download_and_check_hash(resource.url, resource.filename, resource.expected_hash, resource.version,
                                            directory)

        # Unzip the file
        if resource.zipped:
            built_location = unzip_file(file_path, directory)
            # Delete the zip file
            os.remove(file_path)
        else:
            built_location = file_path

        mark_built(built_location, resource.version)
        print("Successfully built dataset at {}".format(built_location))
    else:
        print("Already built at {}. version {}".format(file_path, resource.version))
        built_location = file_path

    return built_location


def check_built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version is regarded as not built.
    """
    fname = os.path.join(path, '.built')
    if not os.path.isfile(fname):
        return False
    else:
        with open(fname, 'r') as read:
            text = read.read().split('\n')
        return len(text) > 1 and text[1] == version_string


class DownloadableFile:
    def __init__(self, url, filename, expected_hash, version="1.0", zipped=True):
        self.url = url
        self.filename = filename
        self.expected_hash = expected_hash
        self.zipped = zipped
        self.version = version


def list_folders(directory, exclude_hidden=True, name_filter=None):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        
        folders = []
        for entry in entries:
            full_path = os.path.join(directory, entry)
            # Check if the entry is a directory
            if os.path.isdir(full_path):
                # Optionally exclude hidden folders
                if exclude_hidden and entry.startswith('.'):
                    continue
                # Optionally filter by name pattern
                if name_filter and name_filter not in entry:
                    continue
                folders.append(entry)
        
        return folders
    except Exception as e:
        return str(e)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def setup_distributed():
    dist.init_process_group(backend='nccl')

# Clean up the process group
def cleanup_distributed():
    dist.destroy_process_group()
# Function to load safetensors part onto the specified device
def load_safetensors_part(path, device):
    state_dict = load_file(path, device='cpu')  # Load onto CPU first
    # Move each tensor to the specified device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device)
    return state_dict


def json_loader(path):
    with open(path, "r") as f:
        return json.load(f)
    

def load_config_from_python(config_file):
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_config_from_yaml(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def check_and_add_special_tokens(tokenizer, special_tokens):
    # Check if the special tokens are new
    new_tokens = [token for token in special_tokens['additional_special_tokens'] if token not in tokenizer.get_vocab()]
    
    # Add the new special tokens
    if new_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

    # Get token IDs for all special tokens
    token_ids = tokenizer.convert_tokens_to_ids(special_tokens['additional_special_tokens'])
    
    # Create a dictionary mapping tokens to their IDs
    token_id_map = dict(zip(special_tokens['additional_special_tokens'], token_ids))

    # Print the tokenizer's vocabulary size and token IDs
    print("Tokenizer's Vocabulary Size:", len(tokenizer))
    print("Special Tokens and their IDs:")
    for token, token_id in token_id_map.items():
        print(f"Token: {token}, ID: {token_id}")

    return tokenizer, token_id_map



def execute_python_code(code):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    
    # Save the current standard output
    old_stdout = sys.stdout
    
    try:
        # Redirect standard output to the buffer
        sys.stdout = buffer
        
        # Execute the code
        exec(code)
        
        # Get the printed output
        output = buffer.getvalue()
        
    except Exception as e:
        output = f"Error: {str(e)}"
    
    finally:
        # Restore the original standard output
        sys.stdout = old_stdout
    
    return output




def response_from_bedrock(text, max_token = 5000, modelID = "anthropic.claude-3-sonnet-20240229-v1:0"):
    # Initialize the Bedrock Runtime client
    runtime = boto3.client("bedrock-runtime", region_name='us-east-1')

    # Construct the request body
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_token,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    }

    # Convert the body to JSON
    body_json = json.dumps(body)

    
    # Invoke the model
    trial_count = 0
    while True:
        try:
            response = runtime.invoke_model(
                modelId=modelID,
                contentType="application/json",
                accept="application/json",
                body=body_json
            )

            # Process the response
            response_body = json.loads(response["body"].read())
            break
        except Exception as e:
            trial_count += 1
            print(e)
            print("Retrying... Trial", trial_count)
            continue

    return response_body





def check_initial_weights(policy, adapter_name):
    adapter_params = [param for name, param in policy.named_parameters() if adapter_name in name]
    for param in adapter_params:
        if not torch.all(param == 0):
            return False
    return True

# Set the weights of the "ref" adapter to zero and freeze them
def zero_and_freeze_adapter(policy, adapter_name):
    adapter_params = [param for name, param in policy.named_parameters() if adapter_name in name]
    for param in adapter_params:
        nn.init.zeros_(param)
        param.requires_grad = False

def make_grad_work_adapter(policy, adapter_name):
    adapter_params = [param for name, param in policy.named_parameters() if adapter_name in name]
    for param in adapter_params:
        param.requires_grad = True

def make_grad_nowork_adapter(policy, adapter_name):
    adapter_params = [param for name, param in policy.named_parameters() if adapter_name in name]
    for param in adapter_params:
        param.requires_grad = False


