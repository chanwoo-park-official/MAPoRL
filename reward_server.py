"""
Reward Model Server

A Flask-based server for serving reward model predictions.
Handles batched inference for reward scoring of query-response pairs.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from utils.reward_model import RewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from safetensors.torch import load_file
import threading

app = Flask(__name__)

# Configuration constants
REWARD_MODEL_DIR = "rm_phi3_gsm8k"
BASE_MODEL = "microsoft/Phi-3-mini-128k-instruct"
BATCH_SIZE = 8
DEVICE = "cuda:0"
HOST = "0.0.0.0"
PORT = 8501

# PEFT configuration
PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Global model and tokenizer instances
rw_model = None
rw_tokenizer = None

# Thread lock for model inference
lock = threading.Lock()

def load_tokenizer():
    """
    Load the tokenizer for the reward model.
    Sets pad_token to eos_token if pad_token is not defined.
    """
    global rw_tokenizer
    if rw_tokenizer is None:
        try:
            rw_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if rw_tokenizer.pad_token is None:
                rw_tokenizer.pad_token = rw_tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")


def load_model():
    """
    Load the reward model with PEFT configuration.
    Loads model weights from safetensors file and moves to GPU.
    """
    global rw_model

    if rw_model is None:
        try:
            rw_model = RewardModel(sft_model=BASE_MODEL, base_model=BASE_MODEL, lora=True, peft_config=PEFT_CONFIG)
            state_dict = load_file(f'{REWARD_MODEL_DIR}/model.safetensors')
            rw_model.load_state_dict(state_dict, strict=False)

            # Move model to GPU if available
            rw_model.half()
            rw_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load reward model: {e}")

# Initialize model and tokenizer on startup
load_model()
load_tokenizer()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle reward prediction requests.

    Expects JSON payload with 'query_responses' containing chat messages.
    Returns reward scores for each query-response pair.

    Request format:
        {"query_responses": [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...]}

    Returns:
        {"result": [score1, score2, ...]}
    """
    global rw_model
    with lock:
        try:
            # Get input data
            data = request.json
            if not data or 'query_responses' not in data:
                return jsonify({'error': 'Missing query_responses in request body'}), 400

            query_responses = data['query_responses']
            if not isinstance(query_responses, list) or not query_responses:
                return jsonify({'error': 'query_responses must be a non-empty list'}), 400
        
            pad_token_id = rw_tokenizer.pad_token_id
            query_responses = rw_tokenizer.apply_chat_template(
                                                query_responses,
                                                add_generation_prompt=True,
                                                return_tensors="pt",
                                                padding=True
                                            ).to(DEVICE)

            batch_size = BATCH_SIZE
            all_results = []

            # Divide query_responses by batch size
            for batch_start in range(0, query_responses.size(0), batch_size):
                batch_end = min(batch_start + batch_size, query_responses.size(0))
                batch_query_responses = query_responses[batch_start:batch_end]
                with torch.no_grad():
                    attention_mask = batch_query_responses != pad_token_id
                    reward_last_token = rw_model(input_ids=batch_query_responses, attention_mask=attention_mask, inference=True)["end_scores"]
                    all_results.extend(reward_last_token.cpu().tolist())
                    del batch_query_responses, reward_last_token, attention_mask

            return jsonify({'result': all_results})

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
