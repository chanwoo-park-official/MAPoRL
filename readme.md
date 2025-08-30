
# MaPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning

It is based on the [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) package. We are also planning to use Verl to make this easy to use, especially for the inference time scaling, so stay tuned!

## Reward Verifier Training

First, if you want to train the verifier, you should use reward_gen_data.py
Then, I additionally had the dataset balancing to make the verifier understand balanced right and wrong answers -- especially if the verifier only has the right answer for easy questions and the wrong answer for the hard questions, they might be trained to classify the difficulty of the questions. That can be balanced by reward_data_balancing.py, but you can also change it by whatever you want.

Additionally, we might only use the answer label for this -- then you do not need reward verifier training.

For the reward_server.py, we used a separate server for the reward (so that we get information about the reward from a separate 1 A24 GPU.)

## Multi-Agent PPO Training

The core training script `train_ppo_v2_multi_agent_multi_model.py` implements a multi-agent cooperative learning framework with the following key features:

### Key Features
- **Multi-agent training**: Support for configurable number of agents working cooperatively
- **Different models per agent**: Support for using different model architectures for different agents

### Dataset Support
- **GSM8K**: Mathematical reasoning with code execution validation
- **ANLI**: Natural language inference with entailment/contradiction classification

### Custom Extensions

This project extends the TRL package with several key additions:

1. **`trl/trl/trainer/ppov2_trainer_multi_different_model.py`**: Enhanced PPOv2 trainer supporting multiple agents with different model architectures
2. **`trl/trl/trainer/utils_multi_unified_chat.py`**: Unified chat template utilities for multi-agent interactions
3. **`trl/trl/trainer/utils_multi_unified.py`**: General utilities for multi-agent training coordination

### Configuration
Configuration files are located in `config/ppo_config/` with various setups for different scenarios:
- Multi-agent with different models
- Turn-based training
- Various dataset configurations
