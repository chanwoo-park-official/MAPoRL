
# MaPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning

(Aug 29, 2025) It is based on the [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) package. We are also planning to use Verl to make this easy to use, especially for the inference time scaling, so stay tuned!

## Reward Verifier Training

First, if you want to train the verifier, you should start with reward_gen_data.py.
After that, I applied dataset balancing so the verifier could learn to distinguish between correct and incorrect answers in a balanced way. Without balancing, the verifier might overfit — for example, if it only sees correct answers for easy questions and incorrect answers for hard ones, it may end up classifying based on question difficulty rather than answer correctness. This balancing can be done with reward_data_balancing.py, though you can adjust or implement your own method if preferred.

Additionally, in some cases we might only rely on the answer label directly. In that scenario, reward verifier training would not be necessary.

Finally, regarding reward_server.py, we used a separate server for the reward process. This allowed us to offload reward computations to a dedicated A24 GPU, ensuring efficiency and separation from the main training loop.

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
