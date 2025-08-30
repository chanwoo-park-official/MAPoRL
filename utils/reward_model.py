import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

class RewardModel(nn.Module):
    def __init__(self, sft_model, base_model, lora=False, peft_config= None, dataset_name = None):
        super().__init__()
        self.sft_model = sft_model
        self.base_model = base_model
        self.model = AutoModelForCausalLM.from_pretrained(sft_model, torch_dtype=torch.bfloat16,  trust_remote_code=True, device_map = "auto")       
        if lora == True:
            self.model = get_peft_model(self.model, peft_config)
        self.config = self.model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        last_param_device = next(self.model.parameters()).device
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False).to(last_param_device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.sigmoid = nn.Sigmoid()
        self.dataset_name = dataset_name

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        rewards=None,
        position_ids=None,
        inputs_embeds=None,
        labels = None,
        inference = False,
    ):

        loss = None
        # print(" ".join(decoded_text), "input_ids")
        if self.base_model in ["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-128k-instruct", "Qwen/Qwen2.5-3B-Instruct"]:
            transformer_outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds, output_hidden_states = True
            )
 
            hidden_states = transformer_outputs.hidden_states[-1]
        else:
            raise ValueError(f"Model {self.base_model} not implemented for the reward modeling.")
        # hidden_states to self.v_head device
        hidden_states = hidden_states.to(self.v_head.weight.device)
        estimated_rewards = self.sigmoid(self.v_head(hidden_states).squeeze(-1))
        end_scores = []

        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] 
        loss = 0
        last_token_loss = 0
        loss_fn = nn.BCELoss(reduction = "mean")
        for i in range(bs):            
            if self.base_model == "meta-llama/Meta-Llama-3-8B-Instruct":
                inds = (input_ids[i] == self.PAD_ID).nonzero()
                if len(inds) <= 1:
                    start_ind = 0
                    end_ind = input_ids.shape[1]
                else:
                    end_ind = inds[1].item() if len(inds) > 0 else input_ids.shape[1]
                    start_ind = inds[0].item() if len(inds) > 0 else 0 
                # Assume that we are using chat-model, so there is a PAD_ID at the end of the question sequence. We might need to re-implement here. 
            elif self.base_model == "microsoft/Phi-3-mini-128k-instruct":
                start_inds = (input_ids[i] == self.tokenizer.convert_tokens_to_ids('<|assistant|>')).nonzero()
                if len(start_inds) == 0:
                    start_ind = 0
                else:
                    start_ind = start_inds[0].item()
                inds = ((input_ids[i] == self.PAD_ID) & (torch.arange(input_ids.shape[1], device=input_ids.device) >= start_ind)).nonzero()
                if len(inds) == 0:
                    end_ind = input_ids.shape[1]
                else:
                    end_ind = inds[0].item()
            elif self.base_model == "Qwen/Qwen2.5-3B-Instruct":
                start_inds = (input_ids[i] == 151644).nonzero()
                if len(start_inds) <= 2:
                    start_ind = 0
                else:
                    start_ind = start_inds[2].item()
                inds = ((input_ids[i] == self.PAD_ID) & (torch.arange(input_ids.shape[1], device=input_ids.device) >= start_ind)).nonzero()
                if len(inds) == 0:
                    end_ind = input_ids.shape[1]
                else:
                    end_ind = inds[0].item()
                # print(start_ind, end_ind, input_ids[i] , "start_ind, end_ind")
            truncated_estimated_reward = estimated_rewards[i][start_ind:end_ind]
            end_scores.append(truncated_estimated_reward[-1])

            if not inference:
                truncated_reward = rewards[i].expand(truncated_estimated_reward.shape).to(truncated_estimated_reward.dtype)
            # if i == 0:
            #     print(input_ids[i][start_ind:end_ind], self.tokenizer.batch_decode(input_ids[i][start_ind:end_ind], skip_special_tokens=True), truncated_estimated_reward, truncated_reward, "input_ids, text, estimated_reward, reward")
                loss += loss_fn(truncated_estimated_reward, truncated_reward)
                last_token_loss += loss_fn(truncated_estimated_reward[-1], truncated_reward[-1])
        if not inference:
            loss = loss / bs

            last_token_loss = last_token_loss / bs
        # if self.dataset_name == "FANTOM":
        #     loss = last_token_loss

        if not inference:
            end_scores = torch.stack(end_scores)

        if inference:
            end_scores = torch.stack(end_scores)
            return {"end_scores": end_scores}

        return {
            "loss": loss,
            "last_token_loss": last_token_loss,
            "end_scores": end_scores,
        }




class RewardModel2(nn.Module):
    def __init__(self, sft_model, base_model, lora=False, peft_config= None):
        super().__init__()
        self.sft_model = sft_model
        self.base_model = base_model
        self.model = AutoModelForCausalLM.from_pretrained(sft_model, torch_dtype=torch.bfloat16,  trust_remote_code=True)
        if lora == True:
            self.model = get_peft_model(self.model, peft_config)
        self.config = self.model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        rewards=None,
        position_ids=None,
        inputs_embeds=None,
        labels = None,
    ):
        loss = None
        if self.base_model in ["meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-mini-128k-instruct"]:
            transformer_outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds, output_hidden_states = True
            )
            hidden_states = transformer_outputs.hidden_states[-1]
        else:
            raise ValueError(f"Model {self.base_model} not implemented for the reward modeling.")

        estimated_rewards = self.sigmoid(self.v_head(hidden_states).squeeze(-1))
        end_scores = []

        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] 
        loss = 0
        last_token_loss = 0
        inference = False
        loss_fn = nn.BCELoss(reduction = "mean")
        for i in range(bs):            
            if self.base_model == "meta-llama/Meta-Llama-3-8B-Instruct":
                inds = (input_ids[i] == self.PAD_ID).nonzero()
                if len(inds) <= 1:
                    start_ind = 0
                    end_ind = input_ids.shape[1]
                else:
                    end_ind = inds[1].item() if len(inds) > 0 else input_ids.shape[1]
                    start_ind = inds[0].item() if len(inds) > 0 else 0 
                # Assume that we are using chat-model, so there is a PAD_ID at the end of the question sequence. We might need to re-implement here. 
            elif self.base_model == "microsoft/Phi-3-mini-128k-instruct":
                start_inds = (input_ids[i] == self.tokenizer.convert_tokens_to_ids('<|assistant|>')).nonzero()
                if len(start_inds) == 0:
                    start_ind = 0
                else:
                    start_ind = start_inds[0].item()
                inds = (input_ids[i] == self.PAD_ID).nonzero()
                if len(inds) == 0:
                    end_ind = input_ids.shape[1]
                else:
                    end_ind = inds[0].item()
                # print(start_ind, end_ind, input_ids[i] , "start_ind, end_ind")
            truncated_estimated_reward = estimated_rewards[i][start_ind:end_ind]
            truncated_reward = rewards[i].expand(truncated_estimated_reward.shape).to(truncated_estimated_reward.dtype)
            end_scores.append(truncated_estimated_reward[-1])
            if i == 0:
                print(input_ids[i][start_ind:end_ind], self.tokenizer.batch_decode(input_ids[i][start_ind:end_ind], skip_special_tokens=True), truncated_estimated_reward, truncated_reward, "input_ids, text, estimated_reward, reward")
            loss += loss_fn(truncated_estimated_reward, truncated_reward)
            last_token_loss += loss_fn(truncated_estimated_reward[-1], truncated_reward[-1])
        loss = loss / bs
        last_token_loss = last_token_loss / bs

        if not inference:
            end_scores = torch.stack(end_scores)

        if inference:
            end_scores = torch.stack(end_scores)
            return {"end_scores": end_scores}

        return {
            "loss": loss,
            "last_token_loss": last_token_loss,
            "end_scores": end_scores,
        }
