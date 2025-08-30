import os
import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments

from utils.utils_general import set_seed, load_config_from_python
from utils.dataset import DatasetFactory_reward, DataCollator_reward
from utils.reward_model import RewardModel
from torch.utils.data import DataLoader

class NoShuffleTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,  # Force no shuffling
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

'''
deepspeed train_reward.py --config_file config/reward_config/config_reward_lr_1e-04_GSM8k.py >output_reward_training.log 2>&1
deepspeed train_reward.py --config_file config/reward_config/config_reward_lr_1e-04_FANTOM.py >output_reward_training.log 2>&1
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    args = parser.parse_args()
    config = load_config_from_python(args.config_file)
    globals().update(config)
    set_seed(random_seed)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = RewardModel(sft_model = sft_model, base_model= base_model, lora=peft, peft_config=peft_config, dataset_name = dataset_name)

    train_dataset = DatasetFactory_reward.create_dataset(dataset_name, tokenizer, "train", max_input_length, test_len, debug)
    val_dataset = DatasetFactory_reward.create_dataset(dataset_name, tokenizer, "test", max_input_length, test_len, debug)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        logging_strategy = logging_strategy,
        eval_accumulation_steps=eval_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        bf16=bf16,
        fp16=fp16,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=False,
        logging_steps=logging_steps,
        report_to=report_to,  
        deepspeed=deepspeed,
        run_name= run_name,   
        save_strategy = save_strategy,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator_reward(),
    )
    
    trainer.train()
    trainer.save_model(output_dir)
