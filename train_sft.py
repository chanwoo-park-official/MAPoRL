import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from peft import get_peft_model


from utils.dataset import DatasetFactory
from utils.utils_general import set_seed, load_config_from_python, check_and_add_special_tokens

'''
nohup deepspeed train_sft.py --config_file config/sft_config/config_sft_phi3.py &
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
    if base_model == "microsoft/Phi-3-mini-128k-instruct":
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto",  trust_remote_code=True)
    elif base_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(base_model)

    if peft:
        model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    
    #check tokenizer has pad_token, and if not, set pad_token = eos_token
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer, token_id_maps = check_and_add_special_tokens(tokenizer, special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"  # Explicitly setting right padding


    train_dataset = DatasetFactory.create_dataset(dataset_name, tokenizer, "train", max_input_length, debug)
    test_dataset = DatasetFactory.create_dataset(dataset_name, tokenizer, "test", max_input_length, debug)
    print(train_dataset[0])
    print(len(train_dataset))
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        logging_strategy = logging_strategy,
        eval_accumulation_steps=eval_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=half_precision_backend,
        fp16=fp16,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=False,
        logging_steps=logging_steps,
        report_to=report_to,  
        deepspeed=deepspeed,
        run_name= run_name,   
        save_strategy = save_strategy
    )    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
    )    


    trainer.train()
    trainer.save_model(output_dir)


