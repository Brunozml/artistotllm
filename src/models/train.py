"""
this script is not currently being used. its objective
is to potentially fine tune a model to an author's style.
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_config, PeftModel, LoraConfig, TaskType
from datasets import Dataset
import yaml
from typing import Dict, List

def load_config(config_path: str = "configs/model_config.yaml") -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(split: str) -> List[str]:
    with open(f'data/processed/ethics_{split}.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(texts: List[str], tokenizer: AutoTokenizer) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset

def main():
    # Load configuration
    config = load_config()
    
    # Initialize model and tokenizer
    model_name = config['model']['base_model']
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['peft']['r'],
        lora_alpha=config['peft']['lora_alpha'],
        lora_dropout=config['peft']['lora_dropout'],
        target_modules=config['peft']['target_modules']
    )
    
    model = get_peft_model(model, peft_config)
    
    # Load and prepare datasets
    train_texts = load_data('train')
    val_texts = load_data('val')
    
    train_dataset = prepare_dataset(train_texts, tokenizer)
    val_dataset = prepare_dataset(val_texts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("./aristotle_model")

if __name__ == "__main__":
    main() 