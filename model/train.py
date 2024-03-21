#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:54:51 2024

@author: sui
"""
import numpy as np
from datasets import Dataset
import json
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments
import torch
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer

data_path = "/Users/tianlimin/dataset/admin.json"

# TrainingArguments parameters
output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
save_steps = 1
num_train_epochs = 4
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 20
warmup_ratio = 0.03
lr_scheduler_type = "linear"

# LoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

with open(data_path,'r',encoding='utf-8') as file:
    data_content = file.read()
user_stories_data = json.loads(data_content)
dataset = Dataset.from_dict({"text":[item["text"] for item in user_stories_data],
                             "label":[item["label"] for item in user_stories_data]})
tokenizer = AutoTokenizer.from_pretrained("/Users/tianlimin/LLM/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM("/Users/tianlimin/LLM/TinyLlama-1.1B-Chat-v1.0").to('mps')

tokenized_inputs = dataset.map(lambda x: tokenizer(x["text"]), batched=True, remove_columns=["text", "label"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
max_source_length = int(np.percentile(input_lenghts, 85))

model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_source_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  
model_to_save.save_pretrained("outputs")