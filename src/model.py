import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import BitsAndBytesConfig

class Model:
    def __init__(self, config, finetuning_filepath='twitter_data.pkl'):
        self.model_name = config["model"]
        self.fine_tuned_dir = f"{config['finetuning_dir']}{config['model']}_finetuned_{os.path.splitext(finetuning_filepath)[0]}"
        self.finetuning_filepath = finetuning_filepath
        self.finetuned = config["finetuned"]
        self.model = None
        self.tokenizer = None
        self.load()

    def load(self):
        if not self.finetuned:
            print(f"Loading base model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif os.path.exists(self.fine_tuned_dir):
            print(f"Loading fine-tuned model from {self.fine_tuned_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(self.fine_tuned_dir, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_dir)
        else:
            print(f"Fine-tuning and saving model to {self.fine_tuned_dir}")
            self.finetune_model()
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def finetune_model(self):
        checkpoint_dir = self.fine_tuned_dir.replace("_finetuned", "_checkpoints")       

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, legacy=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        df = pd.read_pickle(self.finetuning_filepath)
        df_train = df[df['training'] == 1]

        def format_conversation(examples):
            return f"You are @{examples['username']}. Respond to: '{examples['reply_to']}'\n\n{examples['message']}"

        dataset = Dataset.from_pandas(df_train)
        dataset = dataset.map(lambda x: {"text": format_conversation(x)})

        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["text"],  
                padding="max_length", 
                truncation=True, 
                max_length=256
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config,  
            torch_dtype=torch.float16,
            device_map="auto"
        )

        #model.resize_token_embeddings(len(tokenizer))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,  
            num_train_epochs=2,
            fp16=True,
            optim="adamw_torch",
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            #evaluation_strategy="no",
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint-")]

        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
            print(f"🔄 Resuming training from checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            print("🆕 No checkpoint found. Starting fresh.")
            trainer.train()
        
        if not hasattr(model, "generation_config"):
            model.generation_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }

        model.save_pretrained(self.fine_tuned_dir)
        tokenizer.save_pretrained(self.fine_tuned_dir)
        self.model = model
        self.tokenizer = tokenizer
        print("✅ LoRA fine-tuning complete. Model saved!")



def generate_response(model: Model, input_tweet, config):
    print("to be implemented")
    return "test"

