import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import BitsAndBytesConfig
import numpy as np

class Model:
    def __init__(self, config, finetuning_filepath):
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
            
            # REMOVE 8-bit quantization to avoid the hang
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✅ Base model loaded")
            
        elif os.path.exists(self.fine_tuned_dir):
            print(f"Loading fine-tuned model from {self.fine_tuned_dir}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.fine_tuned_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_dir)
            self._configure_generation()
            print("✅ Fine-tuned model loaded")
            
        else:
            print(f"Fine-tuning and saving model to {self.fine_tuned_dir}")
            self.finetune_model()
            
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _configure_generation(self):
        """Configure generation parameters to reduce repetition"""
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = 0.8
            self.model.generation_config.top_p = 0.9
            self.model.generation_config.repetition_penalty = 1.15
            self.model.generation_config.no_repeat_ngram_size = 3
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def finetune_model(self):
        checkpoint_dir = self.fine_tuned_dir.replace("_finetuned", "_checkpoints")       

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        # Check if this is an Apertus model for minimal adjustments
        is_apertus = "apertus" in self.model_name.lower() or "swiss-ai" in self.model_name.lower()

        # MINIMAL CHANGE 1: Apertus needs trust_remote_code, others use original settings
        if is_apertus:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=True,
                trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False, 
                legacy=True
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load and prepare data
        df = pd.read_pickle(self.finetuning_filepath)
        df_train = df[df['training'] == 1].reset_index(drop=True)
        
        # Shuffle the data to prevent overfitting to order
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train and validation
        train_size = int(0.9 * len(df_train))
        df_train_split = df_train[:train_size]
        df_val_split = df_train[train_size:]
        
        print(f"Training samples: {len(df_train_split)}")
        print(f"Validation samples: {len(df_val_split)}")

        def format_conversation(examples):
            # Improved prompt format with clear separation
            prompt = f"You are @{examples['username']}. Respond to the following social media messge:"
            context = f"Social media message: {examples['reply_to']}"
            response = examples['message']
            
            # Use clear separators and end token
            return f"{prompt}\n{context}\nResponse: {response}{tokenizer.eos_token}"

        # Create datasets
        train_dataset = Dataset.from_pandas(df_train_split)
        train_dataset = train_dataset.map(lambda x: {"text": format_conversation(x)})
        
        val_dataset = Dataset.from_pandas(df_val_split)
        val_dataset = val_dataset.map(lambda x: {"text": format_conversation(x)})

        def preprocess_function(examples):
            # Tokenize the full text
            full_text = examples["text"]
            tokenized = tokenizer(
                full_text,
                padding="max_length", 
                truncation=True, 
                max_length=256,
                return_tensors="pt"
            )
            
            # Flatten tensors if they exist
            input_ids = tokenized["input_ids"].flatten() if torch.is_tensor(tokenized["input_ids"]) else tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"].flatten() if torch.is_tensor(tokenized["attention_mask"]) else tokenized["attention_mask"]
            
            # Find where "Response:" starts to only compute loss on the actual response
            try:
                prompt_part = full_text.split("Response: ")[0] + "Response: "
                prompt_tokens = tokenizer(
                    prompt_part, 
                    add_special_tokens=False, 
                    truncation=True, 
                    max_length=256
                )["input_ids"]
                prompt_length = min(len(prompt_tokens), len(input_ids))
            except:
                # Fallback: compute loss on everything if parsing fails
                prompt_length = 0
            
            # Create labels - only compute loss on response part
            labels = input_ids.copy() if isinstance(input_ids, list) else input_ids.clone()
            if prompt_length > 0:
                for i in range(prompt_length):
                    if i < len(labels):
                        labels[i] = -100
            
            return {
                "input_ids": input_ids.tolist() if torch.is_tensor(input_ids) else input_ids,
                "attention_mask": attention_mask.tolist() if torch.is_tensor(attention_mask) else attention_mask,
                "labels": labels.tolist() if torch.is_tensor(labels) else labels
            }

        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=False)
        tokenized_val_dataset = val_dataset.map(preprocess_function, batched=False)
        
        # MINIMAL CHANGE 2: Apertus needs different model loading to avoid dtype issues
        if is_apertus:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float32,  # Use float32 for Apertus
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                quantization_config=bnb_config,  
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Improved LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Slightly higher rank for better capacity
            lora_alpha=32,  # 2x the rank
            lora_dropout=0.1,  # Slightly higher dropout
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        
        # MINIMAL CHANGE 3: Ensure consistent dtype for Apertus
        if is_apertus:
            model = model.float()  # Convert everything to float32 for Apertus

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # MINIMAL CHANGE 4: Disable mixed precision for Apertus only
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=2,  # Smaller batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Maintain effective batch size of 8
            num_train_epochs=1,  # Fewer epochs to prevent overfitting
            learning_rate=1e-4,  # Conservative learning rate
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,  # Add weight decay for regularization
            fp16=False if is_apertus else True,  # Disable fp16 only for Apertus
            optim="adamw_torch",
            save_steps=200,
            save_total_limit=3,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
            dataloader_pin_memory=False,  # Can help with memory issues
        )

        # Custom trainer to handle early stopping
        class EarlyStoppingTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.patience = 3
                self.best_eval_loss = float('inf')
                self.patience_counter = 0
                
            def evaluate(self, eval_dataset=None, **kwargs):
                eval_result = super().evaluate(eval_dataset, **kwargs)
                current_eval_loss = eval_result['eval_loss']
                
                if current_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = current_eval_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered. Best eval loss: {self.best_eval_loss:.4f}")
                    self.control.should_training_stop = True
                    
                return eval_result

        trainer = EarlyStoppingTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
        )

        # Check for existing checkpoints
        checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt.startswith("checkpoint-")] if os.path.exists(checkpoint_dir) else []

        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])
            print(f"🔄 Resuming training from checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            print("🆕 No checkpoint found. Starting fresh.")
            trainer.train()

        # Save the final model
        model.save_pretrained(self.fine_tuned_dir)
        tokenizer.save_pretrained(self.fine_tuned_dir)
        
        # Load the trained model and configure generation
        self.model = model
        self.tokenizer = tokenizer
        self._configure_generation()
        
        print("✅ LoRA fine-tuning complete. Model saved!")
