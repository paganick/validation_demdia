import pandas as pd
import random
import numpy as np
import os
import json
from datasets import Dataset
from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from .model_utils import *
from .model import Model

class Agent:
    def __init__(self, username: str, data_file: str):
        """Initialize an agent with a username and load all past text examples."""
        self.username = username
        self.data_file = data_file
        self.examples = self.load_persona_examples()
        self.create_user_history()
        self.personalized_model = None

    def load_persona_examples(self):
        """Reads a file and stores all examples of text by the agent."""
        try:
            df = pd.read_pickle(self.data_file)
        except Exception as e:
            print(f"Error loading data from {self.data_file}: {e}")
            return []
        
        self.df_user = df[(df['username'] == self.username) & (df['training'] == 1)]
        
        if self.df_user.empty:
            print(f"No training examples found for @{self.username}.")
            return []
        
        return self.df_user['message'].dropna().tolist()

    def sample_examples(self, num_samples=10):
        """Returns a random subsample of the stored examples."""
        return random.sample(self.examples, min(len(self.examples), num_samples))

    def create_user_history(self, history_dir="data/user_histories"):
        user_history = self.df_user['message'].tolist()
        os.makedirs(history_dir, exist_ok=True)
        self.history_file = os.path.join(history_dir, f"user_history_{self.username}.json")
        if not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                json.dump(user_history, f, indent=4)
                print(f"User history for {self.username} saved to {self.history_file}")
        else:
            print(f"User history file already exists at {self.history_file}, skipping write.")

    def read_user_history(self):  
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return []

    def train_personalized_model(self, llm: Model): 
        if llm.fine_tuned_dir:
            base_model_dir = llm.fine_tuned_dir
        else:
            base_model_dir = llm.model_name

        print(f"--- Training adapter for user: {self.username} on model: {base_model_dir} ---")
        self.df_user["text"] = self.df_user.apply(format_conversation, axis=1)

        k_retrieval = 3 # Could be a parameter
        
        # Retrieve BM25 history.
        user_history = self.read_user_history()
        if user_history:
            bm25_client = build_bm25(user_history)
        else:
            bm25_client = None

        # Augment each prompt with retrieved history (if available).
        def augment_prompt(text):
            if bm25_client and len(user_history) > 0:
                retrieved = retrieve_context(text, bm25_client, user_history, k_retrieval)
                return retrieved + "\n" + text
            else:
                return text
        
        self.df_user["text"] = self.df_user["text"].apply(augment_prompt)

        ds_user = Dataset.from_pandas(self.df_user)
        preprocess = partial(preprocess_function, llm.tokenizer)
        ds_user = ds_user.map(preprocess, batched=True) 

        tokenizer = AutoTokenizer.from_pretrained(llm.model_name)

        # -------------------------------
        # Load and Prepare the Base Model for Personalization
        # -------------------------------
        torch.cuda.empty_cache()
        # Load model on CPU to avoid offloaded modules.
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        base_model = base_model.to("cuda")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        model_for_training = get_peft_model(base_model, peft_config)
        model_for_training.gradient_checkpointing_enable()
        model_for_training.enable_input_require_grads()
        
        # Specify directories for saving personalized adapters.
        save_base_dir = "./personal_peft"

        user_output_dir = f"{save_base_dir}/{self.username}/{base_model_dir}"
        os.makedirs(user_output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=user_output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=2,  # Change to desired number of epochs.
            bf16=True,
            optim="adamw_torch",
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            #evaluation_strategy="no",
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model_for_training,
            args=training_args,
            train_dataset=ds_user,
        )
        
        trainer.train()
        
        model_for_training.save_pretrained(user_output_dir)
        tokenizer.save_pretrained(user_output_dir)
        print(f"Adapter for user @{self.username} on model {base_model_dir} saved at {user_output_dir}\n")


    def load_personalized_model(self, llm: Model):
        if self.personalized_model:
            return self.personalized_model
        
        if llm.fine_tuned_dir:
            base_model_dir = llm.fine_tuned_dir
        else:
            base_model_dir = llm.model_name
        user_adapter_dir = os.path.join("personal_peft", self.username, base_model_dir)

        if not(os.path.isdir(user_adapter_dir)):
            self.train_personalized_model(llm)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            ignore_mismatched_sizes=True
        )
        base_model.eval()     
        # Load and merge the personalized adapter.
        try:
            adapter_model = PeftModel.from_pretrained(
                base_model, 
                user_adapter_dir, 
                ignore_mismatched_sizes=True
            )
            self.personalized_model = adapter_model.merge_and_unload()
        except Exception as e:
            print(f"Error loading adapter for {self.username}: {e}")

        self.personalized_model.eval()
        return self.personalized_model

    def generate_response(
        self,
        llm: Model,
        n_examples,
        retrieve_context_bool,
        personalized_bool,
        conversation_history: list = [],
        n_candidates: int = 20,
        max_total_attempts: int = 5
    ):
        """Generate up to `n_candidates` valid responses, retrying if needed."""
        
        persona_examples = self.sample_examples(n_examples)
        k_retrieval = 3
        retrieved_context = ""

        if conversation_history:
            last_message = conversation_history[-1]
            if retrieve_context_bool:
                history = self.read_user_history()
                if history:
                    bm25_client = build_bm25(history)
                    retrieved_context = retrieve_context(last_message, bm25_client, history, k=k_retrieval)

        def build_prompt(username, persona_examples, conversation_history, retrieved_context=""):
            prompt = f"[Instruction] You are @{username}. Continue the conversation naturally adding a concise (one sentence) tweet reply.\n"
            if persona_examples:
                examples = "\n".join(f"- {ex}" for ex in persona_examples)
                prompt += f"[Writing Style] These are some tweets that represent how @{username} writes:\n{examples}\n\n"
            if retrieved_context:
                prompt += f"[User Retrieved Context] This is some useful context retrieved from @{username}'s history \n" + retrieved_context + "\n\n"
            if conversation_history:
                prompt += "[Conversation] " + "\n".join(conversation_history) + f"\n{self.username}:"
            return prompt

        prompt = build_prompt(self.username, persona_examples, conversation_history, retrieved_context)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = llm.model.to(device) if not personalized_bool else self.load_personalized_model(llm)
        tokenizer = llm.tokenizer

        def is_valid_response(response, original_prompt):
            if not response:
                return False
            if "These are some tweets that represent how" in response and len(response) > 100:
                return False
            return True

        def score_response(resp):
            target_length = np.random.poisson(lam=14)
            length_diff = abs(len(resp.split()) - target_length)
            return length_diff if is_valid_response(resp, prompt) else float("inf")

        valid_responses = []
        attempts = 0

        while len(valid_responses) < n_candidates and attempts < max_total_attempts:
            remaining_needed = n_candidates - len(valid_responses)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=True,
                num_return_sequences=remaining_needed,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                min_new_tokens=10,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                response = response[len(prompt):].strip().split("\n")[0]
                if is_valid_response(response, prompt):
                    valid_responses.append(response)

            attempts += 1

        # Sort and return top n_candidates valid responses
        valid_responses_sorted = sorted(valid_responses, key=score_response)
        return valid_responses_sorted[:n_candidates]
