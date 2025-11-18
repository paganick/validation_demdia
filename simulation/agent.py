import pandas as pd
import random
import numpy as np
import os
import json
from scipy.stats import weibull_min
from datasets import Dataset
from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from .model_utils import *
from .model import Model

class Agent:
    """Represents a social media user agent with text generation capabilities.

    An Agent encapsulates a user's persona, writing style, and history from social
    media interactions. It can generate responses that mimic the user's writing
    style using various personalization techniques including few-shot examples,
    retrieval-augmented generation, and fine-tuned models.

    Attributes:
        username: The username of the social media user
        data_file: Path to the pickle file containing user data
        examples: List of example messages written by the user
        df_user: DataFrame filtered to this user's training data
        persona: Text description of the user's persona
        history_file: Path to the JSON file storing user's message history
        personalized_model: Fine-tuned model for this specific user (lazy-loaded)
        user_output_dir: Directory where personalized model adapters are saved
        user_adapter_dir: Directory where personalized model adapters are loaded from
    """

    def __init__(self, username: str, data_file: str):
        """Initialize an agent with a username and load all past text examples.

        Args:
            username: The username of the social media user
            data_file: Path to the pickle file containing user conversation data
        """
        self.username = username
        self.data_file = data_file
        self.examples = self.load_persona_examples()
        self.create_user_history()
        self.personalized_model = None

    def load_persona_examples(self):
        """Load training examples and persona information for this user.

        Reads the data file, filters to this user's training data (training == 1),
        and extracts their writing examples and persona description.

        Returns:
            List of message strings written by this user, or empty list if no
            training data is found.

        Side Effects:
            Sets self.df_user to the filtered DataFrame of training data.
            Sets self.persona to the user's persona description.
        """
        try:
            df = pd.read_pickle(self.data_file)
        except Exception as e:
            print(f"Error loading data from {self.data_file}: {e}")
            return []

        self.df_user = df[(df['username'] == self.username) & (df['training'] == 1)]

        self.persona = self.df_user['persona'].iloc[0] if not self.df_user.empty else None

        if self.df_user.empty:
            print(f"No training examples found for @{self.username}.")
            return []

        return self.df_user['message'].dropna().tolist()

    def sample_examples(self, num_samples=10, deterministic=False):
        """Sample writing style examples from the user's history.

        Returns a subset of the user's example messages to use as few-shot
        demonstrations of their writing style. Can operate in deterministic mode
        for reproducible testing or random mode for diverse sampling.

        Args:
            num_samples: Number of examples to sample (default: 10)
            deterministic: If True, returns sorted examples deterministically.
                If False, samples randomly (default: False)

        Returns:
            List of up to num_samples message strings from the user's history.
            Returns fewer if the user has fewer than num_samples examples.
        """
        if deterministic:
            # Deterministic: always return first N examples (sorted for consistency)
            sorted_examples = sorted(self.examples)
            return sorted_examples[:min(len(sorted_examples), num_samples)]
        else:
            # Random: sample randomly
            return random.sample(self.examples, min(len(self.examples), num_samples))

    def create_user_history(self, history_dir="data/user_histories"):
        """Create and save a JSON file of the user's message history.

        Exports all training messages to a JSON file for use with BM25 retrieval.
        Only creates the file if it doesn't already exist to avoid overwriting.

        Args:
            history_dir: Directory where user history files are stored
                (default: "data/user_histories")

        Side Effects:
            Creates history_dir if it doesn't exist.
            Sets self.history_file to the path of the user's history file.
            Writes a JSON file with the user's message history if not already present.
        """
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
        """Load the user's message history from the JSON file.

        Returns:
            List of message strings from the user's history, or empty list
            if the history file doesn't exist.
        """
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return []

    def train_personalized_model(self, llm: Model):
        """Train a personalized LoRA adapter on the user's writing style.

        Creates a user-specific adapter using Parameter-Efficient Fine-Tuning (PEFT)
        with LoRA. The adapter is trained on the user's messages with BM25 retrieval
        augmentation to capture their writing style and content preferences.

        Args:
            llm: The base language model to personalize. Can be a base model or
                a fine-tuned model.

        Side Effects:
            Trains a LoRA adapter and saves it to self.user_output_dir.
            Saves the tokenizer to the same directory.
            Prints training progress to stdout.
        """
        if llm.fine_tuned_dir:
            base_model_dir = llm.fine_tuned_dir
            tokenizer = AutoTokenizer.from_pretrained(llm.fine_tuned_dir)
        else:
            base_model_dir = llm.model_name
            tokenizer = AutoTokenizer.from_pretrained(llm.model_name)

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
        save_base_dir = "/scratch/nicpag/personal_peft"

        self.user_output_dir = f"{save_base_dir}/{self.username}/{llm.model_name}"
        os.makedirs(self.user_output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.user_output_dir,
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
        
        model_for_training.save_pretrained(self.user_output_dir)
        tokenizer.save_pretrained(self.user_output_dir)
        print(f"Adapter for user @{self.username} on model {base_model_dir} saved at {self.user_output_dir}\n")


    def load_personalized_model(self, llm: Model):
        """Load the user's personalized model with LoRA adapter merged.

        Loads the base model and merges the user-specific LoRA adapter to create
        a fully personalized model. If the adapter doesn't exist, trains it first.
        Uses lazy loading - returns cached model if already loaded.

        Args:
            llm: The base language model that the adapter was trained on

        Returns:
            The personalized model with LoRA adapter merged, ready for inference.

        Side Effects:
            Calls train_personalized_model if adapter doesn't exist.
            Sets self.personalized_model for caching.
            Sets self.user_adapter_dir to the adapter path.
        """
        if self.personalized_model:
            return self.personalized_model

        if llm.fine_tuned_dir:
            base_model_dir = llm.fine_tuned_dir
        else:
            base_model_dir = llm.model_name
        save_base_dir = "/scratch/nicpag/personal_peft"
        self.user_adapter_dir =  f"{save_base_dir}/{self.username}/{llm.model_name}"

        if not(os.path.isdir(self.user_adapter_dir)):
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
                self.user_adapter_dir,
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
        with_persona: bool = True,
        conversation_history: list = [],
        n_candidates: int = 20,
        max_total_attempts: int = 5,
        deterministic: bool = False
    ):
        """Generate responses mimicking the user's writing style.

        Generates multiple candidate responses to continue a conversation in the
        user's style. Supports various personalization methods including few-shot
        examples, BM25 retrieval, persona descriptions, and fine-tuned models.
        Can operate in deterministic or sampling mode.

        Args:
            llm: The language model to use for generation
            n_examples: Number of writing style examples to include in the prompt
            retrieve_context_bool: If True, use BM25 to retrieve relevant messages
                from the user's history
            personalized_bool: If True, use the user's personalized fine-tuned model
                instead of the base model
            with_persona: If True, include the user's persona description in the
                prompt (default: True)
            conversation_history: List of previous messages in the conversation
                to provide context (default: [])
            n_candidates: Number of valid responses to generate (default: 20)
            max_total_attempts: Maximum number of generation attempts to get
                n_candidates valid responses (default: 5)
            deterministic: If True, uses greedy decoding for reproducible outputs.
                If False, uses sampling with temperature for diverse outputs
                (default: False)

        Returns:
            List of up to n_candidates valid response strings. May return fewer
            if max_total_attempts is reached before generating enough valid responses.
            In deterministic mode, returns only 1 response per attempt.

        Notes:
            Deterministic mode uses greedy decoding (do_sample=False) and generates
            only 1 sequence per attempt, ensuring fully reproducible outputs with
            fixed seeds. Sampling mode uses temperature=0.8, top_p=0.9 for diversity.

            A response is considered valid if it doesn't contain instruction text
            and has appropriate content.
        """

        persona_examples = self.sample_examples(n_examples, deterministic=deterministic)
        k_retrieval = 3
        retrieved_context = ""

        if conversation_history:
            last_message = conversation_history[-1]
            if retrieve_context_bool:
                history = self.read_user_history()
                if history:
                    bm25_client = build_bm25(history)
                    retrieved_context = retrieve_context(last_message, bm25_client, history, k=k_retrieval)

        def build_prompt(persona_examples, conversation_history, retrieved_context=""):
            prompt = "[Instruction] "
            if with_persona:
                prompt += f"{self.persona}"
            else:
                prompt += f"You are @{self.username}. "
            prompt+= "Continue the conversation naturally adding a concise (one sentence) reply.\n"
            if persona_examples:
                examples = "\n".join(f"- {ex}" for ex in persona_examples)
                prompt += f"[Writing Style] These are some tweets that represent how @{self.username} writes:\n{examples}\n\n"
            if retrieved_context:
                prompt += f"[User Retrieved Context] This is some useful context retrieved from @{self.username}'s history \n" + retrieved_context + "\n\n"
            if conversation_history:
                prompt += "[Conversation] " + "\n".join(conversation_history) + f"\n{self.username}:"
            return prompt

        prompt = build_prompt(persona_examples, conversation_history, retrieved_context)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = llm.model if not personalized_bool else self.load_personalized_model(llm)
        tokenizer = llm.tokenizer if not personalized_bool else AutoTokenizer.from_pretrained(self.user_adapter_dir)

        def is_valid_response(response, original_prompt):
            if not response:
                return False
            if "continue the conversation naturally adding a concise" in response.lower():
                return False
            if "these are some tweets that represent how" in response.lower():
                return False
                return False
            if "this is some useful context retrieved from" in response.lower():
                return False
            return True

        valid_responses = []
        attempts = 0

        while len(valid_responses) < n_candidates and attempts < max_total_attempts:
            remaining_needed = n_candidates - len(valid_responses)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if deterministic:
                # Deterministic mode: use greedy decoding
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    do_sample=False,  # Greedy decoding
                    num_return_sequences=1,  # Only 1 in deterministic mode
                    min_new_tokens=10,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                # Sampling mode: use temperature and top-p for diversity
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
        #valid_responses_sorted = sorted(valid_responses, key=score_response)
        return valid_responses[:n_candidates]
