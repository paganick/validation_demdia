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
#from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from .model_utils import *
from .model import Model

class Agent:
    def __init__(self, username: str, dataset: str, data_file: str):
        """Initialize an agent with a username and load all past text examples."""
        self.username = username
        self.dataset = dataset
        self.data_file = data_file
        self.examples = self.load_persona_examples()
        self.create_user_history()

    def load_persona_examples(self):
        """Reads a file and stores all examples of text by the agent."""
        try:
            df = pd.read_pickle(self.data_file)
        except Exception as e:
            print(f"Error loading data from {self.data_file}: {e}")
            return []
        
        self.df_user = df[(df['username'] == self.username) & (df['training'] == 1)]

        self.persona = self.df_user['persona'].iloc[0] if not self.df_user.empty else None

        # Load third-person persona if available (for non-instruction-tuned prompts)
        if not self.df_user.empty and 'persona_third_person' in df.columns:
            self.persona_third_person = self.df_user['persona_third_person'].iloc[0]
        else:
            self.persona_third_person = None

        if self.df_user.empty:
            print(f"No training examples found for @{self.username}.")
            return []
        
        return self.df_user['message'].dropna().tolist()

    def sample_examples(self, num_samples=10, seed=None):
        """Returns a random subsample of the stored examples.

        Args:
            num_samples: Number of examples to sample
            seed: Optional seed for reproducibility. If None, uses current random state.
        """
        if seed is not None:
            # Create a local Random instance to avoid affecting global state
            rng = random.Random(seed)
            return rng.sample(self.examples, min(len(self.examples), num_samples))
        return random.sample(self.examples, min(len(self.examples), num_samples))

    def create_user_history(self):
        user_history = self.df_user['message'].tolist()
        history_dir = f"data/{self.dataset}/user_histories"
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

    def generate_response(
        self,
        llm: Model,
        n_examples,
        retrieve_context_bool,
        with_persona: bool = True,
        instruction_tuned: bool = False,
        conversation_history: list = [],
        n_candidates: int = 20,
        max_total_attempts: int = 5,
        seed: int = None
    ):
        """Generate up to `n_candidates` valid responses, retrying if needed.

        Args:
            llm: The language model to use for generation
            n_examples: Number of style examples to include in prompt
            retrieve_context_bool: Whether to retrieve context from user history
            with_persona: Whether to include persona in prompt
            instruction_tuned: Whether to use instruction-tuned prompt format
            conversation_history: List of previous messages in conversation
            n_candidates: Number of valid responses to generate
            max_total_attempts: Maximum generation attempts
            seed: Random seed for reproducibility. If None, uses current random state.
        """
        if self.dataset == "twitter":
            user_prefix = "@"
            content_type = "tweets"
            platform_name = "Twitter"
        elif self.dataset == "reddit":
            user_prefix = "u/"
            content_type = "Reddit posts"
            platform_name = "Reddit"
        elif self.dataset == "bluesky":
            user_prefix = "@"
            content_type = "Bluesky posts"
            platform_name = "Bluesky"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Use seed for sampling examples if provided
        persona_examples = self.sample_examples(n_examples, seed=seed)
        k_retrieval = 3
        retrieved_context = ""

        if conversation_history:
            last_message = conversation_history[-1]
            if retrieve_context_bool:
                history = self.read_user_history()
                if history:
                    bm25_client = build_bm25(history)
                    retrieved_context = retrieve_context(last_message, bm25_client, history, k=k_retrieval)

        def build_instruction_tuned_prompt(persona_examples, conversation_history, retrieved_context=""):
            prompt = "[Instruction]\n"
            
            if with_persona:
                prompt += f"{self.persona}\n\n"
            else:
                prompt += f"You are {user_prefix}{self.username}.\n\n"

            if persona_examples:
                examples = "\n".join(f"- {ex}" for ex in persona_examples)
                prompt += (
                    f"[Writing Style]\nThese are some {content_type} that represent how "
                    f"you, {user_prefix}{self.username}, write:\n{examples}\n\n"
                )

            if retrieved_context:
                prompt += (
                    f"[User Retrieved Context]\nThis is some useful context retrieved from "
                    f"{user_prefix}{self.username}'s history:\n{retrieved_context}\n\n"
                )

            prompt += (
                f"[Conversation]\nHere is a conversation in which you, as {user_prefix}{self.username}, are participating.\n"
                + "\n".join(conversation_history)+"\n\n"
            )

            prompt += (
                f"[Instruction]\nContinue the conversation naturally by adding {user_prefix}{self.username}'s reply "
                f"of a length which is appropriate for {platform_name}.\n"
            )


            return prompt


        def build_non_instruction_tuned_prompt(persona_examples, conversation_history, retrieved_context=""):
            prompt = (
                f"{platform_name} conversation with "
                f"{user_prefix}{self.username}\n\n"
            )

            if with_persona:
                # Use third-person persona for non-instruction-tuned prompts
                persona_text = self.persona_third_person if self.persona_third_person else self.persona
                prompt += f"About {user_prefix}{self.username}: {persona_text}\n\n"

            if persona_examples:
                prompt += f"Example {content_type[:-1]} by {user_prefix}{self.username}:\n"
                prompt += "\n".join(f"- {ex}" for ex in persona_examples) + "\n\n"

            if retrieved_context:
                prompt += (
                    f"Relevant context from {user_prefix}{self.username}'s history:\n"
                    f"{retrieved_context}\n\n"
                )

            prompt += "Conversation:\n"

            if conversation_history:
                prompt += "\n".join(conversation_history) + "\n"

            # Prime completion
            prompt += f"{user_prefix}{self.username}: "

            return prompt


        if instruction_tuned:
            prompt = build_instruction_tuned_prompt(persona_examples, conversation_history, retrieved_context)
        else:
            prompt = build_non_instruction_tuned_prompt(persona_examples, conversation_history, retrieved_context)

        print(prompt)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = llm.model
        tokenizer = llm.tokenizer

        def is_valid_response(response, original_prompt):
            if not response:
                return False

            response_lower = response.lower()
            response_stripped = response.lstrip()

            # Prompt echo / instruction leakage
            if "continue the conversation naturally by adding" in response_lower:
                return False
            if "this is some useful context retrieved from" in response_lower:
                return False
            if "relevant context from" in response_lower:
                return False
            if "write a reply" in response_lower or "generate a" in response_lower:
                return False

            # Reply/response token prefixes (e.g. [REPLY], [Response], Response:, [My Reply])
            for prefix in ["[reply]", "[response]", "response:", "[my reply]", "[my response]"]:
                if response_lower.startswith(prefix):
                    return False
            if response_stripped.startswith("**Response:**") or response_stripped.startswith("**Reply:**"):
                return False

            # HTML / code artifacts
            if response_stripped.startswith("</h1>"):
                return False
            if response_stripped.startswith("```"):
                return False
            if "```tool_code" in response_lower:
                return False

            # Meta-commentary prefixes
            if response_stripped.startswith("Conversation:"):
                return False
            if response_stripped.startswith("Title:"):
                return False
            if response_lower.startswith("here is my") or response_lower.startswith("here's my"):
                return False
            if response_lower.startswith("i'd be happy to"):
                return False

            # AI self-identification
            if "i'm an ai" in response_lower or "i am an ai" in response_lower:
                return False
            if "as an ai language model" in response_lower:
                return False
            if "as a language model" in response_lower:
                return False

            # Chain-of-thought tags
            if "<think>" in response_lower or "</think>" in response_lower:
                return False

            return True

        valid_responses = []
        attempts = 0

        # Create a generator for reproducible sampling if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        while len(valid_responses) < n_candidates and attempts < max_total_attempts:
            remaining_needed = n_candidates - len(valid_responses)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Build generate kwargs
            generate_kwargs = {
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": 300,
                "do_sample": True,
                "num_return_sequences": remaining_needed,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "min_new_tokens": 1,
                "repetition_penalty": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
            }

            # Add generator for reproducibility if available
            if generator is not None:
                # Increment seed for each attempt to get different but reproducible results
                generator.manual_seed(seed + attempts)

            outputs = model.generate(
                inputs.input_ids,
                **generate_kwargs,
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
