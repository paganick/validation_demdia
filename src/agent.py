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
    def __init__(self, username: str, dataset: str, posts_file: str, personas_file: str = None):
        """Initialize an agent with a username and load all past text examples.

        Args:
            username: The user's username
            dataset: Dataset name ('twitter', 'reddit', 'bluesky')
            posts_file: Path to posts pickle file (contains username, message, reply_to, training)
            personas_file: Path to personas pickle file (contains username, persona).
                          If None, assumes posts_file contains persona column (old format).
        """
        self.username = username
        self.dataset = dataset
        self.posts_file = posts_file
        self.personas_file = personas_file
        self.examples = self.load_persona_examples()
        self.create_user_history()

    def load_persona_examples(self):
        """Reads posts and persona files and stores all examples of text by the agent.

        Supports two data formats:
        1. New format: Separate posts_file and personas_file
           - posts_file: username, message, reply_to, training
           - personas_file: username, persona (one row per user, already in third person)
        2. Old format: Single file with all columns (personas_file is None)
           - Contains: username, persona, persona_third_person, message, reply_to, training
        """
        # Load posts data
        try:
            df_posts = pd.read_pickle(self.posts_file)
        except Exception as e:
            print(f"Error loading posts data from {self.posts_file}: {e}")
            return []

        # Filter to user's training posts
        self.df_user = df_posts[(df_posts['username'] == self.username) & (df_posts['training'] == 1)]

        if self.df_user.empty:
            print(f"No training examples found for @{self.username}.")
            self.persona = None
            self.persona_third_person = None
            return []

        # Load persona data
        if self.personas_file is not None:
            # New format: separate personas file (Llama-generated)
            # After transformation, this file has the same format as old personas.pkl:
            # - 'persona': second person (for instruction-tuned prompts)
            # - 'persona_third_person': third person (for non-instruction-tuned prompts)
            try:
                df_personas = pd.read_pickle(self.personas_file)
                user_persona = df_personas[df_personas['username'] == self.username]
                if not user_persona.empty:
                    # Load second person persona (for instruction-tuned prompts)
                    self.persona = user_persona['persona'].iloc[0]

                    # Load third person persona if available (for non-instruction-tuned prompts)
                    if 'persona_third_person' in df_personas.columns:
                        self.persona_third_person = user_persona['persona_third_person'].iloc[0]
                    else:
                        # Fallback: use persona as-is (before transformation was run)
                        self.persona_third_person = self.persona
                else:
                    print(f"No persona found for @{self.username} in {self.personas_file}.")
                    self.persona = None
                    self.persona_third_person = None
            except Exception as e:
                print(f"Error loading personas data from {self.personas_file}: {e}")
                self.persona = None
                self.persona_third_person = None
        else:
            # Old format: persona in the same file as posts
            self.persona = self.df_user['persona'].iloc[0] if 'persona' in df_posts.columns else None

            # Load third-person persona if available (for non-instruction-tuned prompts)
            if 'persona_third_person' in df_posts.columns:
                self.persona_third_person = self.df_user['persona_third_person'].iloc[0]
            else:
                self.persona_third_person = None

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
        rejected_responses = []  # all generated texts that failed is_valid_response()
        attempts = 0

        # Diagnostics showed that instruction-tuned models (Qwen, Gemma) very
        # consistently prepend a format header line before the actual response:
        #   Case 1 — standalone header:  "[Response]\n<actual text>"
        #   Case 2 — inline header:      "[Reply] @user <actual text>"
        # Both cases are fixable by stripping the tag. Applied only when
        # instruction_tuned=True to preserve reproducibility for all other configs.
        _strip_headers = instruction_tuned
        _header_tags = frozenset(
            ["[response]", "[reply]", "response:", "[my reply]", "[my response]",
             "**response:**", "**reply:**"]
        )

        print(f"🔬 [DEBUG] Prompt length (chars): {len(prompt)}", flush=True)
        print(f"🔬 [DEBUG] Prompt preview (first 200 chars): {prompt[:200]!r}", flush=True)

        while len(valid_responses) < n_candidates and attempts < max_total_attempts:
            remaining_needed = n_candidates - len(valid_responses)
            print(f"🔬 [DEBUG] Attempt {attempts+1}/{max_total_attempts}: tokenizing prompt...", flush=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[-1]
            print(f"🔬 [DEBUG] Tokenized: {input_len} tokens, device={inputs.input_ids.device}", flush=True)

            # Set seed for reproducibility before each generation attempt
            if seed is not None:
                attempt_seed = seed + attempts
                torch.manual_seed(attempt_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(attempt_seed)

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

            print(f"🔬 [DEBUG] Calling model.generate(): num_return_sequences={remaining_needed}, max_new_tokens=300", flush=True)
            outputs = model.generate(
                inputs.input_ids,
                **generate_kwargs,
            )
            print(f"🔬 [DEBUG] model.generate() returned: {len(outputs)} sequences, each length {outputs.shape[-1]}", flush=True)

            n_valid_before = len(valid_responses)
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                raw_after_prompt = response[len(prompt):].strip()
                if i == 0:  # Only print full raw for first sequence to avoid spam
                    print(f"🔬 [DEBUG]   seq 0 RAW (first 300 chars): {raw_after_prompt[:300]!r}", flush=True)
                if _strip_headers:
                    lines = raw_after_prompt.split("\n")
                    # Case 1: first non-empty line is a standalone tag → skip to next line
                    for line_idx, line in enumerate(lines):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.lower() in _header_tags:
                            # standalone tag — use the rest
                            raw_after_prompt = "\n".join(lines[line_idx + 1:])
                        else:
                            # Case 2: inline tag prefix like "[Reply] @user ..."
                            for tag in _header_tags:
                                if stripped.lower().startswith(tag + " ") or stripped.lower().startswith(tag + "\t"):
                                    raw_after_prompt = stripped[len(tag):].strip()
                                    break
                            else:
                                raw_after_prompt = "\n".join(lines[line_idx:])
                        break
                original_raw = raw_after_prompt  # full output before split, for rejected logging
                response = raw_after_prompt.split("\n")[0]
                valid = is_valid_response(response, prompt)
                print(f"🔬 [DEBUG]   seq {i}: valid={valid}, len={len(response)}, preview={response[:80]!r}", flush=True)
                if valid:
                    valid_responses.append(response)
                else:
                    rejected_responses.append({
                        "attempt": attempts,
                        "seq_idx": i,
                        "raw": original_raw,
                        "first_line": response,
                    })
            print(f"🔬 [DEBUG] Attempt {attempts+1} done: {len(valid_responses)-n_valid_before} new valid (total {len(valid_responses)}/{n_candidates})", flush=True)

            attempts += 1
            

        # Sort and return top n_candidates valid responses
        #valid_responses_sorted = sorted(valid_responses, key=score_response)
        return valid_responses[:n_candidates], rejected_responses
