"""Consolidated data format for postprocessing results.

This module defines the data structures and I/O utilities for the new
consolidated format that reduces file duplication and organizes results
more efficiently.

Key improvements:
- Shared user/prompt data stored once (not duplicated per config)
- Single JSON per configuration with all selection methods
- Consolidated validation results
- Easy to extend and query
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class UserMetadata:
    """Metadata for a single user.

    Attributes:
        username: User identifier
        persona: GPT-4 generated persona description
        platform: Social media platform (bluesky, twitter, reddit)
        n_training_messages: Number of messages in training set
        metadata: Additional platform-specific metadata
    """
    username: str
    persona: Optional[str] = None
    platform: Optional[str] = None
    n_training_messages: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """Original message/prompt that AI responds to.

    Attributes:
        prompt_id: Unique identifier for this prompt
        user: Username who would respond to this
        text: The message text to respond to
        context: Previous conversation context (if any)
        metadata: Additional metadata (e.g., timestamp, platform-specific)
    """
    prompt_id: str
    user: str
    text: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseEntry:
    """Single response with all selection methods and scores.

    Attributes:
        user: Username this response is for
        prompt_id: ID of prompt being responded to
        candidates: All generated response candidates
        selected_indices: Index in candidates for each selection method
        scores: Scores/probabilities for each candidate and method
        metadata: Additional metadata (generation params, timestamps, etc.)
    """
    user: str
    prompt_id: str
    candidates: List[str]
    selected_indices: Dict[str, int] = field(default_factory=dict)  # {method: idx}
    scores: Dict[str, List[float]] = field(default_factory=dict)    # {method: [scores]}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_selected_text(self, method: str) -> Optional[str]:
        """Get selected response text for a given method.

        Args:
            method: Selection method (random, ml, cosine)

        Returns:
            Selected response text, or None if method not found
        """
        idx = self.selected_indices.get(method)
        if idx is not None and 0 <= idx < len(self.candidates):
            return self.candidates[idx]
        return None


@dataclass
class ConsolidatedResponses:
    """All responses for a single configuration.

    Attributes:
        config_name: Configuration identifier
        model: Model name/path
        metadata: Configuration metadata (finetuned, context, style, etc.)
        responses: List of response entries
    """
    config_name: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    responses: List[ResponseEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config_name': self.config_name,
            'model': self.model,
            'metadata': self.metadata,
            'responses': [asdict(r) for r in self.responses]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsolidatedResponses':
        """Create from dictionary loaded from JSON."""
        responses = [ResponseEntry(**r) for r in data.get('responses', [])]
        return cls(
            config_name=data['config_name'],
            model=data['model'],
            metadata=data.get('metadata', {}),
            responses=responses
        )

    def save(self, filepath: str):
        """Save to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'ConsolidatedResponses':
        """Load from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ConsolidatedResponses instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ValidationResults:
    """Validation results for all configurations and methods.

    Attributes:
        dataset: Dataset name (bluesky, twitter, reddit)
        results: Nested dict: {config_name: {selection_method: {validation_type: results}}}
    """
    dataset: str
    results: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def add_result(self, config_name: str, selection_method: str,
                   validation_type: str, result: Dict[str, Any]):
        """Add validation result.

        Args:
            config_name: Configuration identifier
            selection_method: Selection method (random, ml, cosine)
            validation_type: Validation type (bert, features, empath)
            result: Result dictionary
        """
        if config_name not in self.results:
            self.results[config_name] = {}
        if selection_method not in self.results[config_name]:
            self.results[config_name][selection_method] = {}
        self.results[config_name][selection_method][validation_type] = result

    def get_result(self, config_name: str, selection_method: str,
                   validation_type: str) -> Optional[Dict[str, Any]]:
        """Get validation result.

        Args:
            config_name: Configuration identifier
            selection_method: Selection method
            validation_type: Validation type

        Returns:
            Result dictionary, or None if not found
        """
        return (self.results
                .get(config_name, {})
                .get(selection_method, {})
                .get(validation_type))

    def save(self, filepath: str):
        """Save to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': self.dataset,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'ValidationResults':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            dataset=data['dataset'],
            results=data.get('results', {})
        )


class SharedDataManager:
    """Manages shared user and prompt data.

    This class handles reading/writing the shared data files that are
    stored once per dataset instead of being duplicated per configuration.
    """

    def __init__(self, base_dir: str, dataset: str):
        """Initialize shared data manager.

        Args:
            base_dir: Base results directory
            dataset: Dataset name (bluesky, twitter, reddit)
        """
        self.base_dir = Path(base_dir)
        self.dataset = dataset
        self.shared_dir = self.base_dir / dataset / 'shared'
        self.shared_dir.mkdir(parents=True, exist_ok=True)

    def save_users(self, users: List[UserMetadata]):
        """Save user metadata.

        Args:
            users: List of UserMetadata objects
        """
        filepath = self.shared_dir / 'users.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(u) for u in users], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(users)} users to {filepath}")

    def load_users(self) -> List[UserMetadata]:
        """Load user metadata.

        Returns:
            List of UserMetadata objects
        """
        filepath = self.shared_dir / 'users.json'
        if not filepath.exists():
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [UserMetadata(**u) for u in data]

    def save_prompts(self, prompts: List[Prompt]):
        """Save prompts.

        Args:
            prompts: List of Prompt objects
        """
        filepath = self.shared_dir / 'prompts.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(p) for p in prompts], f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompts)} prompts to {filepath}")

    def load_prompts(self) -> List[Prompt]:
        """Load prompts.

        Returns:
            List of Prompt objects
        """
        filepath = self.shared_dir / 'prompts.json'
        if not filepath.exists():
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Prompt(**p) for p in data]

    def get_prompt_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get prompt by ID.

        Args:
            prompt_id: Prompt identifier

        Returns:
            Prompt object or None if not found
        """
        prompts = self.load_prompts()
        for p in prompts:
            if p.prompt_id == prompt_id:
                return p
        return None

    def get_user(self, username: str) -> Optional[UserMetadata]:
        """Get user metadata.

        Args:
            username: Username to look up

        Returns:
            UserMetadata object or None if not found
        """
        users = self.load_users()
        for u in users:
            if u.username == username:
                return u
        return None


if __name__ == "__main__":
    # Test the data structures
    print("Testing consolidated data format...")

    # Test response entry
    resp = ResponseEntry(
        user="user123",
        prompt_id="msg456",
        candidates=["Response A", "Response B", "Response C"],
        selected_indices={"random": 0, "ml": 2, "cosine": 1},
        scores={
            "ml": [0.3, 0.5, 0.9],
            "cosine": [0.6, 0.8, 0.4]
        }
    )

    print(f"Random selected: {resp.get_selected_text('random')}")
    print(f"ML selected: {resp.get_selected_text('ml')}")
    print(f"Cosine selected: {resp.get_selected_text('cosine')}")

    # Test consolidated responses
    consolidated = ConsolidatedResponses(
        config_name="llama-3.1-8B_base_ctx0_style0",
        model="meta-llama/Llama-3.1-8B",
        metadata={"finetuned": False, "context": 0, "style": 0},
        responses=[resp]
    )

    # Test save/load
    test_path = "/tmp/test_consolidated.json"
    consolidated.save(test_path)
    loaded = ConsolidatedResponses.load(test_path)
    print(f"\n✅ Saved and loaded {len(loaded.responses)} responses")

    # Clean up
    os.remove(test_path)

    print("\n✅ All tests passed!")
