"""
Simulation Package

Core simulation code for generating synthetic social media responses
using large language models with various personalization techniques.

Main Components:
- agent.py: Agent class representing individual users
- model.py: LLM loading, fine-tuning, and generation
- simulate.py: Main simulation loop logic
- model_utils.py: Helper functions for retrieval and preprocessing
- config_utils.py: Configuration file loading

Entry Point:
- run_simulation.py (in project root)

Usage:
    python run_simulation.py --config configs/llama3.1_base.yaml \\
                             --data_file data/bluesky/personas.pkl \\
                             --n_users 250
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .simulate import run_simulation_random_response
from .agent import Agent
from .model import Model
from .config_utils import load_config

__all__ = [
    "run_simulation_random_response",
    "Agent",
    "Model",
    "load_config",
]
