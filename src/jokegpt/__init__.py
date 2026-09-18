"""JokeGPT: a character-level GPT pre-trained on prose and fine-tuned on jokes."""

from .config import ModelConfig, TrainConfig
from .generate import generate_text
from .model import GPTLanguageModel
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

__all__ = [
    "CharTokenizer",
    "GPTLanguageModel",
    "ModelConfig",
    "TrainConfig",
    "__version__",
    "generate_text",
]
