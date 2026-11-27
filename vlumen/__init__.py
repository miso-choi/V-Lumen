"""V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models."""

from .codebook import FrozenCodebookQuantizer, load_lavit_codebook
from .config import VLumenConfig
from .fusion import VLumenModule
from .memory_table import MultiHeadMemoryTable
from .model import VLumenVLM
from .moe import MoEAdapter

__all__ = [
    "FrozenCodebookQuantizer",
    "load_lavit_codebook",
    "MoEAdapter",
    "MultiHeadMemoryTable",
    "VLumenConfig",
    "VLumenModule",
    "VLumenVLM",
]
