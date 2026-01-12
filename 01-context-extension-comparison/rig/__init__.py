"""
RIG - Reinforced Information Gain

Módulo de RAG com ranking Dartboard que combina:
- Similaridade semântica (embeddings)
- Similaridade lexical (TF-IDF)
- Importância/popularidade dos documentos

Baseado no paper Dartboard e implementação do notebook main_claude.ipynb.
"""

from .dartboard_processor import DartboardProcessor
from .utils import PORTUGUESE_STOPWORDS, normalize_embeddings

__all__ = [
    "DartboardProcessor",
    "PORTUGUESE_STOPWORDS",
    "normalize_embeddings",
]
