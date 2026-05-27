import logging
from typing import List

logger = logging.getLogger(__name__)

_encoding_cache = None


def _get_encoding():
    global _encoding_cache
    if _encoding_cache is None:
        import tiktoken
        try:
            _encoding_cache = tiktoken.get_encoding("o200k_base")
        except Exception:
            logger.warning("o200k_base não disponível, usando cl100k_base")
            _encoding_cache = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache


class ChunkSegmenter:
    """Divide texto em chunks de tamanho fixo em tokens com overlap."""

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def segment(self, text: str) -> List[str]:
        enc = _get_encoding()
        token_ids = enc.encode(text)
        if not token_ids:
            return []

        step = max(1, self.chunk_size - self.overlap)
        chunks = []
        i = 0
        while i < len(token_ids):
            chunk_ids = token_ids[i : i + self.chunk_size]
            chunks.append(enc.decode(chunk_ids))
            if i + self.chunk_size >= len(token_ids):
                break
            i += step
        return chunks

    def count_tokens(self, text: str) -> int:
        return len(_get_encoding().encode(text))
