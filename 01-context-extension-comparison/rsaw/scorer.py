import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DartboardScorer:
    """Pontua chunks usando o ranking Dartboard do DartboardProcessor."""

    def __init__(self, top_k: int, alpha: float, beta: float, gamma: float):
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            try:
                from rig import DartboardProcessor
            except ImportError:
                from ..rig import DartboardProcessor
            self._processor = DartboardProcessor(
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
        return self._processor

    def score_chunks(self, chunks: List[str], query: str) -> List[Dict[str, Any]]:
        """
        Retorna lista de dicts {chunk, original_index, score} para todos os chunks.
        Mantém ordem original. Chunks sem score recebem 0.0.
        """
        if not chunks:
            return []
        if not query:
            return [{"chunk": c, "original_index": i, "score": 0.0} for i, c in enumerate(chunks)]

        processor = self._get_processor()
        processor.index_chunks(chunks)

        # Pontua todos os chunks de uma só vez
        n = len(chunks)
        results = processor.dartboard_ranking(query, top_k=n)
        score_map = {r["chunk_id"]: r["score"] for r in results}

        return [
            {"chunk": chunks[i], "original_index": i, "score": score_map.get(i, 0.0)}
            for i in range(n)
        ]
