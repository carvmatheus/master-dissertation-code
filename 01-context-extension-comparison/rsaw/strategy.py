from typing import List

try:
    from context_strategies import ContextStrategy
except ImportError:
    from ..context_strategies import ContextStrategy

from .segmenter import ChunkSegmenter
from .scorer import DartboardScorer
from .assembler import TierAssembler


class RSAWStrategy(ContextStrategy):
    """
    Relevance-Stratified Adaptive Window.

    Pipeline de 4 etapas:
    1. Segmentação em chunks (tiktoken)
    2. Pontuação Dartboard
    3. Estratificação em Tier1/Tier2/Tier3 e compressão por tier
    4. Montagem com budget_tokens e marcadores de omissão
    """

    def __init__(
        self,
        theta_alto: float,
        theta_baixo: float,
        budget_tokens: int,
        chunk_size: int,
        overlap: int,
        tier2_ratio: float,
        top_k: int,
        alpha: float,
        beta: float,
        gamma: float,
        summarizer_model: str,
    ):
        self.theta_alto = theta_alto
        self.theta_baixo = theta_baixo
        self.budget_tokens = budget_tokens
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tier2_ratio = tier2_ratio
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.summarizer_model = summarizer_model

        self._segmenter = ChunkSegmenter(chunk_size=chunk_size, overlap=overlap)
        self._scorer = DartboardScorer(top_k=top_k, alpha=alpha, beta=beta, gamma=gamma)
        self._assembler = TierAssembler(
            theta_alto=theta_alto,
            theta_baixo=theta_baixo,
            budget_tokens=budget_tokens,
            tier2_ratio=tier2_ratio,
            summarizer_model=summarizer_model,
            segmenter=self._segmenter,
        )

    def process(self, text: str, query: str) -> List[str]:
        if not text or not query:
            return [text] if text else []

        chunks = self._segmenter.segment(text)
        if not chunks:
            return []

        scored = self._scorer.score_chunks(chunks, query)
        assembled = self._assembler.assemble(scored)
        return [assembled]
