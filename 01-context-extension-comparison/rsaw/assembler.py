import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TierAssembler:
    """
    Estratifica chunks em 3 tiers e monta contexto dentro do budget_tokens.

    Tier 1 (score >= theta_alto): incluído inteiro.
    Tier 2 (theta_baixo <= score < theta_alto): comprimido com PerplexityCompressor.
    Tier 3 (score < theta_baixo): resumido com GroqSemanticCompressor.
    """

    def __init__(
        self,
        theta_alto: float,
        theta_baixo: float,
        budget_tokens: int,
        tier2_ratio: float,
        summarizer_model: str,
        segmenter,
    ):
        self.theta_alto = theta_alto
        self.theta_baixo = theta_baixo
        self.budget_tokens = budget_tokens
        self.tier2_ratio = tier2_ratio
        self.summarizer_model = summarizer_model
        self.segmenter = segmenter
        self._perplexity = None
        self._groq = None

    def _perplexity_compressor(self):
        if self._perplexity is None:
            try:
                from prompt_compression import PerplexityCompressor
            except ImportError:
                from ..prompt_compression import PerplexityCompressor
            self._perplexity = PerplexityCompressor()
        return self._perplexity

    def _groq_compressor(self):
        if self._groq is None:
            try:
                from prompt_compression import GroqSemanticCompressor
            except ImportError:
                from ..prompt_compression import GroqSemanticCompressor
            self._groq = GroqSemanticCompressor(model_name=self.summarizer_model)
        return self._groq

    def _process_tier(self, chunk: str, tier: int) -> Optional[str]:
        if tier == 1:
            return chunk

        if tier == 2:
            try:
                return self._perplexity_compressor().compress(chunk, self.tier2_ratio)
            except Exception as e:
                logger.warning(f"PerplexityCompressor falhou (Tier 2): {e}. Usando chunk original.")
                return chunk

        # tier == 3
        try:
            return self._groq_compressor().compress(chunk, 0.3)
        except Exception as e:
            logger.warning(f"GroqSemanticCompressor falhou (Tier 3): {e}. Omitindo chunk.")
            return None

    def assemble(self, scored_chunks: List[dict]) -> str:
        """
        scored_chunks: lista de {chunk, original_index, score} em ordem original.
        Retorna string com contexto montado, com marcadores de omissão.
        """
        # Classifica e processa cada chunk
        processed = []
        for item in scored_chunks:
            score = item["score"]
            if score >= self.theta_alto:
                tier = 1
            elif score >= self.theta_baixo:
                tier = 2
            else:
                tier = 3
            text = self._process_tier(item["chunk"], tier)
            processed.append({"original_index": item["original_index"], "tier": tier, "text": text})

        # Monta em ordem original com controle de budget
        parts: List[str] = []
        remaining = self.budget_tokens
        omitted = 0
        budget_exhausted = False

        for item in processed:
            text = item["text"]

            if text is None:
                omitted += 1
                continue

            token_count = self.segmenter.count_tokens(text)

            if token_count > remaining:
                if not budget_exhausted:
                    budget_exhausted = True
                    if item["tier"] == 1:
                        logger.warning(
                            f"Budget esgotado ao processar Tier 1 "
                            f"({remaining} tokens restantes). Retornando só Tier 1 acumulado."
                        )
                    else:
                        logger.warning(
                            f"Budget de {self.budget_tokens} tokens esgotado. "
                            "Contexto truncado."
                        )
                omitted += 1
                continue

            if omitted > 0:
                parts.append(f"[...{omitted} chunks omitidos...]")
                omitted = 0

            parts.append(text)
            remaining -= token_count

        if omitted > 0:
            parts.append(f"[...{omitted} chunks omitidos...]")

        return "\n---\n".join(parts)
