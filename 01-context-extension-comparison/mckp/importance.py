"""
Função de importância I(t_j; q).

Combina três sinais por partição, conforme a formalização do Capítulo 3.

  R(t_j, q)  relevância à consulta, sinais semântico e lexical reaproveitados
             do ranking Dartboard (embeddings MiniLM e TF-IDF).
  D(t_j)     densidade factual, proporção de números, datas e entidades.
  P(t_j)     saliência posicional, protege regiões de alto valor do viés
             lost in the middle.

I(t_j; q) = w_r * R + w_d * D + w_p * P, com cada componente normalizado por
min-max entre as partições. Os pesos vêm da configuração.
"""
from __future__ import annotations

import re
from typing import List

from .models import MCKPConfig, Partition

_NUMERIC = re.compile(r"\d[\d.,:/\-]*%?")
_PROPER = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b")
_WORD = re.compile(r"\b\w+\b", re.UNICODE)


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _positional(n: int, shape: str) -> List[float]:
    if n <= 1:
        return [1.0] * n
    out = []
    for i in range(n):
        pos = i / (n - 1)  # 0 no início, 1 no fim
        if shape == "central":
            out.append(1.0 - abs(2.0 * pos - 1.0))   # alto no meio
        elif shape == "u_shaped":
            out.append(abs(2.0 * pos - 1.0))          # alto nas bordas
        else:  # flat
            out.append(1.0)
    return out


class ImportanceScorer:
    def __init__(self, config: MCKPConfig):
        self.config = config
        self._dartboard = None

    def _relevance(self, partitions: List[Partition], query: str) -> List[float]:
        """R(t_j, q) via sinais semântico e lexical do Dartboard."""
        texts = [p.text for p in partitions]
        if not query or not any(texts):
            return [0.0] * len(partitions)
        try:
            from rig.dartboard_processor import DartboardProcessor
        except ImportError:
            from ..rig.dartboard_processor import DartboardProcessor

        if self._dartboard is None:
            self._dartboard = DartboardProcessor(
                embedding_model=self.config.embedding_model
            )
        db = self._dartboard
        db.index_chunks(texts)
        ranked = db.dartboard_ranking(query, top_k=len(texts))

        rel = [0.0] * len(partitions)
        for item in ranked:
            j = item["chunk_id"]
            if 0 <= j < len(rel):
                rel[j] = 0.7 * item["semantic_score"] + 0.3 * item["lexical_score"]
        return rel

    def _density(self, partitions: List[Partition]) -> List[float]:
        dens = []
        for p in partitions:
            words = _WORD.findall(p.text)
            n = max(1, len(words))
            factual = len(_NUMERIC.findall(p.text)) + len(_PROPER.findall(p.text))
            dens.append(factual / n)
        return dens

    def score(self, partitions: List[Partition], query: str) -> List[Partition]:
        """Preenche partition.importance para cada partição e as retorna."""
        w = self.config.weights
        r = _minmax(self._relevance(partitions, query))
        d = _minmax(self._density(partitions))
        p = _minmax(_positional(len(partitions), self.config.positional_shape))

        for j, part in enumerate(partitions):
            part.importance_components = {
                "relevance": r[j],
                "factual_density": d[j],
                "positional_salience": p[j],
            }
            part.importance = (
                w.get("w_r", 0.6) * r[j]
                + w.get("w_d", 0.2) * d[j]
                + w.get("w_p", 0.2) * p[j]
            )
        return partitions
