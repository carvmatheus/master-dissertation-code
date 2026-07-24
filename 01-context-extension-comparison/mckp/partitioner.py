"""
Particionadores do contexto.

Ao contrário da segmentação por janela com sobreposição usada no baseline RSAW,
as partições aqui são disjuntas por construção. Isto preserva a hipótese central
do MCKP, custos aditivos e independentes entre partições, sem duplicação de
tokens entre segmentos vizinhos.

  StructuralPartitioner  agrupa parágrafos até um teto de tokens, respeitando
                         fronteiras naturais do texto.
  SemanticPartitioner    posiciona fronteiras onde a similaridade entre
                         sentenças adjacentes cai, ao estilo de segmentação
                         semântica, também respeitando um teto de tokens.
"""
from __future__ import annotations

import re
from typing import List

from .models import MCKPConfig, Partition
from .fidelity import count_tokens
from .tokenization import DEFAULT_TOKEN_COUNTER

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Último recurso para garantir o teto mesmo sem fronteira de sentença."""
    words = text.split()
    if not words:
        return []
    out: List[str] = []
    buf: List[str] = []
    for word in words:
        candidate = " ".join([*buf, word])
        if buf and DEFAULT_TOKEN_COUNTER.count(candidate) > max_tokens:
            out.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        out.append(" ".join(buf))
    return out


def _bounded_sentences(text: str, max_tokens: int) -> List[str]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    bounded: List[str] = []
    for sentence in sentences or [text]:
        if count_tokens(sentence) > max_tokens:
            bounded.extend(_split_by_tokens(sentence, max_tokens))
        else:
            bounded.append(sentence)
    return bounded


def _split_blocks(text: str) -> List[str]:
    blocks = [b.strip() for b in _PARA_SPLIT.split(text) if b and b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


class StructuralPartitioner:
    def __init__(self, config: MCKPConfig):
        self.max_tokens = config.max_partition_tokens

    def partition(self, text: str, query: str = "") -> List[Partition]:
        blocks = _split_blocks(text)
        partitions: List[Partition] = []
        buffer: List[str] = []
        buffer_tokens = 0

        def flush():
            nonlocal buffer, buffer_tokens
            if buffer:
                partitions.append(
                    Partition(
                        index=len(partitions),
                        text="\n\n".join(buffer),
                        kind="paragraph_group",
                    )
                )
                buffer = []
                buffer_tokens = 0

        for block in blocks:
            bt = count_tokens(block)
            if bt > self.max_tokens:
                # Bloco grande demais, quebra por sentença mantendo disjunção.
                flush()
                for part_text in self._split_large(block):
                    partitions.append(
                        Partition(index=len(partitions), text=part_text, kind="sentence_group")
                    )
                continue
            if buffer_tokens + bt > self.max_tokens and buffer:
                flush()
            buffer.append(block)
            buffer_tokens += bt
        flush()
        return partitions

    def _split_large(self, block: str) -> List[str]:
        sentences = _bounded_sentences(block, self.max_tokens)
        out: List[str] = []
        buf: List[str] = []
        buf_tokens = 0
        for sent in sentences:
            st = count_tokens(sent)
            if buf_tokens + st > self.max_tokens and buf:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(sent)
            buf_tokens += st
        if buf:
            out.append(" ".join(buf))
        return out or [block]


class SemanticPartitioner:
    def __init__(self, config: MCKPConfig):
        self.max_tokens = config.max_partition_tokens
        self.embedding_model = config.embedding_model
        self._model = None

    def _embedder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def partition(self, text: str, query: str = "") -> List[Partition]:
        sentences = _bounded_sentences(text.replace("\n", " "), self.max_tokens)
        if len(sentences) <= 1:
            return [Partition(index=0, text=text.strip(), kind="sentence_group")] if text.strip() else []

        import numpy as np

        emb = self._embedder().encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )
        emb = np.asarray(emb)
        sims = [float(emb[i] @ emb[i + 1]) for i in range(len(sentences) - 1)]
        # Fronteira onde a similaridade cai abaixo do percentil 25 (queda local).
        threshold = float(np.percentile(sims, 25)) if sims else 0.0

        partitions: List[Partition] = []
        buf: List[str] = [sentences[0]]
        buf_tokens = count_tokens(sentences[0])

        def flush():
            nonlocal buf, buf_tokens
            if buf:
                partitions.append(
                    Partition(index=len(partitions), text=" ".join(buf), kind="semantic_block")
                )
                buf, buf_tokens = [], 0

        for i in range(1, len(sentences)):
            st = count_tokens(sentences[i])
            drop = sims[i - 1] < threshold
            if (drop and buf) or (buf_tokens + st > self.max_tokens and buf):
                flush()
            buf.append(sentences[i])
            buf_tokens += st
        flush()
        return partitions


def build_partitioner(config: MCKPConfig):
    if config.partitioner == "semantic":
        return SemanticPartitioner(config)
    return StructuralPartitioner(config)
