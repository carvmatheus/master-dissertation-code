"""
Registry de compressores plugáveis.

Cada compressor materializa uma opção de compressão de uma partição, ou seja,
um item de uma classe do MCKP. A interface é única, compress(text, param, query),
onde param é a taxa de retenção alvo. O conjunto de compressores ativo é definido
por configuração, com a seleção final deferida aos resultados dos benchmarks.

Compressores sempre presentes.
  identity   preserva a partição inteira, custo alto, fidelidade 1.
  omission   descarta a partição, custo nulo, fidelidade nula.

Compressores opcionais, com import preguiçoso e dependências opcionais.
  perplexity          remove tokens previsíveis (embrulha PerplexityCompressor).
  semantic            reescreve via LLM local (embrulha OllamaSemanticCompressor).
  sentence_extractive seleciona sentenças relevantes à consulta (MiniLM).
  llmlingua2          compressão por classificação de tokens (pacote llmlingua).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Embedder compartilhado (MiniLM), reutilizado pelo sentence_extractive.
# ---------------------------------------------------------------------------

_EMBEDDERS: Dict[str, object] = {}


def _embedder(name: str):
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer

        _EMBEDDERS[name] = SentenceTransformer(name)
    return _EMBEDDERS[name]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# Interface e adaptadores
# ---------------------------------------------------------------------------


class Compressor:
    name = "base"
    query_dependent = False

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        raise NotImplementedError


class IdentityCompressor(Compressor):
    name = "identity"

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        return text


class OmissionCompressor(Compressor):
    name = "omission"

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        return ""


class PerplexityAdapter(Compressor):
    name = "perplexity"

    def __init__(self, model_name: str = "gpt2"):
        try:
            from prompt_compression import PerplexityCompressor
        except ImportError:
            from ..prompt_compression import PerplexityCompressor
        self._impl = PerplexityCompressor(model_name=model_name)

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        return self._impl.compress(text, param)


class SemanticAdapter(Compressor):
    name = "semantic"

    def __init__(self, model_name: str):
        try:
            from prompt_compression import OllamaSemanticCompressor
        except ImportError:
            from ..prompt_compression import OllamaSemanticCompressor
        self._impl = OllamaSemanticCompressor(model_name=model_name)

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        return self._impl.compress(text, param)


class SentenceExtractiveCompressor(Compressor):
    """Seleção extrativa de sentenças orientada à consulta.

    Ordena as sentenças da partição pela similaridade de cosseno com a
    consulta e mantém as de maior relevância até atingir a fração alvo de
    palavras, preservando a ordem original. Sem consulta, mantém as primeiras
    sentenças. Aproxima abordagens extrativas orientadas à tarefa descritas na
    literatura de compressão de contexto.
    """

    name = "sentence_extractive"
    query_dependent = True

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._model_name = embedding_model

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return text
        total_words = sum(len(s.split()) for s in sentences)
        target_words = max(1, int(total_words * param))

        if query:
            import numpy as np

            model = _embedder(self._model_name)
            emb = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
            qemb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores = np.asarray(emb) @ np.asarray(qemb).T
            order = list(np.argsort(-scores.ravel()))
        else:
            order = list(range(len(sentences)))

        kept: set = set()
        words = 0
        for idx in order:
            if words >= target_words:
                break
            kept.add(int(idx))
            words += len(sentences[idx].split())
        if not kept:
            kept.add(int(order[0]))

        return " ".join(sentences[i] for i in range(len(sentences)) if i in kept)


class LLMLingua2Adapter(Compressor):
    """Compressão por classificação de tokens via pacote llmlingua.

    Import preguiçoso. O pacote llmlingua não é obrigatório para executar o
    módulo, se estiver ausente, este compressor simplesmente não é
    disponibilizado pelo registry.
    """

    name = "llmlingua2"

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "cpu",
    ):
        from llmlingua import PromptCompressor  # ImportError propaga ao registry

        self._impl = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )

    def compress(self, text: str, param: float, query: Optional[str] = None) -> str:
        out = self._impl.compress_prompt(text, rate=float(param))
        if isinstance(out, dict):
            return out.get("compressed_prompt", text)
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CompressorUnavailableError(RuntimeError):
    """Um compressor configurado não pôde ser materializado."""


def compressor_uses_query(name: str) -> bool:
    """Indica se a consulta deve fazer parte da chave de cache."""
    return name == SentenceExtractiveCompressor.name


def build_compressor(name: str, config) -> Compressor:
    """Instancia um compressor pelo nome.

    Falhas de dependência são propagadas com contexto para que o gerador registre
    a degradação no diagnóstico da execução.
    """
    try:
        if name == "identity":
            return IdentityCompressor()
        if name == "omission":
            return OmissionCompressor()
        if name == "perplexity":
            return PerplexityAdapter()
        if name == "semantic":
            return SemanticAdapter(model_name=config.summarizer_model)
        if name == "sentence_extractive":
            return SentenceExtractiveCompressor(embedding_model=config.embedding_model)
        if name == "llmlingua2":
            return LLMLingua2Adapter(model_name=config.llmlingua2_model)
    except Exception as exc:
        raise CompressorUnavailableError(
            f"compressor '{name}' indisponível: {type(exc).__name__}: {exc}"
        ) from exc
    raise ValueError(f"compressor desconhecido: {name}")
