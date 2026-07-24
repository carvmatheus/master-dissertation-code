"""
Validador de QUALIDADE da compressão (reference-free, 100% local).

Operacionaliza a "avaliação epistêmica" da teoria de densificação informacional:
avaliar só a resposta final subestima perdas de grounding, entidades e fidelidade.
Aqui mede-se a qualidade da REPRESENTAÇÃO comprimida em si — o eixo `Q_{j,o}` da
formulação knapsack (PIBPC) — separado da performance downstream.

Três tiers, do mais barato ao mais caro:

  Tier 1 — Lexical/factual (determinístico, regex): content_recall,
           number/date retention, evidence_recall e answer_present.
  Tier 2 — Semântico (embeddings MiniLM, local): semantic_fidelity (cobertura).
  Densidade — info_density = semantic_fidelity / compression_ratio, a tradução
              numérica de "densificação informacional orientada à tarefa".

Nada aqui usa API paga: apenas regex e o mesmo all-MiniLM-L6-v2 do RIG.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Tier 1 — retenção lexical e factual
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\b\w+\b", re.UNICODE)
# Números, datas, percentuais, valores monetários, horas.
_NUMERIC = re.compile(r"\d[\d.,:/\-]*%?")

# Stopwords PT + EN — evita inflar o recall com palavras funcionais.
_STOP = frozenset(
    """
    a o e de da do das dos em no na nos nas um uma uns umas que se por para com
    ao aos à às como mas ou os as sua seu suas seus é são foi era ser ter há
    the a an of to in and or is are was were be been being on at by for with as
    that this these those it its from not can will would should has have had
    """.split()
)


def _content_tokens(text: str) -> List[str]:
    return [t for t in (w.lower() for w in _WORD.findall(text)) if t not in _STOP]


def content_recall(original: str, compressed: str) -> Optional[float]:
    """Fração das palavras de conteúdo (types) do original preservadas."""
    o = set(_content_tokens(original))
    if not o:
        return None
    c = set(_content_tokens(compressed))
    return len(o & c) / len(o)


def number_retention(original: str, compressed: str) -> Optional[float]:
    """Fração de números/datas/valores do original preservados (crítico p/ QA)."""
    originals = _NUMERIC.findall(original)
    if not originals:
        return None  # não aplicável a este caso
    present = set(_NUMERIC.findall(compressed))
    return sum(1 for n in originals if n in present) / len(originals)


def evidence_recall(compressed: str, expected: str) -> Optional[float]:
    """Recall (por palavra de conteúdo) da resposta-gold dentro do comprimido.

    Usa a resposta esperada do benchmark como âncora de evidência: se o
    compressor descartou a evidência que sustenta a resposta, isto cai.
    """
    if not expected:
        return None
    tokens = _content_tokens(expected)
    if not tokens:
        return None
    present = set(w.lower() for w in _WORD.findall(compressed))
    return sum(1 for t in tokens if t in present) / len(tokens)


def answer_present(compressed: str, expected: str) -> Optional[float]:
    """1.0 se a string-gold aparece literalmente no comprimido, senão 0.0."""
    if not expected or not expected.strip():
        return None
    return 1.0 if expected.strip().lower() in compressed.lower() else 0.0


# ---------------------------------------------------------------------------
# Tier 2 — fidelidade semântica (cobertura via embeddings)
# ---------------------------------------------------------------------------

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        name = os.environ.get("VALIDATOR_EMB_MODEL", "all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(name)
    return _MODEL


def _chunk(text: str, size: int = 600) -> List[str]:
    """Janela por caracteres (~150 tokens) — respeita o limite do MiniLM (256)."""
    text = text.strip()
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def semantic_fidelity(
    original: str,
    compressed: str,
    max_chunks: int = 48,
) -> Optional[float]:
    """Cobertura semântica: quão bem o comprimido representa o conteúdo original.

    Para cada trecho do original, mede a maior similaridade de cosseno com
    algum trecho do comprimido; retorna a média. Alto = o significado do
    original foi preservado; baixo = houve perda de informação.
    """
    oc = _chunk(original)
    cc = _chunk(compressed)
    if not oc or not cc:
        return None
    # Amostra uniforme se houver trechos demais (mantém custo previsível).
    if len(oc) > max_chunks:
        step = len(oc) / max_chunks
        oc = [oc[int(i * step)] for i in range(max_chunks)]
    if len(cc) > max_chunks:
        step = len(cc) / max_chunks
        cc = [cc[int(i * step)] for i in range(max_chunks)]

    import numpy as np

    model = _model()
    oe = model.encode(oc, normalize_embeddings=True, show_progress_bar=False)
    ce = model.encode(cc, normalize_embeddings=True, show_progress_bar=False)
    sim = np.asarray(oe) @ np.asarray(ce).T  # cosseno (já normalizado)
    return float(sim.max(axis=1).mean())


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------


@dataclass
class QualityMetrics:
    content_recall: Optional[float]
    number_retention: Optional[float]
    evidence_recall: Optional[float]
    answer_present: Optional[float]
    semantic_fidelity: Optional[float]
    info_density: Optional[float]  # semantic_fidelity / compression_ratio

    def as_details(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key in (
            "content_recall",
            "number_retention",
            "evidence_recall",
            "answer_present",
            "semantic_fidelity",
            "info_density",
        ):
            val = getattr(self, key)
            if val is not None:
                out[f"quality_{key}"] = round(val, 4)
        return out


def measure_quality(
    original: str,
    compressed: str,
    expected: str = "",
    compression_ratio: Optional[float] = None,
    enable_semantic: bool = True,
) -> QualityMetrics:
    """Avalia a qualidade da compressão original -> comprimido.

    Args:
        original: contexto antes da compressão.
        compressed: contexto enviado ao LLM.
        expected: resposta-gold do benchmark (âncora de evidência).
        compression_ratio: ratio de custo (para info_density); se None não calcula.
        enable_semantic: liga o Tier 2 (embeddings). Desligue para runs rápidos.
    """
    fidelity = semantic_fidelity(original, compressed) if enable_semantic else None
    density: Optional[float] = None
    if fidelity is not None and compression_ratio and compression_ratio > 0:
        density = fidelity / compression_ratio

    return QualityMetrics(
        content_recall=content_recall(original, compressed),
        number_retention=number_retention(original, compressed),
        evidence_recall=evidence_recall(compressed, expected),
        answer_present=answer_present(compressed, expected),
        semantic_fidelity=fidelity,
        info_density=density,
    )
