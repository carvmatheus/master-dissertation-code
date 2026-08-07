"""
Estruturas de dados do framework MCKP.

Traduz os conceitos da formulação da mochila de múltipla escolha (Cap. 3 da
dissertação) em objetos concretos:

  Partição semântica (t_j)      -> Partition        (classe do MCKP)
  Representação comprimida       -> CompressionOption (item da classe)
  Custo em tokens (c_{j,o})      -> CompressionOption.token_cost
  Fidelidade (f_{j,o})           -> CompressionOption.fidelity
  Importância I(t_j;q)           -> Partition.importance
  Valor Q_{j,o}=I(t_j;q)f_{j,o}  -> CompressionOption.value
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Partition:
    """Uma partição disjunta do contexto (classe do MCKP)."""

    index: int          # posição na ordem original (0..n-1)
    text: str
    kind: str = "text"  # "document" | "paragraph" | "sentence_group" | ...
    importance: float = 0.0  # I(t_j;q), preenchida pelo ImportanceScorer
    importance_components: Dict[str, float] = field(default_factory=dict)
    required: bool = False  # partições obrigatórias não recebem opção de omissão


@dataclass
class CompressionOption:
    """Uma opção de compressão de uma partição (item de uma classe do MCKP)."""

    partition_index: int
    compressor: str          # nome do compressor que gerou esta opção
    param: float             # parâmetro (ex. taxa de retenção)
    text: str                # representação materializada enviada ao LLM
    token_cost: int          # c_{j,o}
    fidelity: float          # f_{j,o} em [0, 1]
    importance: float        # I(t_j;q) da partição (repetida por conveniência)

    @property
    def value(self) -> float:
        """Q_{j,o} = I(t_j;q) * f_{j,o}."""
        return self.importance * self.fidelity


@dataclass
class Solution:
    """Resultado da resolução do MCKP."""

    chosen: List[CompressionOption]  # uma opção por partição, em ordem
    total_value: float
    total_cost: int


@dataclass
class MCKPConfig:
    """Configuração do framework, carregada de mckp/config.json."""

    partitioner: str = "structural"          # structural | semantic | whole_context
    max_partition_tokens: int = 400
    weights: Dict[str, float] = field(
        default_factory=lambda: {"w_r": 0.6, "w_d": 0.2, "w_p": 0.2}
    )
    positional_shape: str = "central"        # "central" | "u_shaped" | "flat"
    mu: float = 0.0                          # penalização de transição
    distance: str = "compressor_family"      # "compressor_family" | "param_diff" | "none"
    selection_mode: str = "mckp"             # "mckp" | "uniform_control"
    option_set: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"compressor": "identity", "param": 1.0},
            {"compressor": "cpc_minilm", "param": 0.5},
            {"compressor": "cpc_minilm", "param": 0.3},
            {"compressor": "omission", "param": 0.0},
        ]
    )
    number_retention_threshold: float = 0.9
    # Orçamento. Se model_context_tokens vier definido, o orçamento efetivo de
    # contexto é calculado por reserva (ver strategy.py); senão usa budget_tokens.
    budget_tokens: int = 4000
    model_context_tokens: Optional[int] = None
    output_tokens: int = 500
    instruction_tokens: int = 64              # compatibilidade com configs antigas
    token_safety_margin: int = 256             # margem entre tiktoken e tokenizer alvo
    budget_bucket: int = 1                    # 1 mantém o MCKP exato
    embedding_model: str = "all-MiniLM-L6-v2"
    summarizer_model: str = "llama3.1:8b"     # usado pelo compressor semântico
    semantic_fidelity: bool = True            # liga o Tier 2 dos validadores
    llmlingua2_model: str = (
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    )
    selective_context_model: str = "gpt2"
    selective_context_lang: str = "en"
    adaptive_budget_rate: bool = True
    adaptive_rate_safety: float = 0.9
    min_adaptive_rate: float = 0.05
    required_compressors: List[str] = field(default_factory=list)
    audit_log_path: Optional[str] = None
    audit_include_text: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Valida também configurações alteradas após a carga do JSON."""
        if self.max_partition_tokens <= 0:
            raise ValueError("max_partition_tokens deve ser positivo")
        if self.partitioner not in {"structural", "semantic", "whole_context"}:
            raise ValueError(f"partitioner inválido: {self.partitioner}")
        if self.budget_tokens < 0:
            raise ValueError("budget_tokens não pode ser negativo")
        if self.model_context_tokens is not None and self.model_context_tokens <= 0:
            raise ValueError("model_context_tokens deve ser positivo")
        if self.output_tokens < 0 or self.token_safety_margin < 0:
            raise ValueError("reservas de tokens não podem ser negativas")
        if self.budget_bucket < 1:
            raise ValueError("budget_bucket deve ser pelo menos 1")
        if self.mu < 0:
            raise ValueError("mu não pode ser negativo")
        if self.distance not in {"compressor_family", "param_diff", "none"}:
            raise ValueError(f"distance inválida: {self.distance}")
        if self.selection_mode not in {"mckp", "uniform_control"}:
            raise ValueError(f"selection_mode inválido: {self.selection_mode}")
        if not self.option_set:
            raise ValueError("option_set não pode ser vazio")
        if not 0 < self.adaptive_rate_safety <= 1:
            raise ValueError("adaptive_rate_safety deve estar em (0, 1]")
        if not 0 < self.min_adaptive_rate <= 1:
            raise ValueError("min_adaptive_rate deve estar em (0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> "MCKPConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        known = {k: raw[k] for k in raw if k in cls.__dataclass_fields__}
        return cls(**known)
