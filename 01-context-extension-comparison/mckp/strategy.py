"""
Estratégia MCKP.

Orquestra o pipeline completo, particionamento, pontuação de importância, geração
de opções, resolução do MCKP e reconstrução do contexto comprimido. Retorna uma
única string de contexto, na interface esperada pelo runner de benchmarks.

O orçamento efetivo de contexto é obtido por reserva. Do orçamento total do
modelo subtraem-se a instrução, a pergunta e a saída esperada, de modo que apenas
o material de fundo entra na mochila.

  B_contexto = B_modelo - |instrução| - |pergunta| - B_saída
"""
from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .audit import append_audit
from .importance import ImportanceScorer
from .models import MCKPConfig
from .options import OptionGenerator
from .partitioner import build_partitioner
from .reconstructor import reconstruct
from .reconstructor import serialized_option_cost
from .solver import MCKPSolver, UniformControlSolver
from .tokenization import DEFAULT_TOKEN_COUNTER, TokenCounter

try:
    from context_strategies import ContextStrategy
except ImportError:
    from ..context_strategies import ContextStrategy


logger = logging.getLogger(__name__)


def _default_prompt_builder(context: str, query: str) -> str:
    return (
        "Baseado no contexto abaixo, responda a pergunta de forma direta e concisa.\n\n"
        f"CONTEXTO:\n{context}\n\nPERGUNTA: {query}\n\nRESPOSTA:"
    )


class MCKPBudgetError(ValueError):
    """O prompt não pode ser materializado dentro do orçamento configurado."""


class MCKPStrategy(ContextStrategy):
    def __init__(
        self,
        config: MCKPConfig,
        token_counter: TokenCounter | None = None,
        prompt_builder: Callable[[str, str], str] | None = None,
        option_cache: Optional[Dict[str, Tuple[str, float]]] = None,
    ):
        self.config = config
        self.token_counter = token_counter or DEFAULT_TOKEN_COUNTER
        self.prompt_builder = prompt_builder or _default_prompt_builder
        self.partitioner = build_partitioner(config)
        self.importance = ImportanceScorer(config)
        self.option_generator = OptionGenerator(
            config,
            token_counter=self.token_counter,
            cache=option_cache,
        )
        if config.selection_mode == "uniform_control":
            self.solver = UniformControlSolver()
        else:
            self.solver = MCKPSolver(
                mu=config.mu,
                distance=config.distance,
                budget_bucket=config.budget_bucket,
            )
        self.last_diagnostics: Dict[str, object] = {}

    def _budget(self, query: str) -> int:
        if self.config.model_context_tokens is not None:
            prompt_without_context = self.prompt_builder("", query)
            reserved = self.token_counter.count(prompt_without_context)
            reserved += self.config.output_tokens + self.config.token_safety_margin
            budget = self.config.model_context_tokens - reserved
        else:
            budget = self.config.budget_tokens
        if budget < 0:
            raise MCKPBudgetError(
                "prompt, saída e margem de segurança excedem a janela do modelo"
            )
        return int(budget)

    def process(
        self,
        text: str,
        query: str,
        required_partition_indices: Optional[List[int]] = None,
    ) -> List[str]:
        budget = self._budget(query)

        partitions = self.partitioner.partition(text, query)
        required = set(required_partition_indices or [])
        for partition in partitions:
            partition.required = partition.index in required
        if not partitions:
            self.last_diagnostics = {
                "budget_tokens": budget,
                "solver_cost_tokens": 0,
                "actual_context_tokens": 0,
                "num_partitions": 0,
                "num_compressor_failures": 0,
                "budget_bucket": self.config.budget_bucket,
            }
            return [""]

        self.importance.score(partitions, query)
        option_set = list(self.config.option_set)
        adaptive_rate = None
        adaptive_rates: List[float] = []
        if self.config.adaptive_budget_rate:
            identity_cost = sum(
                serialized_option_cost(
                    partition.text, partition.index, self.token_counter
                )
                for partition in partitions
            )
            if identity_cost > budget:
                adaptive_rate = max(
                    self.config.min_adaptive_rate,
                    min(1.0, budget / identity_cost * self.config.adaptive_rate_safety),
                )
                adaptive_rate = round(adaptive_rate, 4)
                adaptive_rates = [adaptive_rate]
                if self.config.min_adaptive_rate < adaptive_rate:
                    adaptive_rates.append(self.config.min_adaptive_rate)
                families = {
                    str(spec["compressor"])
                    for spec in option_set
                    if spec["compressor"] not in {"identity", "omission"}
                }
                existing = {
                    (str(spec["compressor"]), float(spec.get("param", 1.0)))
                    for spec in option_set
                }
                for compressor in sorted(families):
                    for rate in adaptive_rates:
                        if (compressor, rate) not in existing:
                            option_set.append(
                                {"compressor": compressor, "param": rate}
                            )
        options = self.option_generator.generate(
            partitions, query, option_set=option_set
        )
        solution = self.solver.solve(options, budget)
        compressed = reconstruct(solution.chosen)
        actual_cost = self.token_counter.count(compressed)
        if actual_cost > budget:
            raise MCKPBudgetError(
                f"reconstrução excedeu o orçamento: {actual_cost} > {budget} tokens"
            )
        prompt_tokens = self.token_counter.count(self.prompt_builder(compressed, query))
        if self.config.model_context_tokens is not None:
            reserved_total = (
                prompt_tokens
                + self.config.output_tokens
                + self.config.token_safety_margin
            )
            if reserved_total > self.config.model_context_tokens:
                raise MCKPBudgetError(
                    "prompt final e reservas excederam a janela: "
                    f"{reserved_total} > {self.config.model_context_tokens} tokens"
                )

        failures = list(self.option_generator.failures)
        if failures:
            logger.warning(
                "MCKP ignorou %d opção(ões) por falha de compressor",
                len(failures),
            )
        self.last_diagnostics = {
            "budget_tokens": budget,
            "solver_cost_tokens": solution.total_cost,
            "actual_context_tokens": actual_cost,
            "prompt_tokens": prompt_tokens,
            "unused_budget_tokens": budget - actual_cost,
            "total_value": solution.total_value,
            "num_partitions": len(partitions),
            "num_options": sum(len(class_options) for class_options in options),
            "num_compressor_failures": len(failures),
            "num_option_adjustments": len(self.option_generator.adjustments),
            "option_cache_hits": self.option_generator.cache_hits,
            "option_cache_misses": self.option_generator.cache_misses,
            "compressor_failures": failures,
            "budget_bucket": self.config.budget_bucket,
            "selection_mode": self.config.selection_mode,
            "adaptive_rate": adaptive_rate,
            "adaptive_rates": adaptive_rates,
            "evaluated_option_set": option_set,
            "chosen_compressors": [option.compressor for option in solution.chosen],
        }
        chosen_by_partition = {o.partition_index: o for o in solution.chosen}
        audit_record = {
            "schema_version": 1,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "query": query,
            "budget": {
                "available_tokens": budget,
                "solver_cost_tokens": solution.total_cost,
                "actual_context_tokens": actual_cost,
                "unused_tokens": budget - actual_cost,
                "prompt_tokens_estimated": prompt_tokens,
            },
            "solver": {
                "total_value": solution.total_value,
                "mu": self.config.mu,
                "distance": self.config.distance,
                "budget_bucket": self.config.budget_bucket,
                "selection_mode": self.config.selection_mode,
                "adaptive_rate": adaptive_rate,
                "adaptive_rates": adaptive_rates,
                "evaluated_option_set": option_set,
            },
            "partitions": [],
            "compressor_failures": failures,
            "option_rejections": list(self.option_generator.rejections),
            "option_adjustments": list(self.option_generator.adjustments),
        }
        for partition, class_options in zip(partitions, options):
            chosen = chosen_by_partition[partition.index]
            part_record = {
                "index": partition.index,
                "kind": partition.kind,
                "required": partition.required,
                "text_sha256": hashlib.sha256(
                    partition.text.encode("utf-8")
                ).hexdigest(),
                "importance": partition.importance,
                "importance_components": partition.importance_components,
                "options": [
                    {
                        "compressor": option.compressor,
                        "param": option.param,
                        "token_cost": option.token_cost,
                        "fidelity": option.fidelity,
                        "value": option.value,
                        **({"text": option.text} if self.config.audit_include_text else {}),
                    }
                    for option in class_options
                ],
                "chosen": {
                    "compressor": chosen.compressor,
                    "param": chosen.param,
                    "token_cost": chosen.token_cost,
                    "fidelity": chosen.fidelity,
                    "value": chosen.value,
                },
            }
            if self.config.audit_include_text:
                part_record["text"] = partition.text
            audit_record["partitions"].append(part_record)
        if self.config.audit_include_text:
            audit_record["compressed_context"] = compressed
        if self.config.audit_log_path:
            append_audit(self.config.audit_log_path, audit_record)
        self.last_diagnostics["audit_record"] = audit_record
        return [compressed]
