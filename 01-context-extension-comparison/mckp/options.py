"""
Gerador de opções de compressão.

Para cada partição e cada compressor ativo, materializa uma CompressionOption
com custo em tokens, fidelidade e valor Q_{j,o} = I(t_j;q) * f_{j,o}. As opções
de uma partição formam uma classe do MCKP, entre as quais o solver escolhe
exatamente uma.

Cada partição recebe sempre as opções de identidade e de omissão, o que garante
a existência de uma solução viável para qualquer orçamento, já que a omissão tem
custo nulo.

Opções que degradam demais números ou datas permanecem viáveis, mas têm sua
fidelidade limitada pela retenção factual. Isso preserva a comparabilidade com
o controle uniforme e faz o solver evitar essas opções quando há alternativa.

Chamadas caras de compressão são memorizadas por texto, compressor, parâmetro e,
quando o compressor é orientado à tarefa, pela consulta.
"""
from __future__ import annotations

import hashlib
import gc
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .compressors import build_compressor, compressor_uses_query
from .fidelity import fidelity, number_retention
from .models import CompressionOption, MCKPConfig, Partition
from .reconstructor import serialized_option_cost
from .tokenization import DEFAULT_TOKEN_COUNTER, TokenCounter


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(text: str, compressor: str, param: float, query: str = "") -> str:
    query_key = _digest(query) if compressor_uses_query(compressor) else "global"
    return f"{_digest(text)}:{query_key}:{compressor}:{param}"


class OptionGenerator:
    def __init__(
        self,
        config: MCKPConfig,
        token_counter: TokenCounter | None = None,
        cache: Optional[Dict[str, Tuple[str, float]]] = None,
    ):
        self.config = config
        self.token_counter = token_counter or DEFAULT_TOKEN_COUNTER
        self._compressors: Dict[str, object] = {}
        # O custo depende da posição da partição e é calculado após o cache.
        self._cache = cache if cache is not None else {}
        self.failures: List[Dict[str, object]] = []
        self.rejections: List[Dict[str, object]] = []
        self.adjustments: List[Dict[str, object]] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _get(self, name: str):
        if name not in self._compressors:
            self._compressors[name] = build_compressor(name, self.config)
        return self._compressors[name]

    def _materialize(
        self, text: str, compressor: str, param: float, query: str, partition_index: int
    ) -> Optional[Tuple[str, float]]:
        key = _key(text, compressor, param, query)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        if compressor == "identity":
            result = (text, 1.0)
        elif compressor == "omission":
            result = ("", 0.0)
        else:
            try:
                comp = self._get(compressor)
            except Exception as exc:
                self._record_failure(text, compressor, param, exc)
                return None
            try:
                compressed = comp.compress(text, param, query=query)
            except Exception as exc:
                self._record_failure(text, compressor, param, exc)
                return None
            if compressed is None or str(compressed).startswith("[Erro"):
                # Erro real de execução: o compressor sinalizou falha.
                self._record_failure(
                    text, compressor, param, ValueError("compressor retornou saída inválida")
                )
                return None
            if not str(compressed).strip():
                # Saída vazia legítima: nesta partição o compressor não reteve
                # conteúdo, o que equivale à omissão. Rejeita-se explicitamente
                # a opção em vez de tratá-la como erro de execução.
                self.rejections.append({
                    "partition_index": partition_index,
                    "compressor": compressor,
                    "param": param,
                    "reason": "empty_output",
                })
                return None
            result = (
                compressed,
                fidelity(
                    text,
                    compressed,
                    enable_semantic=self.config.semantic_fidelity,
                ),
            )

        self._cache[key] = result
        return result

    def _record_failure(
        self, text: str, compressor: str, param: float, exc: Exception
    ) -> None:
        self.failures.append(
            {
                "text_hash": _digest(text)[:12],
                "compressor": compressor,
                "param": param,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    def generate(
        self,
        partitions: List[Partition],
        query: str,
        option_set: Optional[List[Dict[str, float]]] = None,
    ) -> List[List[CompressionOption]]:
        """Retorna, por partição, a lista de opções (a classe do MCKP)."""
        self.failures = []
        self.rejections = []
        self.adjustments = []
        self.cache_hits = 0
        self.cache_misses = 0
        specs = list(option_set if option_set is not None else self.config.option_set)
        # Garante identidade e omissão presentes.
        active = {s["compressor"] for s in specs}
        if "identity" not in active:
            specs.append({"compressor": "identity", "param": 1.0})
        if "omission" not in active:
            specs.append({"compressor": "omission", "param": 0.0})

        threshold = self.config.number_retention_threshold
        all_options: List[List[CompressionOption]] = [[] for _ in partitions]
        specs_by_compressor: OrderedDict[str, List[Dict[str, float]]] = OrderedDict()
        for spec in specs:
            specs_by_compressor.setdefault(str(spec["compressor"]), []).append(spec)

        # Processa uma família por vez. Isso evita manter simultaneamente
        # XLM-RoBERTa, GPT-2 e os encoders MiniLM residentes no mesmo processo.
        for name, family_specs in specs_by_compressor.items():
            family_cached = all(
                _key(part.text, name, float(spec.get("param", 1.0)), query)
                in self._cache
                for part in partitions
                for spec in family_specs
            )
            if name not in {"identity", "omission"} and not family_cached:
                try:
                    self._get(name)
                except Exception as exc:
                    self._record_failure("", name, float(family_specs[0].get("param", 1.0)), exc)
                    if name in self.config.required_compressors:
                        raise RuntimeError(
                            f"compressores obrigatórios indisponíveis: {name} "
                            f"({type(exc).__name__}: {exc})"
                        ) from exc
                    continue

            for part, options in zip(partitions, all_options):
                for spec in family_specs:
                    param = float(spec.get("param", 1.0))
                    if part.required and name == "omission":
                        self.rejections.append({
                            "partition_index": part.index,
                            "compressor": name,
                            "param": param,
                            "reason": "required_partition",
                        })
                        continue
                    mat = self._materialize(part.text, name, param, query, part.index)
                    if mat is None:
                        continue
                    text, fid = mat
                    cost = serialized_option_cost(text, part.index, self.token_counter)

                    # Penalização factual (identidade e omissão isentas).
                    if name not in ("identity", "omission"):
                        nr = number_retention(part.text, text)
                        if nr is not None and nr < threshold:
                            original_fidelity = fid
                            fid = min(fid, nr)
                            self.adjustments.append({
                                "partition_index": part.index,
                                "compressor": name,
                                "param": param,
                                "reason": "number_retention_penalty",
                                "number_retention": nr,
                                "threshold": threshold,
                                "fidelity_before": original_fidelity,
                                "fidelity_after": fid,
                            })

                    options.append(
                        CompressionOption(
                            partition_index=part.index,
                            compressor=name,
                            param=param,
                            text=text,
                            token_cost=cost,
                            fidelity=fid,
                            importance=part.importance,
                        )
                    )

            if name not in {"identity", "omission"}:
                self._compressors.pop(name, None)
                gc.collect()

        for part, options in zip(partitions, all_options):
            # Segurança, sempre existe identidade e, salvo quando protegida, omissão.
            if not any(o.compressor == "identity" for o in options):
                options.append(
                    CompressionOption(
                        part.index, "identity", 1.0, part.text,
                        serialized_option_cost(part.text, part.index, self.token_counter),
                        1.0, part.importance,
                    )
                )
            if not part.required and not any(o.compressor == "omission" for o in options):
                options.append(
                    CompressionOption(
                        part.index, "omission", 0.0, "", 0, 0.0, part.importance
                    )
                )

        return all_options
