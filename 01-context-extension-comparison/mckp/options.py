"""
Gerador de opções de compressão.

Para cada partição e cada compressor ativo, materializa uma CompressionOption
com custo em tokens, fidelidade e valor Q_{j,o} = I(t_j;q) * f_{j,o}. As opções
de uma partição formam uma classe do MCKP, entre as quais o solver escolhe
exatamente uma.

Cada partição recebe sempre as opções de identidade e de omissão, o que garante
a existência de uma solução viável para qualquer orçamento, já que a omissão tem
custo nulo.

Um pré-filtro descarta opções que degradam demais a retenção de itens factuais
em partições que contêm números ou datas, exceto a identidade, que é sempre
mantida.

Chamadas caras de compressão são memorizadas por texto, compressor, parâmetro e,
quando o compressor é orientado à tarefa, pela consulta.
"""
from __future__ import annotations

import hashlib
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
    def __init__(self, config: MCKPConfig, token_counter: TokenCounter | None = None):
        self.config = config
        self.token_counter = token_counter or DEFAULT_TOKEN_COUNTER
        self._compressors: Dict[str, object] = {}
        # O custo depende da posição da partição e é calculado após o cache.
        self._cache: Dict[str, Tuple[str, float]] = {}
        self.failures: List[Dict[str, object]] = []
        self.rejections: List[Dict[str, object]] = []

    def _get(self, name: str):
        if name not in self._compressors:
            self._compressors[name] = build_compressor(name, self.config)
        return self._compressors[name]

    def _materialize(
        self, text: str, compressor: str, param: float, query: str
    ) -> Optional[Tuple[str, float]]:
        key = _key(text, compressor, param, query)
        if key in self._cache:
            return self._cache[key]

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
            if compressed is None or compressed.startswith("[Erro"):
                self._record_failure(
                    text, compressor, param, ValueError("compressor retornou saída inválida")
                )
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
        self, partitions: List[Partition], query: str
    ) -> List[List[CompressionOption]]:
        """Retorna, por partição, a lista de opções (a classe do MCKP)."""
        self.failures = []
        self.rejections = []
        specs = list(self.config.option_set)
        # Garante identidade e omissão presentes.
        active = {s["compressor"] for s in specs}
        if "identity" not in active:
            specs.append({"compressor": "identity", "param": 1.0})
        if "omission" not in active:
            specs.append({"compressor": "omission", "param": 0.0})

        threshold = self.config.number_retention_threshold
        all_options: List[List[CompressionOption]] = []

        for part in partitions:
            options: List[CompressionOption] = []
            for spec in specs:
                name = spec["compressor"]
                param = float(spec.get("param", 1.0))
                if part.required and name == "omission":
                    self.rejections.append({
                        "partition_index": part.index,
                        "compressor": name,
                        "param": param,
                        "reason": "required_partition",
                    })
                    continue
                mat = self._materialize(part.text, name, param, query)
                if mat is None:
                    continue
                text, fid = mat
                cost = serialized_option_cost(text, part.index, self.token_counter)

                # Pré-filtro de retenção factual (identidade e omissão isentas).
                if name not in ("identity", "omission"):
                    nr = number_retention(part.text, text)
                    if nr is not None and nr < threshold:
                        self.rejections.append({
                            "partition_index": part.index,
                            "compressor": name,
                            "param": param,
                            "reason": "number_retention",
                            "number_retention": nr,
                            "threshold": threshold,
                        })
                        continue

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

            # Segurança, sempre existe pelo menos identidade e omissão.
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
            all_options.append(options)

        failed_required = sorted(
            {
                str(failure["compressor"])
                for failure in self.failures
                if failure["compressor"] in self.config.required_compressors
            }
        )
        if failed_required:
            raise RuntimeError(
                "compressores obrigatórios indisponíveis: " + ", ".join(failed_required)
            )

        return all_options
