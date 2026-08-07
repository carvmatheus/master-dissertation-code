"""
Solver do MCKP com penalização de transição.

Resolve, por programação dinâmica, a formulação do Capítulo 3.

  D[1, b, o] = Q_{1,o},                         se c_{1,o} <= b, senão -inf
  D[j, b, o] = Q_{j,o} + max_{o' in O_{j-1}} { D[j-1, b - c_{j,o}, o'] - mu * d(o', o) }
  ótimo      = max_{o in O_n} D[n, B, o]

com escolhas recuperadas por backtracking. A complexidade é O(n B m^2). Quando
mu = 0 a recorrência reduz ao MCKP clássico, sem acoplamento entre partições
adjacentes. Um método de força bruta é fornecido como referência de correção
para instâncias pequenas.

O orçamento é inteiro. Para contextos grandes, os custos podem ser quantizados
em blocos (budget_bucket) de modo a limitar o eixo de orçamento da tabela.
"""
from __future__ import annotations

import math
from itertools import product
from typing import Callable, List

import numpy as np

from .models import CompressionOption, Solution

NEG_INF = -1e18


class InfeasibleMCKPError(ValueError):
    """Nenhuma combinação escolhe uma opção por classe dentro do orçamento."""


def _distance_fn(kind: str) -> Callable[[CompressionOption, CompressionOption], float]:
    if kind == "param_diff":
        return lambda a, b: abs(a.param - b.param)
    if kind == "none":
        return lambda a, b: 0.0
    if kind == "compressor_family":
        return lambda a, b: 0.0 if a.compressor == b.compressor else 1.0
    raise ValueError(f"distância desconhecida: {kind}")


class MCKPSolver:
    def __init__(self, mu: float = 0.0, distance: str = "compressor_family", budget_bucket: int = 1):
        self.mu = float(mu)
        if not math.isfinite(self.mu) or self.mu < 0:
            raise ValueError("mu deve ser finito e não negativo")
        self.distance = distance
        self.budget_bucket = int(budget_bucket)
        if self.budget_bucket < 1:
            raise ValueError("budget_bucket deve ser pelo menos 1")
        self._dist = _distance_fn(distance)

    @staticmethod
    def _validate(options: List[List[CompressionOption]], budget: int) -> None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget deve ser um inteiro não negativo")
        for j, opts in enumerate(options):
            if not opts:
                raise ValueError(f"a classe de opções {j} está vazia")
            for option in opts:
                if option.partition_index != j:
                    raise ValueError(
                        f"opção da classe {j} referencia partição {option.partition_index}"
                    )
                if (
                    isinstance(option.token_cost, bool)
                    or not isinstance(option.token_cost, int)
                    or option.token_cost < 0
                ):
                    raise ValueError("custos devem ser inteiros não negativos")
                if not math.isfinite(option.value):
                    raise ValueError("valores das opções devem ser finitos")

    # ------------------------------------------------------------------
    # Programação dinâmica
    # ------------------------------------------------------------------
    def solve(self, options: List[List[CompressionOption]], budget: int) -> Solution:
        if not options:
            return Solution(chosen=[], total_value=0.0, total_cost=0)
        self._validate(options, budget)

        bucket = self.budget_bucket
        B = budget // bucket
        n = len(options)

        # Custos quantizados (arredonda para cima para não subestimar).
        costs = [
            np.array([math.ceil(o.token_cost / bucket) for o in opts], dtype=np.int64)
            for opts in options
        ]
        values = [
            np.array([o.value for o in opts], dtype=np.float64) for opts in options
        ]

        # Base, partição 0.
        D_prev = np.full((B + 1, len(options[0])), NEG_INF, dtype=np.float64)
        c0, v0 = costs[0], values[0]
        for o in range(len(options[0])):
            if c0[o] <= B:
                D_prev[c0[o]:, o] = v0[o]  # viável para todo orçamento >= custo
        parents: List[np.ndarray] = [np.full((B + 1, len(options[0])), -1, dtype=np.int64)]

        for j in range(1, n):
            m_prev = len(options[j - 1])
            m_cur = len(options[j])
            # Matriz de distância entre opções adjacentes.
            dist = np.zeros((m_prev, m_cur), dtype=np.float64)
            for op in range(m_prev):
                for oc in range(m_cur):
                    dist[op, oc] = self._dist(options[j - 1][op], options[j][oc])

            D_cur = np.full((B + 1, m_cur), NEG_INF, dtype=np.float64)
            parent = np.full((B + 1, m_cur), -1, dtype=np.int64)
            cj, vj = costs[j], values[j]

            for oc in range(m_cur):
                c = int(cj[oc])
                if c > B:
                    continue
                # D_prev deslocado em c ao longo do eixo de orçamento.
                shifted = np.full((B + 1, m_prev), NEG_INF, dtype=np.float64)
                shifted[c:, :] = D_prev[: B + 1 - c, :]
                adjusted = shifted - self.mu * dist[:, oc][None, :]
                best = adjusted.max(axis=1)
                arg = adjusted.argmax(axis=1)
                feasible = best > NEG_INF / 2
                D_cur[feasible, oc] = vj[oc] + best[feasible]
                parent[feasible, oc] = arg[feasible]

            D_prev = D_cur
            parents.append(parent)

        # Ótimo, monotonicidade em b garante que a melhor solução usa o topo.
        last = D_prev[B, :]
        best_o = int(np.argmax(last))
        best_val = float(last[best_o])
        if best_val <= NEG_INF / 2:
            raise InfeasibleMCKPError(
                f"nenhuma solução viável para {n} partições e orçamento {budget}"
            )

        # Backtracking.
        chosen_idx: List[int] = [0] * n
        b = B
        o = best_o
        for j in range(n - 1, -1, -1):
            chosen_idx[j] = o
            c = int(math.ceil(options[j][o].token_cost / bucket))
            prev_o = int(parents[j][b, o])
            b = b - c
            o = prev_o

        chosen = [options[j][chosen_idx[j]] for j in range(n)]
        total_cost = sum(op.token_cost for op in chosen)
        return Solution(chosen=chosen, total_value=best_val, total_cost=total_cost)

    # ------------------------------------------------------------------
    # Referência de correção
    # ------------------------------------------------------------------
    def brute_force(self, options: List[List[CompressionOption]], budget: int) -> Solution:
        """Enumeração exaustiva. Uso restrito a instâncias pequenas (testes)."""
        if not options:
            return Solution(chosen=[], total_value=0.0, total_cost=0)
        self._validate(options, budget)

        best_val = NEG_INF
        best_combo = None
        for combo in product(*[range(len(opts)) for opts in options]):
            chosen = [options[j][combo[j]] for j in range(len(options))]
            cost = sum(op.token_cost for op in chosen)
            if cost > budget:
                continue
            val = sum(op.value for op in chosen)
            for j in range(1, len(chosen)):
                val -= self.mu * self._dist(chosen[j - 1], chosen[j])
            if val > best_val:
                best_val = val
                best_combo = chosen
        if best_combo is None:
            raise InfeasibleMCKPError(
                f"nenhuma solução viável para {len(options)} partições e orçamento {budget}"
            )
        return Solution(
            chosen=best_combo,
            total_value=best_val,
            total_cost=sum(op.token_cost for op in best_combo),
        )


class UniformControlSolver:
    """Controle restrito a uma única família e taxa em todas as partições.

    O conjunto viável é um subconjunto daquele usado pelo MCKP. Assim, a
    comparação isola o efeito da alocação seletiva sem redefinir o Raw como
    baseline de compressão.
    """

    def solve(self, options: List[List[CompressionOption]], budget: int) -> Solution:
        if not options:
            return Solution(chosen=[], total_value=0.0, total_cost=0)
        MCKPSolver._validate(options, budget)

        indexed = [
            {(option.compressor, option.param): option for option in class_options}
            for class_options in options
        ]
        common = set(indexed[0])
        for class_options in indexed[1:]:
            common.intersection_update(class_options)

        best = None
        for key in sorted(common):
            chosen = [class_options[key] for class_options in indexed]
            cost = sum(option.token_cost for option in chosen)
            if cost > budget:
                continue
            value = sum(option.value for option in chosen)
            candidate = (value, -cost, chosen)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            raise InfeasibleMCKPError(
                "nenhuma opção uniforme é viável para "
                f"{len(options)} partições e orçamento {budget}"
            )
        value, neg_cost, chosen = best
        return Solution(chosen=chosen, total_value=value, total_cost=-neg_cost)
