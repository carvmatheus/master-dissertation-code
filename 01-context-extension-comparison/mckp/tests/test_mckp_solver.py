"""
Correção do solver de MCKP contra enumeração exaustiva.

Verifica que a programação dinâmica reproduz o ótimo da força bruta em
instâncias aleatórias pequenas, tanto no caso clássico (mu = 0) quanto com
penalização de transição (mu > 0).
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mckp.models import CompressionOption, MCKPConfig
from mckp.solver import InfeasibleMCKPError, MCKPSolver


def _random_instance(rng):
    n = rng.randint(1, 5)
    options = []
    for j in range(n):
        m = rng.randint(1, 4)
        opts = []
        for o in range(m):
            # A primeira opção de cada partição tem custo zero (garante viabilidade).
            cost = 0 if o == 0 else rng.randint(1, 10)
            fid = round(rng.random(), 3)
            comp = rng.choice(["identity", "semantic", "omission", "extractive"])
            opts.append(
                CompressionOption(
                    partition_index=j,
                    compressor=comp,
                    param=round(rng.random(), 2),
                    text="x" * cost,
                    token_cost=cost,
                    fidelity=fid,
                    importance=1.0,
                )
            )
        options.append(opts)
    budget = rng.randint(0, 25)
    return options, budget


@pytest.mark.parametrize("seed", range(60))
def test_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    options, budget = _random_instance(rng)
    mu = rng.choice([0.0, 0.0, 0.5, 1.0, 2.0])
    dist = rng.choice(["compressor_family", "param_diff", "none"])

    solver = MCKPSolver(mu=mu, distance=dist, budget_bucket=1)
    dp = solver.solve(options, budget)
    bf = solver.brute_force(options, budget)

    assert dp.total_value == pytest.approx(bf.total_value, abs=1e-6)
    assert dp.total_cost <= budget


@pytest.mark.parametrize("seed", range(30))
def test_mu_zero_is_classical_mckp(seed):
    """Com mu = 0 o valor ótimo é a soma dos melhores itens viáveis por classe."""
    rng = random.Random(1000 + seed)
    options, budget = _random_instance(rng)

    solver = MCKPSolver(mu=0.0, distance="compressor_family", budget_bucket=1)
    dp = solver.solve(options, budget)
    bf = solver.brute_force(options, budget)
    assert dp.total_value == pytest.approx(bf.total_value, abs=1e-6)


def test_one_option_per_partition_selected():
    rng = random.Random(7)
    options, budget = _random_instance(rng)
    solver = MCKPSolver(mu=0.5, distance="param_diff", budget_bucket=1)
    sol = solver.solve(options, budget)
    assert len(sol.chosen) == len(options)
    for j, opt in enumerate(sol.chosen):
        assert opt.partition_index == j


def _option(partition, compressor, cost, fidelity):
    return CompressionOption(
        partition_index=partition,
        compressor=compressor,
        param=1.0,
        text=compressor,
        token_cost=cost,
        fidelity=fidelity,
        importance=1.0,
    )


def test_default_configuration_uses_exact_budget_axis():
    assert MCKPConfig().budget_bucket == 1


def test_exact_solver_keeps_feasible_cross_bucket_combination():
    options = [
        [_option(0, "identity", 9, 1.0), _option(0, "omission", 0, 0.0)],
        [_option(1, "identity", 7, 1.0), _option(1, "omission", 0, 0.0)],
    ]
    solution = MCKPSolver(budget_bucket=1).solve(options, budget=16)
    assert solution.total_value == pytest.approx(2.0)
    assert solution.total_cost == 16


def test_infeasible_instance_is_reported():
    options = [[_option(0, "identity", 5, 1.0)]]
    solver = MCKPSolver()
    with pytest.raises(InfeasibleMCKPError):
        solver.solve(options, budget=1)
    with pytest.raises(InfeasibleMCKPError):
        solver.brute_force(options, budget=1)


def test_empty_option_class_is_rejected():
    with pytest.raises(ValueError, match="classe de opções 0 está vazia"):
        MCKPSolver().solve([[]], budget=10)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_budget_is_rejected(budget):
    options = [[_option(0, "omission", 0, 0.0)]]
    with pytest.raises(ValueError, match="budget"):
        MCKPSolver().solve(options, budget=budget)
