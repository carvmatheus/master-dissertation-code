# Phase 1: RSAW Implementation — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-26
**Phase:** 01-rsaw-implementation
**Areas discussed:** Tokenização (tiktoken), Tier 2 — ratio de compressão, Tier 3 — modelo do resumo semântico, Estrutura do submódulo rsaw/

---

## Tokenização (tiktoken)

| Opção | Descrição | Selecionado |
|-------|-----------|-------------|
| `o200k_base` | Encoding dos modelos gpt-oss (GPT-4o). Garante que budget_tokens corresponde ao limite real do modelo | ✓ |
| `cl100k_base` | Encoding clássico GPT-3.5/4. Pode divergir em ~5-10% para gpt-oss | |
| Configurável via parâmetro no construtor | Máxima flexibilidade, mas mais verboso | |

**User's choice:** `o200k_base` com fallback automático para `cl100k_base`

| Opção (fallback Llama) | Selecionado |
|---|---|
| Fallback para `cl100k_base` se encoding falhar | ✓ |
| Heurística `len(text.split()) * 1.3` para Llama | |
| Forçar tiktoken, falhar explicitamente | |

**Notes:** Modelos Llama via Groq não têm encoding tiktoken nativo — o fallback garante que o segmentador funciona para todos os modelos testados sem interromper benchmarks.

---

## Tier 2 — Ratio de Compressão

| Opção | Descrição | Selecionado |
|-------|-----------|-------------|
| Dinâmico: `ratio = orçamento_restante / tokens_tier2_bruto` | Maximiza informação sem estourar budget | |
| Fixo em 0.5 | Simples e previsível | |
| Parâmetro `tier2_ratio` no construtor | Explícito e configurável por experimento | ✓ |

**User's choice:** `tier2_ratio` como parâmetro obrigatório no construtor

| Opção (budget esgotado) | Selecionado |
|---|---|
| Ignorar silenciosamente | |
| Logar aviso e retornar só Tier 1 | ✓ |
| Reduzir Tier 1 para abrir espaço | |

**Notes:** Parâmetro explícito forçado (sem default) — consistente com a decisão de spec-phase de não ter defaults em nenhum parâmetro de RSAWStrategy, para que o pesquisador raciocine sobre cada valor antes de rodar.

---

## Tier 3 — Modelo do Resumo Semântico

| Opção (model_name) | Descrição | Selecionado |
|---|---|---|
| Sempre `llama-3.1-8b-instant` | Leve e barato, independente do modelo principal | |
| Mesmo modelo do benchmark via parâmetro | Flexível; pode ser caro com gpt-oss-120b | ✓ |
| Sem sumarização — Tier 3 vira só marcador | Simplifica implementação | |

**User's choice:** `summarizer_model` como parâmetro obrigatório (mesmo modelo do benchmark repassado)

| Opção (falha de API) | Selecionado |
|---|---|
| Pular Tier 3 silenciosamente | |
| Logar aviso e retornar sem Tier 3 | ✓ |
| Propagar exceção | |

**Notes:** Logar aviso (não silencioso) porque uma falha no Tier 3 pode indicar problema de API key que afeta todo o experimento — o pesquisador precisa saber que o resumo não foi gerado.

---

## Estrutura do Submódulo rsaw/

| Opção (organização) | Descrição | Selecionado |
|---|---|---|
| Arquivo único `rsaw_processor.py` + `__init__.py` | Análogo ao `rig/` existente | |
| `segmenter.py` + `scorer.py` + `assembler.py` + `__init__.py` | Mais granular, separação clara por etapa | ✓ |
| `RSAWStrategy` em `context_strategies.py` + pasta `rsaw/` para utils | Mais consistente com Sliding/Parallel | |

**User's choice:** Estrutura dividida em 3 arquivos por responsabilidade

| Opção (__init__.py) | Selecionado |
|---|---|
| `from rsaw import RSAWStrategy` | ✓ |
| `from rsaw.rsaw_processor import RSAWStrategy` | |
| Exportar `RSAWStrategy` + constantes de configuração | |

| Opção (parâmetros de benchmark) | Selecionado |
|---|---|
| `theta_alto=0.7, theta_baixo=0.4, ...` hardcoded | |
| Variáveis de ambiente `RSAW_THETA_ALTO` | |
| Arquivo `rsaw/config.json` versionado | ✓ |

**Notes:** `config.json` permite rastrear qual configuração gerou cada resultado de benchmark sem alterar código — importante para reprodutibilidade científica da dissertação.

---

## Claude's Discretion

- Nenhuma área delegada ao Claude nesta discussão — todas as decisões foram explicitamente tomadas pelo usuário.

## Deferred Ideas

- Integração com `chat.py` menu interativo (opção 5) — pós validação empírica
- Testes unitários `tests/test_rsaw.py` — fase posterior
- Grid search de hiperparâmetros (θ_alto, θ_baixo, tier2_ratio) — após primeira rodada de benchmarks
- Cache persistente FAISS entre sessões — integrar `DartboardProcessor.save_state/load_state`
