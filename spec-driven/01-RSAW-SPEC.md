# Phase 1: RSAW — Specification

**Created:** 2026-05-26
**Ambiguity score:** 0.19 (gate: ≤ 0.20) ✓
**Requirements:** 7 locked

---

## Goal

Implementar o framework **RSAW (Relevance-Stratified Adaptive Window)** como submódulo Python em `01-context-extension-comparison/rsaw/`, registrá-lo em `run_benchmarks.py` como estratégia comparável, e validar que atinge score RULER ≥ 0.70 com ao menos um modelo — superando o RIG que zerhou nesse benchmark.

---

## Background

### Estado atual do repositório

O módulo `01-context-extension-comparison/` possui 5 estratégias implementadas:

| Estratégia | Arquivo | Score RULER (last run) |
|---|---|---|
| raw | `run_benchmarks.py` | 0.639 (120b) / 1.000 (20b) |
| sliding_window | `context_strategies.py` | 0.806 (120b) / 1.000 (20b) |
| parallel_window | `context_strategies.py` | 0.722 (120b) / 1.000 (20b) |
| semantic_compression | `prompt_compression.py` | 0.125 (120b) / 0.000 (20b) |
| rig | `context_strategies.py` + `rig/` | **0.000 (ambos modelos)** |

O RIG falhou completamente no RULER porque corta em top-K chunks e destrói a continuidade semântica necessária para recuperar múltiplos fatos espalhados. A semantic_compression colapsa silenciosamente em contextos maiores (evidenciado por latências ~70ms). Nenhuma estratégia resolve simultaneamente: (1) recuperação posicionalmente agnóstica, (2) preservação de continuidade, e (3) gestão dinâmica de orçamento de tokens.

O RSAW combina pontuação Dartboard (do RIG) + janelamento adjacente (do sliding_window) + compressão seletiva por tier (da semantic_compression), eliminando os pontos de falha de cada estratégia isolada.

A arquitetura formal do RSAW está descrita em `Chapters/003Methodology/Methodology.tex` (seção 3.2) com figura TikZ, algoritmo LaTeX formal e 4 hipóteses (H1–H4). O código Python ainda não existe.

O submódulo `rig/` (com `DartboardProcessor` e `utils.py`) é análogo ao que será criado para o RSAW — serve como referência de estrutura.

---

## Requirements

### R1 — Submódulo rsaw/ criado

O diretório `01-context-extension-comparison/rsaw/` deve existir com estrutura equivalente ao `rig/`.

- **Current:** O diretório `rsaw/` não existe. Nenhum código RSAW existe no repositório.
- **Target:** `01-context-extension-comparison/rsaw/` contém `__init__.py` e `rsaw_processor.py` (ou equivalente) com a classe principal.
- **Acceptance:** `from rsaw import RSAWStrategy` (ou `from rsaw.rsaw_processor import RSAWStrategy`) executado de dentro de `01-context-extension-comparison/` não lança `ImportError`.

---

### R2 — Classe RSAWStrategy com API compatível com ContextStrategy

A classe `RSAWStrategy` deve herdar de `ContextStrategy` e implementar o método `process(text, query) -> List[str]`.

- **Current:** `ContextStrategy` existe em `context_strategies.py` com interface `process(text: str, query: str) -> List[str]`. Nenhuma classe RSAW existe.
- **Target:** `RSAWStrategy(ContextStrategy)` com construtor que aceita: `theta_alto: float`, `theta_baixo: float`, `budget_tokens: int`, `chunk_size: int`, `overlap: int`, `alpha: float`, `beta: float`, `gamma: float`, `top_k: int`. Sem valores default fixos no construtor — todos os parâmetros devem ser passados explicitamente.
- **Acceptance:** `RSAWStrategy(theta_alto=0.7, theta_baixo=0.4, budget_tokens=4000, chunk_size=500, overlap=50, alpha=0.7, beta=0.2, gamma=0.1, top_k=5).process("texto longo...", "query de teste")` retorna `List[str]` com ao menos 1 elemento sem lançar exceção.

---

### R3 — Pipeline de 4 etapas corretamente implementado

A lógica interna de `process()` deve seguir exatamente as 4 etapas da arquitetura RSAW definida na dissertação.

- **Current:** Nenhuma implementação.
- **Target:**
  - **Etapa 1 (Segmentação):** Divide `text` em chunks de `chunk_size` tokens com `overlap` tokens de sobreposição. Usa `tiktoken` (ou `transformers AutoTokenizer`) para contagem real de tokens.
  - **Etapa 2 (Pontuação Dartboard):** Usa `DartboardProcessor` de `rig/` para calcular `score(ci) = α·sem + β·lex + γ·imp` para cada chunk contra `query`.
  - **Etapa 3 (Estratificação):** Classifica chunks em Tier 1 (`score >= theta_alto`), Tier 2 (`theta_baixo <= score < theta_alto`), Tier 3 (`score < theta_baixo`).
  - **Etapa 4 (Montagem com orçamento):** Injeta Tier 1 inteiro até esgotar `budget_tokens`. Se sobrar orçamento: injeta Tier 2 com `PerplexityCompressor`. Se ainda sobrar: injeta resumo semântico do Tier 3 via `GroqSemanticCompressor`. Insere marcadores `[...N chunks omitidos...]` entre chunks não-adjacentes.
- **Acceptance:** Dado texto de 10.000 tokens e `budget_tokens=2000`, o contexto retornado por `process()` tem contagem de tokens ≤ 2000 (verificável via `tiktoken`).

---

### R4 — Orçamento medido em tokens reais (tiktoken)

A contagem de tokens deve usar uma biblioteca de tokenização real, não heurísticas de caracteres ou palavras.

- **Current:** `DartboardProcessor` usa `chunk_size` em caracteres (`len(word) + 1`). `PerplexityCompressor` usa GPT-2 tokenizer internamente.
- **Target:** O RSAW usa `tiktoken` (ou `transformers AutoTokenizer` com o mesmo modelo do compressor) para: (a) dividir texto em chunks com tamanho exato em tokens, e (b) contabilizar o orçamento consumido durante a montagem.
- **Acceptance:** `rsaw_processor.py` importa `tiktoken` (ou equivalente) e o usa em pelo menos dois pontos: segmentação e contagem de orçamento.

---

### R5 — Registro em run_benchmarks.py como estratégia "rsaw"

A estratégia RSAW deve aparecer na lista de estratégias disponíveis do benchmark runner.

- **Current:** `run_benchmarks.py` tem `strategy_factories` com as chaves: `"raw"`, `"sliding_window"`, `"parallel_window"`, `"semantic_compression"`, `"rig"`. Nenhuma chave `"rsaw"` existe.
- **Target:** `strategy_factories["rsaw"] = lambda m: create_rsaw_strategy(m)` adicionado. A função `create_rsaw_strategy(model)` inicializa `RSAWStrategy` com parâmetros razoáveis e retorna uma `strategy_fn(context, query) -> str` compatível com o `BenchmarkRunner`.
- **Acceptance:** Executar `python run_benchmarks.py --strategies rsaw --mock-only` não lança exceção e registra pelo menos 1 estratégia chamada `"rsaw_mock"` ou equivalente.

---

### R6 — Geração de CSVs com estratégia "rsaw" via docker compose

O benchmark completo com modelos reais deve gerar arquivos CSV com linhas `rsaw_modelname`.

- **Current:** `comparison_table.csv` tem linhas para `raw_*`, `sliding_window_*`, `parallel_window_*`, `semantic_compression_*`, `rig_*`. Nenhuma linha `rsaw_*`.
- **Target:** Ao executar `docker compose run --rm benchmark --strategies rsaw`, o arquivo `benchmark_results/comparison_table.csv` contém ao menos uma linha com prefix `rsaw_`.
- **Acceptance:** Após execução com ao menos um modelo, `grep "^rsaw" benchmark_results/comparison_table.csv` retorna ao menos 1 linha.

---

### R7 — Score RULER ≥ 0.70 com ao menos um modelo testado

A validação empírica da hipótese H2 da dissertação: o RSAW deve superar o RIG no benchmark RULER.

- **Current:** Todos os modelos testados com RIG têm score RULER = 0.000 (`rig_gpt-oss-120b: 0.000`, `rig_gpt-oss-20b: 0.000`).
- **Target:** Ao rodar os benchmarks completos, a linha `rsaw_<modelo>` em `model_comparison.csv` ou `comparison_table.csv` mostra `ruler >= 0.70` para ao menos um modelo.
- **Acceptance:** `grep "rsaw" benchmark_results/comparison_table.csv` contém ao menos uma linha onde o campo `ruler` é ≥ 0.70.

---

## Boundaries

**In scope:**
- Submódulo `01-context-extension-comparison/rsaw/` com `__init__.py` e implementação principal
- Classe `RSAWStrategy(ContextStrategy)` com pipeline de 4 etapas (segmentação → Dartboard → tiers → montagem com orçamento)
- Contagem de tokens reais via `tiktoken` (ou equivalente)
- Função `create_rsaw_strategy(model)` em `run_benchmarks.py`
- Registro de `"rsaw"` em `strategy_factories` de `run_benchmarks.py`
- Atualização de `requirements.txt` com `tiktoken` (se não presente)
- Execução dos benchmarks para gerar CSVs comparativos

**Out of scope:**
- Integração com `chat.py` (menu interativo opção 5) — nice-to-have para sessão futura; o foco é a validação científica via benchmarks
- Testes unitários (`tests/test_rsaw.py`) — não requeridos nesta fase; a validação vem pelos benchmarks empíricos
- Otimização automática de pesos α/β/γ (grid search / tuning) — trabalho futuro pós-benchmark
- Cache persistente do índice FAISS entre sessões — `DartboardProcessor.save_state/load_state` existe mas não é obrigatório usar agora
- Módulo 02 (Framework Socrático) — fase separada, não relacionada ao RSAW

---

## Constraints

- **Reutilização obrigatória:** `DartboardProcessor` de `rig/dartboard_processor.py` deve ser usado para a pontuação Dartboard — não reimplementar a lógica de embeddings/TF-IDF.
- **Reutilização obrigatória:** `PerplexityCompressor` de `prompt_compression.py` deve ser usado para compressão do Tier 2.
- **Reutilização obrigatória:** `GroqSemanticCompressor` de `prompt_compression.py` deve ser usado para resumo do Tier 3 (requer `GROQ_API_KEY`).
- **Compatibilidade de interface:** `RSAWStrategy` deve implementar `ContextStrategy.process(text, query) -> List[str]` — sem quebrar imports existentes em `run_benchmarks.py` ou `chat.py`.
- **Sem defaults no construtor:** Todos os parâmetros de `RSAWStrategy.__init__` devem ser passados explicitamente — sem valores default para θ_alto, θ_baixo e budget_tokens. Isso força o pesquisador a raciocinar sobre os valores em cada experimento.
- **Tokenizer:** Deve usar `tiktoken` (preferência por compatibilidade com GPT-OSS) ou um tokenizer configurável via parâmetro.
- **Docker:** A imagem existente já tem as dependências do `rig/`. `tiktoken` pode precisar ser adicionado ao `requirements.txt` e `Dockerfile`.

---

## Acceptance Criteria

- [ ] `from rsaw import RSAWStrategy` executado de `01-context-extension-comparison/` não lança `ImportError`
- [ ] `RSAWStrategy(...).process(text, query)` retorna `List[str]` com `sum(tokens(c) for c in result) <= budget_tokens` para texto de 10.000 tokens com `budget_tokens=2000`
- [ ] `process()` insere marcadores `[...N chunks omitidos...]` entre chunks não-adjacentes no contexto montado
- [ ] `python run_benchmarks.py --strategies rsaw --mock-only` executa sem exceção e registra estratégia
- [ ] `docker compose run --rm benchmark --strategies rsaw` gera linha `rsaw_*` em `benchmark_results/comparison_table.csv`
- [ ] Ao menos uma linha `rsaw_*` em `comparison_table.csv` com valor de `ruler` ≥ 0.70
- [ ] `requirements.txt` inclui `tiktoken` (ou equivalente para contagem real de tokens)

---

## Ambiguity Report

| Dimension          | Score | Min  | Status | Notes                                                        |
|--------------------|-------|------|--------|--------------------------------------------------------------|
| Goal Clarity       | 0.82  | 0.75 | ✓      | Arquitetura das 4 etapas definida na dissertação             |
| Boundary Clarity   | 0.85  | 0.70 | ✓      | chat.py e testes explicitamente fora do escopo               |
| Constraint Clarity | 0.82  | 0.65 | ✓      | Reutilização de DartboardProcessor/compressores obrigatória  |
| Acceptance Criteria| 0.75  | 0.70 | ✓      | 7 critérios pass/fail incluindo métrica RULER ≥ 0.70         |
| **Ambiguity**      | **0.19** | ≤0.20 | ✓   | Gate passed após 3 rounds                                    |

---

## Interview Log

| Round | Perspectiva      | Pergunta                                        | Decisão locked                                                        |
|-------|------------------|-------------------------------------------------|-----------------------------------------------------------------------|
| 1     | Researcher       | Estrutura: novo arquivo, context_strategies ou rsaw/? | Novo submódulo `rsaw/` análogo ao `rig/`                    |
| 1     | Researcher       | Budget B medido em chars, palavras ou tokens reais? | Tokens reais via `tiktoken` ou `transformers`                    |
| 1     | Researcher       | Registrar no run_benchmarks.py?                 | Sim — gerar CSVs comparáveis com estratégias existentes               |
| 2     | Researcher       | Valores de θ_alto e θ_baixo?                    | Sem defaults fixos — todos os parâmetros obrigatórios no construtor   |
| 2     | Simplifier       | O que significa "done"?                         | CSVs gerados + RULER ≥ 0.70 em ao menos 1 modelo                     |
| 3     | Boundary Keeper  | O que fica FORA do escopo?                      | chat.py menu interativo explicitamente excluído                       |
| 3     | Boundary Keeper  | Testes unitários nesta fase?                    | Não — validação vem pelos benchmarks empíricos                        |

---

*Phase: 01-rsaw-implementation*
*Spec created: 2026-05-26*
*Next step: /gsd:discuss-phase — decisões de implementação (como construir o que foi especificado acima)*
