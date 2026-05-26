# Phase 1: RSAW Implementation — Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Implementar o framework RSAW (Relevance-Stratified Adaptive Window) como submódulo Python dividido em `segmenter.py`, `scorer.py`, `assembler.py` e `__init__.py` dentro de `01-context-extension-comparison/rsaw/`. Integrar ao `run_benchmarks.py` como estratégia `"rsaw"` com parâmetros de experimento lidos de `rsaw/config.json`. Validar empiricamente que o RSAW atinge score RULER ≥ 0.70 com ao menos um modelo.

</domain>

<spec_lock>
## Requirements (locked via SPEC.md)

**7 requirements are locked.** Ver `spec-driven_development/01-RSAW-SPEC.md` para requisitos completos, boundaries e acceptance criteria.

Downstream agents MUST read `spec-driven_development/01-RSAW-SPEC.md` antes de planejar ou implementar. Os requisitos não são duplicados aqui.

**In scope (from SPEC.md):**
- Submódulo `01-context-extension-comparison/rsaw/` com `__init__.py` e implementação dividida
- Classe `RSAWStrategy(ContextStrategy)` com pipeline de 4 etapas
- Contagem de tokens reais via `tiktoken`
- Função `create_rsaw_strategy(model)` em `run_benchmarks.py`
- Registro de `"rsaw"` em `strategy_factories` de `run_benchmarks.py`
- Atualização de `requirements.txt` com `tiktoken`
- Execução dos benchmarks para gerar CSVs comparativos

**Out of scope (from SPEC.md):**
- Integração com `chat.py` (menu interativo opção 5)
- Testes unitários (`tests/test_rsaw.py`)
- Otimização automática de pesos α/β/γ
- Cache persistente do índice FAISS entre sessões
- Módulo 02 (Framework Socrático)

</spec_lock>

<decisions>
## Implementation Decisions

### D1 — Tokenização (tiktoken)

- **D-01:** Usar `tiktoken` com encoding `o200k_base` como padrão — corresponde aos modelos `gpt-oss-120b` e `gpt-oss-20b` usados nos benchmarks.
- **D-02:** Fallback automático para `cl100k_base` se `tiktoken.encoding_for_model()` lançar `KeyError` — garante compatibilidade com modelos Llama (usados via Groq) sem interromper a execução.
- **D-03:** A contagem de tokens é usada em dois lugares: (a) segmentação em chunks de `chunk_size` tokens exatos (`segmenter.py`), e (b) contabilidade do orçamento durante montagem (`assembler.py`).

```python
# Padrão de implementação esperado
try:
    enc = tiktoken.encoding_for_model(model_name)
except KeyError:
    enc = tiktoken.get_encoding("cl100k_base")
```

### D2 — Tier 2: Compressão por Perplexidade

- **D-04:** `tier2_ratio` é parâmetro **explícito e obrigatório** no construtor de `RSAWStrategy` — sem valor default. O pesquisador deve raciocinar sobre o ratio antes de cada experimento.
- **D-05:** Se após inserir Tier 1 o orçamento já estiver esgotado, o Tier 2 é **pulado com aviso** via `print()` ou `logging.warning()` indicando quantos chunks Tier 2 foram omitidos. O resultado ainda é retornado com Tier 1.
- **D-06:** Compressão do Tier 2 usa `PerplexityCompressor` de `prompt_compression.py` — não reimplementar.

### D3 — Tier 3: Resumo Semântico

- **D-07:** `summarizer_model` é parâmetro **explícito e obrigatório** no construtor de `RSAWStrategy`. Não há modelo hardcoded — o mesmo modelo do benchmark é repassado (ex: `"openai/gpt-oss-120b"`).
- **D-08:** Se `GroqSemanticCompressor` falhar por qualquer motivo (sem `GROQ_API_KEY`, erro de API, timeout): **logar aviso e retornar contexto sem Tier 3**. O contexto Tier 1 + Tier 2 é suficiente — não propagar a exceção.
- **D-09:** Tier 3 é processado apenas se ainda houver orçamento após Tier 1 + Tier 2. Sem orçamento → skip silencioso (sem aviso, pois é comportamento esperado).

### D4 — Estrutura do Submódulo `rsaw/`

- **D-10:** Estrutura dividida em **4 arquivos** dentro de `01-context-extension-comparison/rsaw/`:
  - `segmenter.py` — Etapa 1: divide texto em chunks de `chunk_size` tokens com `overlap` tokens de sobreposição. Usa tiktoken.
  - `scorer.py` — Etapa 2: envolve `DartboardProcessor` de `rig/` para pontuação Dartboard. Retorna lista de `(chunk, score)`.
  - `assembler.py` — Etapas 3 e 4: estratificação em tiers + montagem com orçamento dinâmico + inserção de marcadores.
  - `__init__.py` — exporta `RSAWStrategy` diretamente: `from rsaw import RSAWStrategy`.

- **D-11:** `RSAWStrategy` é definida em `__init__.py` (ou importada de `assembler.py`) e herda de `ContextStrategy` de `context_strategies.py`. Interface: `process(text: str, query: str) -> List[str]`.

- **D-12:** Parâmetros de benchmark definidos em **`rsaw/config.json`** (versionado no repositório). Lido por `create_rsaw_strategy(model)` em `run_benchmarks.py`. Estrutura esperada:

```json
{
  "theta_alto": 0.7,
  "theta_baixo": 0.4,
  "budget_tokens": 4000,
  "chunk_size": 500,
  "overlap": 50,
  "tier2_ratio": 0.5,
  "top_k": 5,
  "alpha": 0.7,
  "beta": 0.2,
  "gamma": 0.1
}
```

- **D-13:** `create_rsaw_strategy(model)` em `run_benchmarks.py` lê `rsaw/config.json`, instancia `RSAWStrategy(**config, summarizer_model=model)`, e retorna `strategy_fn(context, query) -> str` compatível com `BenchmarkRunner`.

### Construtor Completo Esperado de RSAWStrategy

```python
RSAWStrategy(
    theta_alto: float,          # limiar Tier 1
    theta_baixo: float,         # limiar Tier 2/3
    budget_tokens: int,         # orçamento total em tokens
    chunk_size: int,            # tamanho de cada chunk em tokens
    overlap: int,               # overlap entre chunks em tokens
    tier2_ratio: float,         # compression_ratio para PerplexityCompressor
    top_k: int,                 # top-k inicial para DartboardProcessor
    alpha: float,               # peso semântico Dartboard
    beta: float,                # peso lexical Dartboard
    gamma: float,               # peso importância Dartboard
    summarizer_model: str,      # model_name para GroqSemanticCompressor (Tier 3)
)
```

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spec e Requisitos
- `spec-driven_development/01-RSAW-SPEC.md` — **7 requisitos locked. LEITURA OBRIGATÓRIA.** Boundaries, constraints e acceptance criteria completos.

### Arquitetura RSAW (dissertação)
- `Master_Thesis___Matheus_Cardoso.pdf` — PDF completo da dissertação. Seção 3.2 descreve RSAW formalmente.
- *(Nota: o arquivo LaTeX source está em repositório separado `00-master-thesis-overleaf-help/` — não disponível neste repositório)*

### Código existente para reutilização obrigatória
- `01-context-extension-comparison/rig/dartboard_processor.py` — `DartboardProcessor`: scoring Dartboard com FAISS + TF-IDF + importance. `scorer.py` do RSAW deve envolver esta classe.
- `01-context-extension-comparison/rig/utils.py` — `PORTUGUESE_STOPWORDS`, `normalize_embeddings`.
- `01-context-extension-comparison/prompt_compression.py` — `PerplexityCompressor` (Tier 2) e `GroqSemanticCompressor` (Tier 3).
- `01-context-extension-comparison/context_strategies.py` — `ContextStrategy` (classe base a herdar), `SlidingWindowStrategy` (referência de padrão).
- `01-context-extension-comparison/run_benchmarks.py` — Ponto de integração: `strategy_factories`, `create_rsaw_strategy()`, `get_model_short_name()`.

### Benchmarks
- `01-context-extension-comparison/benchmarks/base.py` — `BenchmarkResult`, `TestCase`, `BaseBenchmark`. Interface que `BenchmarkRunner` usa.
- `benchmark_results/comparison_table.csv` — Resultados atuais. Referência para baseline (RIG=0.000 no RULER).
- `requirements.txt` — Dependências atuais. `tiktoken` precisa ser adicionado.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`DartboardProcessor`** (`rig/dartboard_processor.py`): Encapsula FAISS + SentenceTransformer + TF-IDF. `scorer.py` do RSAW chama `processor.index_text(text)` e `processor.dartboard_ranking(query, top_k)` — não reimplementar.
- **`PerplexityCompressor`** (`prompt_compression.py`): `.compress(text, compression_ratio)` — usado diretamente em `assembler.py` para Tier 2.
- **`GroqSemanticCompressor`** (`prompt_compression.py`): `.compress(text, compression_ratio)` — usado em `assembler.py` para Tier 3. Requer `GROQ_API_KEY`.
- **`ContextStrategy`** (`context_strategies.py`): Interface `process(text, query) -> List[str]` que `RSAWStrategy` deve implementar.

### Established Patterns

- **Submódulo análogo:** `rig/` tem estrutura `__init__.py` + `dartboard_processor.py` + `utils.py`. O `rsaw/` segue a mesma convenção com granularidade maior (4 arquivos por decisão de design).
- **Import pattern:** `from rig import DartboardProcessor` → `from rsaw import RSAWStrategy`. Consistente.
- **Lazy imports:** `rig/dartboard_processor.py` usa `_ensure_imports()` para FAISS/SentenceTransformer. O `rsaw/scorer.py` pode reusar este padrão ou importar diretamente.
- **Strategy function signature:** `create_<strategy>(model) -> Callable[[str, str], str]` — todas as funções factory em `run_benchmarks.py` seguem este padrão. `create_rsaw_strategy(model)` deve seguir identicamente.

### Integration Points

- **`run_benchmarks.py:strategy_factories`** — dicionário onde `"rsaw": lambda m: create_rsaw_strategy(m)` deve ser adicionado.
- **`run_benchmarks.py:parse_args()`** — a lista de estratégias disponíveis no `--strategies` help text deve incluir `"rsaw"`.
- **`requirements.txt`** — adicionar `tiktoken`.
- **`Dockerfile`** — verificar se `tiktoken` requer build dependencies (não requer — é pure Python wheel).
- **`docker-compose.yml`** — nenhuma mudança necessária; `benchmark` service já monta o código correto.

</code_context>

<specifics>
## Specific Ideas

- O arquivo `rsaw/config.json` serve como registro versionado dos hiperparâmetros usados em cada experimento — permite rastrear qual configuração gerou cada resultado de benchmark sem alterar código.
- Marcadores de continuidade no contexto montado devem usar o formato: `[...{N} chunks omitidos...]` (ex: `[...3 chunks omitidos...]`) entre chunks não-adjacentes do Tier 1.
- O `scorer.py` pode envolver `DartboardProcessor` lazily (instanciar na primeira chamada a `process()`) para evitar carregar o modelo de embeddings na importação — mesmo padrão lazy do `rig/dartboard_processor.py`.

</specifics>

<deferred>
## Deferred Ideas

- **Integração com chat.py (menu opção 5)** — `RSAWStrategy` pode ser adicionada ao menu interativo de `chat.py` em sessão futura após validação empírica.
- **Testes unitários `tests/test_rsaw.py`** — Não incluídos nesta fase. Candidato natural para fase posterior à validação de benchmark.
- **Otimização automática de θ_alto, θ_baixo, tier2_ratio** — Grid search / Bayesian optimization dos hiperparâmetros após ter resultados da primeira rodada de benchmarks.
- **Cache persistente FAISS entre sessões** — `DartboardProcessor.save_state/load_state` já existe; integrar ao RSAW para acelerar re-runs com mesmo corpus.

</deferred>

---

*Phase: 01-rsaw-implementation*
*Context gathered: 2026-05-26*
