# Testes

- `unit/` — testes unitários (pytest) do código em `01-context-extension-comparison/`.
  Rodar com `pytest tests/unit -v`.
- `../01-context-extension-comparison/mckp/tests/` — testes diferenciais do solver,
  orçamento, cache e reconstrução do método proposto. Rodar com
  `pytest 01-context-extension-comparison/mckp/tests -v`.
- `old-api/` — runs de benchmark antigos (maio/2026) feitos via APIs externas
  (Groq, Cerebras, Gemini), incluindo os runners `groq_run/` e `cerebras_run/`.
  Abandonados por causa de rate limits. Preservados como histórico para a dissertação.
- `ollama-local/` — runs atuais com modelos locais via Ollama
  (ver `ollama-local/README_DECISAO_TESTES.md` para a justificativa de modelos/contextos):
  - `benchmark_ollama_smoke/` e `benchmark_new_datasets_smoke/` — validação do pipeline local (14/jul/2026)
  - `ollama_benchmark_runs/` — matriz modelo × num_ctx (saída de `scripts/run_ollama_benchmark_matrix.py`)
  - `benchmark_calibration_ampla/` — calibração ampla llama3.1-8b em 8k/16k/32k (16/jul/2026)
  - `benchmark_matrix_compressores/` — triagem separada de LLMLingua,
    LLMLingua-2, Selective Context e CPC-MiniLM em 8k/32k nos cinco modelos;
    880 execuções e 879 respostas válidas após um `timeout` do CPC-MiniLM no
    DeepSeek-R1 32B/8k (20/jul/2026)
  - `benchmark_mckp_artigo/` — saída separada do MCKP em 8k/32k, criada após
    a conclusão da matriz principal.
