# Invalid run

Esta rodada não deve ser usada como evidência. O arquivo `longbench.jsonl`
continha amostras oficiais, mas o registry do runner ainda instanciava
`LongBenchTasks`, a implementação sintética anterior. O problema foi detectado
pelos nomes `longbench_multidoc_qa_*` e `longbench_summarization_*`.

A rodada corrigida usa uma raiz posterior e registra `LongBenchBenchmark`.
