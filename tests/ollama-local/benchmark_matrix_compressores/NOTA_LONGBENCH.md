# Nota metodológica — implementação do LongBench na matriz de compressores

## Decisão

Os quatro compressores avaliados (LLMLingua baseline, LLMLingua-2, CPC e
Selective Context) foram executados sobre a **mesma implementação sintética do
LongBench** já utilizada nas cinco estratégias da matriz original
(`benchmark_matrix_artigo`), a saber, a classe `LongBenchTasks`
(`01-context-extension-comparison/benchmarks/longbench.py`).

Os casos são idênticos em ambas as matrizes.

- `longbench_multidoc_qa_1`
- `longbench_multidoc_qa_2`
- `longbench_multidoc_qa_3`
- `longbench_summarization_1`

## Por que essa escolha

O objetivo da matriz de compressores é servir de material para selecionar três
compressores para o conjunto de opções do MCKP, agregando os resultados aos das
cinco estratégias não invasivas já avaliadas. Para que a agregação seja válida,
cada compressor precisa ser medido sobre **exatamente os mesmos casos de teste**
das estratégias anteriores, sob as mesmas métricas.

O runner havia sido posteriormente alterado para uma versão do LongBench baseada
em dados reais do Hugging Face (`benchmarks/real_world.py`, `LongBenchBenchmark`,
que lê `data/benchmarks/longbench.jsonl`). Adotar essa versão nos novos
compressores tornaria o LongBench não comparável, pois as cinco estratégias
originais foram medidas na versão sintética. Os outros seis benchmarks
(ZeroSCROLLS, Natural Questions, TriviaQA, HotpotQA, MuSiQue e sumarização de
reuniões) permanecem idênticos entre as duas matrizes, então apenas o LongBench
exigia essa reconciliação.

Por isso o runner foi mantido na implementação sintética `LongBenchTasks` para a
matriz de compressores, preservando a comparabilidade dos sete benchmarks entre
as duas matrizes.

## Limitação a declarar no texto

A implementação sintética não corresponde ao benchmark LongBench original de Bai
et al. (2023), que é citado na fundamentação. Trata-se de documentos construídos
internamente que reproduzem o formato de QA multi-documento e sumarização de
contexto longo. Essa limitação deve ser declarada na seção de metodologia ou de
limitações, indicando que a escolha priorizou a comparabilidade interna entre as
estratégias e os compressores avaliados sob condições idênticas, e não a
reprodução do conjunto de dados oficial do LongBench.
