# Validação do piloto MCKP

## Configuração

- Modelo: `llama3.1:8b-instruct-q8_0`
- Servidor local: Ollama 0.32.3
- Janela: 8.192 tokens
- Estratégias: `raw`, `mckp` e `mckp_uniform_control`
- Benchmarks: LongBench, ZeroScrolls, Natural Questions, TriviaQA, HotpotQA,
  MuSiQue e Meeting Summarization
- Amostra: um caso por benchmark e estratégia, totalizando 21 execuções
- Parâmetros principais: `mu=0.1` e `budget_bucket=1`

## Resultado downstream

| Estratégia | Média macro | Latência média |
|---|---:|---:|
| Raw | 0,699 | 13,19 s |
| MCKP | 0,580 | 87,70 s |
| Controle uniforme | 0,657 | 60,03 s |

O MCKP empatou com Raw em ZeroScrolls, TriviaQA, HotpotQA e MuSiQue. Obteve
resultado inferior em LongBench, Natural Questions e Meeting Summarization.
O controle uniforme superou o MCKP nesses três casos e ficou 0,013 acima de
Raw em Meeting Summarization. A amostra contém uma única execução por tarefa;
essas diferenças são descritivas e não constituem estimativas estatísticas.

## Integridade

- 21 resultados presentes: sete por estratégia e três por benchmark.
- Nenhum resultado contém `error` ou `ollama_error`.
- 14 registros no `mckp_audit.jsonl`: sete MCKP e sete controles.
- Nenhuma violação do orçamento de contexto.
- Maior uso estimado da janela: 8.173 de 8.192 tokens, incluindo a reserva de
  500 tokens de saída e a margem de segurança de 256 tokens.
- Diferenças entre custo do solver e contagem reconstruída ocorreram em seis
  casos MCKP, sempre com a contagem real menor que a estimada.

## Seleção de compressores

Nas 167 partições processadas pelo MCKP, as escolhas foram:

| Compressor | Partições selecionadas |
|---|---:|
| LLMLingua-2 | 99 |
| Identidade | 54 |
| Selective Context | 14 |
| CPC-MiniLM | 0 |
| Omissão | 0 |

No controle uniforme, identidade foi escolhida em cinco casos e LLMLingua-2 em
dois. CPC-MiniLM e Selective Context não foram escolhidos.

## Ocorrências

O LongBench registrou 17 opções inválidas do LLMLingua-2, distribuídas por cinco
partições. A execução descartou essas opções e continuou com alternativas
viáveis. O log também registrou avisos do tokenizador auxiliar ao processar
sequências acima de 512 tokens, principalmente no controle de contexto
integral.

## Decisão

O piloto é aceito como validação da integração, do registro de auditoria e do
controle de orçamento. Ele não é aceito como avaliação final da superioridade
do MCKP. Antes da matriz principal, devem ser investigados:

1. o desalinhamento entre a função de valor e o desempenho downstream, indicado
   pela ausência de seleções do CPC-MiniLM apesar de seu melhor resultado na
   triagem individual;
2. as saídas inválidas do LLMLingua-2 no LongBench;
3. o tratamento de entradas acima do limite declarado pelo tokenizador no
   controle uniforme;
4. o custo de materialização das opções, que elevou a latência média do MCKP
   para 6,65 vezes a latência de Raw neste piloto.

Os artefatos primários estão em
`runs/llama3.1-8b-instruct-q8_0/ctx-8192/`.
