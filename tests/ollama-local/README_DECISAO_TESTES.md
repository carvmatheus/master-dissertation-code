# Decisão dos testes — seleção de modelos e contextos

Este documento registra **por que** a matriz de benchmarks roda com os modelos,
contextos, estratégias e benchmarks escolhidos. O objetivo é a reprodutibilidade
e a justificativa metodológica para a dissertação.

## Objetivo do experimento

Validar **quanto de informação cada estratégia de compressão consegue preservar**
— ou seja, o quanto o modelo ainda recupera a resposta correta quando o contexto
foi comprimido por uma estratégia, comparado ao baseline sem compressão (`raw`).

A pergunta central: *compressão não é só encurtar; é alocar a perda onde ela
custa menos*. Precisamos medir esse trade-off (custo ↓ × qualidade ↓ × acerto ↓)
de forma consistente entre modelos de **arquiteturas diferentes**, para mostrar
que o efeito das estratégias não depende de um único modelo.

## Escolha dos modelos (5) — diversidade arquitetural

A dissertação não busca *qual modelo é melhor*, e sim se as estratégias de
compressão se comportam de forma consistente em **famílias e arquiteturas
distintas**. Por isso os 5 modelos foram escolhidos para cobrir eixos diferentes
de design, não para serem os "maiores":

| Modelo | Família | Arquitetura | Tamanho | Papel na matriz |
|---|---|---|---|---|
| `llama3.1:8b-instruct-q8_0` | Llama 3.1 (Meta) | **Dense** | 8B | Baseline pequeno e rápido — dense clássico |
| `gemma4:26b-mlx` | Gemma 4 (Google) | **Dense** | ~26B | Dense grande, família diferente de Llama |
| `qwen3:30b-a3b` | Qwen 3 (Alibaba) | **MoE (sparse)** | 30B / 3B ativos | Mixture-of-Experts — só ativa parte dos pesos |
| `deepseek-r1:32b` | DeepSeek-R1 | **Reasoning** | 32B | Ajustado para raciocínio (chain-of-thought) |
| `gpt-oss:20b` | GPT-OSS (OpenAI) | **MoE** | 20B | Família OpenAI aberta — outro estilo de treino |

**Eixos cobertos:** dense pequeno (Llama 8B), dense grande (Gemma 26B), MoE
sparse (Qwen3 MoE + GPT-OSS), e reasoning (DeepSeek-R1). Se as estratégias de
compressão mostram o mesmo padrão de trade-off nesses 5, o resultado generaliza.

### Por que NÃO os outros modelos instalados

| Modelo descartado | Motivo |
|---|---|
| `llama3.1:8b-text-q4_K_M` | Mesma arquitetura do baseline (Llama 3.1) — só muda quantização/variante. Não adiciona diversidade. |
| `qwen3.6:35b-mlx` | Dense grande da Alibaba — redundante com Gemma (dense grande) e com o Qwen3 MoE já incluído. |
| `Llama-4-Scout-17B (Q4, 28GB)` | Mais lento de carregar (28GB) e outra variante Llama. Custo de tempo alto para pouca diversidade extra. |

## Escolha dos contextos — menor e maior (8192 e 32768)

Rodar 3 contextos (8192 / 16384 / 32768) triplica o tempo. A calibração ampla
anterior (llama3.1 em 8k/16k/32k) já mostrou que **16k e 32k saturam no mesmo
score (~0.954)** — o ponto intermediário não trouxe informação nova, só latência.

Por isso, para os demais modelos usamos apenas os **extremos**:

- **8192** — pressão de compressão alta (contexto é bem truncado; a estratégia
  precisa escolher bem o que manter).
- **32768** — pressão baixa (quase tudo cabe; mede o teto de qualidade).

Dois pontos extremos bastam para traçar a curva de trade-off; o intermediário
é dispensável. **Precisamos ser rápidos**, e cortar o 16k reduz ~1/3 do tempo
por modelo sem perder o formato do resultado.

> Exceção: o baseline `llama3.1:8b` roda nos 3 contextos porque já estava em
> execução — mantém uma curva completa de referência sem custo adicional.

## Estimativa de tempo

Ritmo medido no baseline (`llama3.1:8b`, q8): **~47s por execução** em média.
Uma "execução" = 1 benchmark × 1 estratégia × 1 contexto.

| Configuração | Execuções | Estimativa |
|---|---|---|
| 1 modelo × 3 ctx × 7 bench × 5 estrat (baseline atual) | 105 | ~80 min |
| 1 modelo × 2 ctx × 7 bench × 5 estrat | 70 | ~55 min (8B) |
| 5 modelos × 2 ctx × 7 bench × 5 estrat | 350 | **~6–12 h** |

Os modelos de 20–35B são **2–4× mais lentos** por token que o 8B, e o contexto
32768 é mais pesado que o 8192 — por isso a faixa larga (6–12h). Na prática é um
lote para **rodar à noite**. Cortar o contexto intermediário (16k) foi a
principal alavanca para caber nesse orçamento de tempo.

## Estratégias testadas (5)

`raw` (baseline sem compressão), `sliding_window`, `parallel_window`,
`semantic_compression`, `rig` (ranking Dartboard).

> **RSAW foi removido** — era o método autoral em desenvolvimento; o foco atual
> é medir estratégias estabelecidas de compressão, não validar o RSAW.

### Matriz separada do método proposto (MCKP)

O MCKP é avaliado depois das baselines, nos mesmos extremos `8192` e `32768`,
mas em `benchmark_mckp_artigo/`. O diretório separado impede que uma execução
com apenas `--strategies mckp` sobrescreva os resultados das cinco estratégias
da matriz principal.

O caminho experimental usa `budget_bucket=1` (DP exata sobre os custos
declarados) e calcula o orçamento efetivo novamente para cada `num_ctx`.

## Benchmarks (7) — suíte do artigo

Exatamente os benchmarks *text-only* citados na Seção 9.2 do artigo
`ref_docs/artigo_solucao_cpoi.pdf`:

`longbench`, `zeroscrolls`, `naturalquestions`, `triviaqa`, `hotpotqa`,
`musique`, `meeting_summarization`.

Benchmarks sintéticos (`needle_in_haystack`, `ruler`) e `babilong` **não** fazem
parte da suíte oficial do artigo e foram deixados de fora da matriz principal.

## Como rodar

```bash
# Matriz (ajuste --models conforme o modelo da vez)
.venv/bin/python scripts/run_ollama_benchmark_matrix.py \
  --models gemma4:26b-mlx \
  --contexts 8192,32768 \
  --strategies raw,sliding_window,parallel_window,semantic_compression,rig \
  --benchmarks longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization \
  --output-root tests/ollama-local/benchmark_matrix_artigo

# Barra de progresso ao vivo (atualiza no mesmo lugar a cada 15s)
.venv/bin/python scripts/benchmark_progress.py \
  --log tests/ollama-local/benchmark_matrix_artigo_run.log --watch 15

# Calibração (agrega L/Q e correlação métrica×score) — roda no fim
.venv/bin/python scripts/calibrate_from_runs.py \
  --output-root tests/ollama-local/benchmark_matrix_artigo

# Método proposto, Raw e controle uniforme no mesmo conjunto de casos
.venv/bin/python scripts/run_ollama_benchmark_matrix.py \
  --models llama3.1:8b-instruct-q8_0,gemma4:26b-mlx,qwen3:30b-a3b,deepseek-r1:32b,gpt-oss:20b \
  --contexts 8192,32768 \
  --strategies raw,mckp,mckp_uniform_control \
  --mckp-mu 0.1 \
  --mckp-budget-bucket 1 \
  --benchmarks longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization \
  --output-root tests/ollama-local/benchmark_mckp_artigo_v2 \
  --full
```

Para acompanhar o piloto local do Llama 3.1 8B em outra janela do terminal:

```bash
./scripts/watch_mckp_progress.sh
```

O script acompanha por padrão o diretório
`tests/ollama-local/benchmark_mckp_smoke_v5`. Um caminho de log diferente pode
ser informado como primeiro argumento.

`Raw` permanece o único baseline. `mckp_uniform_control` é uma ablação: aplica
uma única família e taxa ao contexto completo sob o mesmo orçamento efetivo.
Ela não deve ser apresentada como baseline adicional.

O conjunto de opções do MCKP contém CPC-MiniLM, LLMLingua-2 e Selective
Context, além de identidade e omissão. A escolha decorre da triagem de
compressores: CPC-MiniLM liderou oito das dez combinações de modelo e orçamento,
LLMLingua-2 liderou as duas combinações do Qwen 3 30B-A3B e Selective Context
apresentou a maior fidelidade semântica. As taxas fixas são 0,5 e 0,3. Quando
elas não são suficientes para satisfazer o orçamento, o runner acrescenta uma
taxa derivada do orçamento com margem de 10% e uma taxa mínima de 0,05; ambas
são registradas em `mckp_audit.jsonl`.

O valor principal inicial é `mu=0.1`, não derivado do teorema. Ele corresponde
a uma regularização de 10% da escala unitária de valor por transição de família.
As ablações `mu=0` e `mu=0.5` devem ser executadas em raízes próprias e sem
consultar os resultados finais para selecionar o hiperparâmetro.

## Triagem de compressores para o MCKP (atualizado em 18/jul/2026)

Antes de fixar as três opções de compressão do MCKP, uma matriz separada avalia
quatro métodos. LLMLingua é o baseline de compressão; LLMLingua-2, Selective
Context e CPC são candidatos. A seleção final deve ser feita somente depois da
validação dos CSVs de qualidade, fidelidade, latência e taxa efetiva de
compressão.

### Delimitação das implementações

- `semantic_compression` não é LLMLingua: é reescrita do contexto pela própria
  LLM, orientada por prompt.
- O `PerplexityCompressor` antigo é apenas um protótipo inspirado no critério de
  perplexidade e não estava registrado na matriz anterior.
- `llmlingua` usa o pacote `llmlingua==0.2.2`, com
  `use_llmlingua2=False`, scorer `gpt2`, `device_map="cpu"` e retenção 0.5.
  O GPT-2 foi escolhido para não disputar a RAM dos modelos Ollama de 20--32B.
- `llmlingua2` usa o mesmo pacote, o checkpoint
  `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` e
  `device_map="cpu"`. A definição explícita do dispositivo corrige a tentativa
  padrão de inicialização em CUDA nesta máquina.
- `selective_context` usa `selective-context==0.1.4`, GPT-2 e inglês. A interface
  recebe a fração removida, então o adaptador converte retenção `rate` em
  `reduce_ratio=1-rate`.
- `cpc` é a aproximação CPC-MiniLM: seleção extrativa de sentenças condicionada
  à pergunta com `all-MiniLM-L6-v2`. O repositório oficial do CPC atualmente
  oferece checkpoints Mistral-7B e Llama-1B, mas a triagem evita carregar outro
  modelo desse porte junto das LLMs locais. Resultados dessa variante não devem
  ser atribuídos ao CPC publicado.

LongLLMLingua não foi incluído para conter o tamanho da matriz e porque a
triagem já cobre compressão por perplexidade, classificação de palavras e
seleção condicionada à pergunta. Essa exclusão não implica equivalência entre
LongLLMLingua e LLMLingua-2. Perception Compressor foi descartado pelo custo de
um scorer de 7B; RECOMP e FILCO, pelo acoplamento a treinamento por tarefa ou
conjunto de dados.

### Ambiente executável

O ambiente usado pelo runner é `.venv`; `.venv-benchmarks` e `es-wrapper` não
contêm o conjunto completo de dependências. As versões observadas na triagem são:

| Dependência | Versão |
|---|---:|
| `llmlingua` | 0.2.2 |
| `selective-context` | 0.1.4 |
| `spacy` | 3.8.14 |
| `en_core_web_sm` | 3.8.0 |
| `sentence-transformers` | 5.6.0 |
| `transformers` | 4.46.3 |
| `torch` | 2.12.1 |

O pacote `selective-context` declara `spacy==3.2.0`, que tenta compilar uma
versão incompatível com este ambiente. A instalação reprodutível usada foi:

```bash
.venv/bin/pip install --only-binary=:all: spacy==3.8.14
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/pip install selective-context==0.1.4 --no-deps
```

`llmlingua==0.2.2` espera o formato legado do KV cache. Com
`transformers==5.12.1`, a compressão iterativa recebia `DynamicCache` e falhava
com `ValueError: too many values to unpack`. O ambiente foi fixado em
`transformers==4.46.3`, versão também compatível com
`sentence-transformers==5.6.0`. Os wrappers propagam falhas de compressão ao
executor para que sejam registradas como erro, em vez de retornar silenciosamente
o contexto original.

O checkpoint XLM-RoBERTa do LLMLingua-2 e os modelos GPT-2/MiniLM são obtidos do
cache do Hugging Face na primeira inicialização.

### Separação de artefatos e protocolo

As quatro estratégias só rodam quando informadas em `--strategies`; elas não
foram adicionadas ao atalho `all`. A matriz anterior permanece intacta em
`benchmark_matrix_artigo/`, enquanto a triagem grava em
`benchmark_matrix_compressores/`. A execução consolidada usa um processo por
combinação de modelo e contexto e registra os quatro compressores conjuntamente.
Assim, os auxiliares podem coexistir durante uma rodada, mas são liberados antes
da combinação seguinte. As latências medem o custo de ponta a ponta desse
protocolo e não isolam o custo computacional de cada compressor.

O protocolo preserva os cinco modelos, os contextos 8192 e 32768, os sete
benchmarks e retenção 0.5. O LongBench sintético da matriz original materializa
3 casos de QA e 1 de sumarização; cada um dos outros seis arquivos locais contém
3 exemplos. Assim, o modo `quick` produz 22 casos por estratégia. A contagem deve
ser validada nos CSVs após cada execução.

### Resultados consolidados

Os quatro compressores foram executados nos cinco modelos e nos dois orçamentos:
4 casos do LongBench e 3 casos em cada um dos outros seis benchmarks, ou 22 casos
por compressor. Cada CSV consolidado contém 88 linhas. Os dez artefatos totalizam
880 execuções e estão em `benchmark_matrix_compressores/runs/`. Uma chamada do
CPC-MiniLM ao DeepSeek-R1 32B, no contexto 8192 e no caso
`meeting_summarization_0`, terminou por `timeout`; por isso, as métricas que
dependem da resposta usam 879 observações válidas. As métricas da compressão
desse caso permanecem válidas.

| Modelo | Contexto | LLMLingua-GPT2 | LLMLingua-2 | Selective Context | CPC-MiniLM |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 8192 | 0.379 | 0.492 | 0.480 | **0.531** |
| Llama 3.1 8B | 32768 | 0.367 | 0.503 | 0.487 | **0.531** |
| Gemma 4 26B | 8192 | 0.274 | 0.457 | 0.353 | **0.547** |
| Gemma 4 26B | 32768 | 0.292 | 0.467 | 0.336 | **0.573** |
| Qwen 3 30B-A3B | 8192 | 0.206 | **0.430** | 0.258 | 0.394 |
| Qwen 3 30B-A3B | 32768 | 0.190 | **0.413** | 0.282 | 0.394 |
| DeepSeek-R1 32B | 8192 | 0.321 | 0.581 | 0.576 | **0.613*** |
| DeepSeek-R1 32B | 32768 | 0.333 | 0.578 | 0.585 | **0.634** |
| GPT-OSS 20B | 8192 | 0.290 | 0.487 | 0.427 | **0.585** |
| GPT-OSS 20B | 32768 | 0.290 | 0.438 | 0.405 | **0.537** |

\* Média macro calculada após excluir a resposta encerrada por `timeout`; o
QMSum dessa combinação contém duas respostas válidas.

| Modelo | 8192 | 32768 |
|---|---:|---:|
| Llama 3.1 8B | concluído | concluído |
| Gemma 4 26B | concluído | concluído |
| Qwen 3 30B-A3B | concluído | concluído |
| DeepSeek-R1 32B | concluído com 1 `timeout` | concluído |
| GPT-OSS 20B | concluído | concluído |

Com base nas dez combinações, o conjunto inicial de opções do MCKP é
**LLMLingua-2, Selective Context e CPC-MiniLM**. LLMLingua-GPT2 permanece como
baseline externo. CPC-MiniLM lidera oito combinações, e LLMLingua-2 lidera as
duas do Qwen. Selective Context permanece no conjunto por sua complementaridade
e por apresentar a maior fidelidade semântica. CPC-MiniLM não é o checkpoint do
artigo, e a métrica de fidelidade usa o mesmo `all-MiniLM-L6-v2` empregado por
essa aproximação; por isso, a pontuação downstream tem precedência sobre a
fidelidade na seleção.
