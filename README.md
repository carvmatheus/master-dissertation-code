# Codigo da Dissertacao de Mestrado

Este repositorio implementa e avalia estrategias nao invasivas de extensao de contexto para LLMs. A ideia central e transformar um texto maior do que a janela de contexto do modelo em uma representacao menor, mais relevante e ainda util para responder uma pergunta.

Em linguagem matematica: dado um texto longo `T`, uma pergunta `q` e uma janela de contexto `C`, quando `|T| > |C|` precisamos construir uma projecao ou transformacao

```text
pi_s(T, q) = T'
```

tal que

```text
|T'| <= |C|
```

e `T'` preserve o maximo possivel da informacao de `T` que e relevante para  `q`. Cada estrategia `s` implementada no projeto e uma forma diferente de definir essa funcao `pi_s`.

## Problema Matematico

Seja:

```text
T = (x_1, x_2, ..., x_n)
```

uma sequencia de tokens ou palavras, com tamanho `|T| = n`.

Seja:

```text
q
```

a pergunta do usuario.

Seja:

```text
C
```

a janela maxima de contexto aceita por um modelo de linguagem.

Na pratica, nem toda a janela `C` esta disponivel para o texto, porque tambem precisamos inserir instrucao de sistema, pergunta, separadores e resposta esperada. Por isso, o projeto trabalha conceitualmente com um budget efetivo:

```text
B = |C| - |prompt_sistema| - |q| - |margem|
```

O problema operacional e:

```text
encontrar T' = pi_s(T, q)

sujeito a:
  |T'| <= B

maximizando:
  U(T', T, q)
```

onde `U` e uma funcao de utilidade nao observavel diretamente. O codigo aproxima essa utilidade por sinais como:

- similaridade semantica entre chunks e pergunta;
- similaridade lexical entre chunks e pergunta;
- preservacao de tokens informativos;
- cobertura de posicoes diferentes do documento;
- continuidade local entre trechos adjacentes;
- score final nos benchmarks.

O fluxo geral do projeto e:

```text
T, q
  -> estrategia pi_s
  -> T' = pi_s(T, q)
  -> prompt final P(T', q)
  -> LLM
  -> resposta y
  -> score(y, resposta_esperada)
```

## Interface Comum

As estrategias principais seguem a interface:

```python
class ContextStrategy:
    def process(self, text: str, query: str) -> List[str]:
        ...
```

Matematicamente, `process(T, q)` implementa uma funcao:

```text
pi_s: (T, q) -> [T'_1, T'_2, ..., T'_k]
```

Algumas estrategias retornam um unico contexto final (`k = 1`). Outras retornam varios chunks, que depois sao concatenados com separadores.

## Algoritmos de Projecao de Contexto

### 1. Contexto bruto

Arquivo principal: `run_benchmarks.py`

Esta e a baseline identidade:

```text
pi_raw(T, q) = T
```

Ela so resolve o problema quando:

```text
|T| <= B
```

Se `|T| > B`, nao ha compressao nem selecao. Nos wrappers de benchmark, antes da chamada ao provedor, existe uma truncagem por limite de caracteres do modelo:

```text
tau_L(T) = T[1:L]
```

onde `L` e o limite de caracteres configurado para o modelo. Assim, na avaliacao automatica, a baseline efetiva pode ser vista como:

```text
pi_raw_eval(T, q) = tau_L(T)
```

### 2. Compressao semantica com LLM

Arquivo principal: `prompt_compression.py`

Classe: `OllamaSemanticCompressor`

Esta estrategia usa outro LLM como compressor. Dado um fator de compressao `r`, com `0 < r <= 1`, o codigo calcula um alvo aproximado em palavras:

```text
a = floor(r * words(T))
```

e pede ao modelo para produzir um texto comprimido:

```text
pi_sem(T, q; r) = G(T, a)
```

onde `G` e a funcao induzida pelo LLM compressor.

O objetivo ideal e:

```text
minimizar d_sem(T, T')
sujeito a words(T') aproximadamente <= a
```

Aqui `d_sem` representa perda semantica: entidades removidas, relacoes distorcidas, termos tecnicos apagados ou logica alterada.

Observacao importante: no codigo atual, a pergunta `q` nao entra no compressor.
Logo, a compressao e global:

```text
pi_sem(T, q; r) = pi_sem(T; r)
```

Ela tenta preservar a estrutura geral de `T`, nao apenas o que e relevante para uma pergunta especifica.

### 3. Compressao por perplexidade

Arquivo principal: `prompt_compression.py`

Classe: `PerplexityCompressor`

Esta estrategia usa um modelo causal local, por padrao GPT-2, para estimar a
informatividade de cada token. Seja:

```text
T = (x_1, x_2, ..., x_n)
```

Para cada token `x_i`, o modelo calcula uma perda:

```text
ell_i = -log p_M(x_i | x_1, ..., x_{i-1})
```

Tokens com maior `ell_i` sao menos previsiveis para o modelo proxy e, portanto,
sao tratados como mais informativos.

Dado um fator de compressao `r`, o numero de tokens preservados e:

```text
k = floor(r * n)
```

Selecionamos os indices dos `k` maiores valores de perda:

```text
K = top_k({ell_1, ell_2, ..., ell_n})
```

e reconstruimos o texto mantendo a ordem original:

```text
pi_ppl(T; r) = (x_i : i in sort(K))
```

Essa estrategia resolve o problema por selecao extrativa de tokens:

```text
|pi_ppl(T; r)| = floor(r * |T|)
```

Sua hipotese e que tokens de alta surpresa carregam mais informacao do que
tokens previsiveis. A desvantagem e que a fluencia pode degradar, porque tokens
sao removidos sem reescrita global.

### 4. Janela deslizante

Arquivo principal: `context_strategies.py`

Classe: `SlidingWindowStrategy`

Esta estrategia nao comprime semanticamente o texto. Ela segmenta `T` em blocos
com sobreposicao.

No codigo, a segmentacao e feita por palavras:

```text
T = (w_1, w_2, ..., w_n)
```

Com tamanho de chunk `m` e sobreposicao `o`, o passo e:

```text
d = m - o
```

O chunk `j` e:

```text
c_j = (w_{jd+1}, ..., w_{jd+m})
```

Assim:

```text
Pi_slide(T) = [c_0, c_1, ..., c_J]
```

A sobreposicao cria continuidade local:

```text
c_j cap c_{j+1} possui o palavras
```

No chat interativo, o codigo usa os tres primeiros chunks:

```text
T' = join(c_0, c_1, c_2)
```

Nos benchmarks, para evitar pegar apenas o inicio do documento, os chunks sao
amostrados uniformemente ao longo do texto. Para `K` chunks desejados:

```text
i_l = floor(l * (J - 1) / (K - 1)), para l = 0, ..., K - 1
```

e:

```text
T' = join(c_{i_0}, c_{i_1}, ..., c_{i_{K-1}})
```

Essa estrategia aproxima cobertura posicional. Ela nao garante que os chunks
selecionados sejam os mais relevantes para `q`.

### 5. Janela paralela

Arquivo principal: `context_strategies.py`

Classe: `ParallelWindowStrategy`

Esta estrategia particiona o texto em blocos independentes, sem sobreposicao.
Com chunk size `m`:

```text
p_j = (w_{jm+1}, ..., w_{(j+1)m})
```

Logo:

```text
Pi_parallel(T) = [p_0, p_1, ..., p_J]
```

Como nao ha overlap:

```text
p_i cap p_j = vazio, para i != j
```

No desenho conceitual, esses blocos podem ser usados em um esquema MapReduce:

```text
map(p_j, q) -> y_j
reduce(y_0, ..., y_J, q) -> y
```

No codigo atual de benchmark, os chunks tambem sao amostrados uniformemente e
concatenados antes da chamada ao modelo:

```text
T' = join(p_{i_0}, ..., p_{i_{K-1}})
```

A diferenca principal em relacao a janela deslizante e a remocao da redundancia.
O ganho e maior cobertura por token enviado; a perda e menor continuidade entre
fronteiras.

### 6. RIG com ranking Dartboard

Arquivos principais:

- `context_strategies.py`
- `rig/dartboard_processor.py`
- `rig/utils.py`

Classe: `RIGStrategy`

RIG transforma o problema de contexto em um problema de recuperacao ranqueada.
Primeiro o texto e dividido em chunks:

```text
T -> [c_1, c_2, ..., c_n]
```

Depois cada chunk recebe tres sinais.

Sinal semantico:

```text
e_i = normalize(Embed(c_i))
e_q = normalize(Embed(q))
sem_i = e_i dot e_q
```

Como os embeddings sao normalizados, o produto interno equivale a similaridade
do cosseno.

Sinal lexical:

```text
v_i = TFIDF(c_i)
v_q = TFIDF(q)
lex_i = cos(v_i, v_q)
```

Sinal de importancia:

```text
imp_i = words(c_i) / sum_j words(c_j)
```

No codigo atual, importancia e proporcional ao tamanho do chunk.

O score Dartboard e:

```text
S_i = alpha * sem_i + beta * lex_i + gamma * imp_i
```

com pesos padrao:

```text
alpha = 0.7
beta  = 0.2
gamma = 0.1
```

O algoritmo ranqueia os chunks por `S_i` e retorna os `top_k` mais relevantes:

```text
pi_rig(T, q) = join(top_k({c_i}, score=S_i))
```

Antes do retorno final, existe um filtro de diversidade. Um candidato `c_i` e
descartado se for muito parecido com algum chunk ja selecionado:

```text
max_{c_j selecionado} e_i dot e_j > delta
```

com:

```text
delta = 0.95
```

A vantagem do RIG e selecionar conteudo diretamente condicionado a `q`. A
desvantagem e que ele pode destruir continuidade global, porque seleciona poucos
chunks isolados.

### 7. Compressao seletiva como MCKP

Arquivos principais:

- `mckp/strategy.py`
- `mckp/options.py`
- `mckp/solver.py`
- `mckp/reconstructor.py`

Classe: `MCKPStrategy`

O contexto e particionado em unidades disjuntas. Para cada particao `t_j`, sao
materializadas opcoes de compressao com custo serializado `c_{j,o}` e valor:

```text
Q_{j,o} = I(t_j; q) * f_{j,o}
```

O solver escolhe exatamente uma opcao por particao:

```text
max sum_j Q_{j,o_j} - mu * sum_j d(o_j, o_{j+1})
sujeito a sum_j c_{j,o_j} <= B
```

A programacao dinamica usa o eixo de orcamento em tokens inteiros
(`budget_bucket = 1`), portanto resolve exatamente o modelo de custos declarado.
Quantizacoes maiores continuam disponiveis apenas como aproximacao conservadora.

Para os experimentos, `B` e calculado separadamente a partir de `num_ctx=8192`
ou `num_ctx=32768`, descontando o prompt completo, a saida reservada e uma margem
para diferencas entre tokenizadores. A reconstrucao nao cria marcadores de
omissao fora do custo otimizado, e o contexto final e validado antes do Ollama.

Compressores orientados a consulta incluem `q` na chave de cache. Falhas ao
materializar opcoes sao registradas nos diagnosticos `mckp_*` do benchmark.

### 8. RSAW: Relevance-Stratified Adaptive Window

Arquivos principais:

- `rsaw/strategy.py`
- `rsaw/segmenter.py`
- `rsaw/scorer.py`
- `rsaw/assembler.py`
- `rsaw/config.json`

Classe: `RSAWStrategy`

RSAW e a estrategia proposta no projeto. Ela combina:

- segmentacao por tokens reais;
- ranking Dartboard;
- estratificacao por relevancia;
- compressao seletiva;
- montagem adaptativa com budget.

O pipeline matematico e:

```text
T
  -> segmentacao token-aware
  -> score S_i para cada chunk
  -> tier(c_i)
  -> transformacao g_i(c_i)
  -> montagem T' com |T'| <= B
```

#### Etapa 1: segmentacao

O texto e tokenizado com `tiktoken`:

```text
T = (x_1, ..., x_n)
```

Com chunk size `m`, overlap `o` e passo `d = m - o`:

```text
c_j = (x_{jd+1}, ..., x_{jd+m})
```

Essa etapa difere da janela deslizante simples porque o tamanho e medido em
tokens reais, nao em palavras.

#### Etapa 2: pontuacao Dartboard

Cada chunk recebe o mesmo score hibrido do RIG:

```text
S_i = alpha * sem_i + beta * lex_i + gamma * imp_i
```

O resultado e uma lista:

```text
[(c_1, S_1), (c_2, S_2), ..., (c_n, S_n)]
```

mantida na ordem original do texto.

#### Etapa 3: estratificacao por relevancia

Dados dois limiares:

```text
theta_alto
theta_baixo
```

cada chunk e classificado em um tier:

```text
tier(c_i) = 1, se S_i >= theta_alto
tier(c_i) = 2, se theta_baixo <= S_i < theta_alto
tier(c_i) = 3, se S_i < theta_baixo
```

#### Etapa 4: transformacao por tier

Cada tier recebe uma politica diferente:

```text
g_i(c_i) = c_i,                         se tier(c_i) = 1
g_i(c_i) = pi_ppl(c_i; tier2_ratio),    se tier(c_i) = 2
g_i(c_i) = pi_sem(c_i; 0.3),            se tier(c_i) = 3
```

Ou seja:

- Tier 1 preserva chunks altamente relevantes sem compressao.
- Tier 2 aplica compressao por perplexidade.
- Tier 3 tenta uma compressao semantica curta; se a chamada falhar, o chunk e omitido.

#### Etapa 5: montagem com budget

O assembler percorre os chunks na ordem original. Seja `R` o budget restante,
inicialmente:

```text
R = B
```

Para cada chunk processado `g_i(c_i)`:

```text
se g_i(c_i) = vazio:
    omitir

se |g_i(c_i)| <= R:
    adicionar g_i(c_i) a T'
    R = R - |g_i(c_i)|

se |g_i(c_i)| > R:
    omitir
```

Quando um ou mais chunks sao omitidos entre trechos aceitos, o codigo insere:

```text
[...N chunks omitidos...]
```

Assim, a RSAW tenta resolver simultaneamente tres problemas:

```text
relevancia:       escolher conteudo util para q
continuidade:     preservar ordem original e vizinhanca
orcamento:        respeitar B por contagem de tokens
```

Observacao de implementacao: o budget e aplicado aos textos dos chunks aceitos.
Os marcadores de omissao sao metadados textuais adicionados durante a montagem e
podem acrescentar um pequeno overhead.

Configuracao atual:

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

### 9. Estrategia mock

Arquivo principal: `run_benchmarks.py`

Esta estrategia existe apenas para testes sem API.

O contexto e dividido em sentencas:

```text
T -> [s_1, s_2, ..., s_n]
```

A pergunta vira um conjunto de palavras:

```text
W(q)
```

Cada sentenca recebe score por sobreposicao lexical:

```text
score(s_i, q) = sum_{w in W(q)} 1[w aparece em s_i]
```

O retorno e:

```text
s* = argmax_i score(s_i, q)
```

Se o maior score for zero, retorna:

```text
Informacao nao encontrada no contexto.
```

### 10. MemGPT placeholder

Arquivo principal: `context_strategies.py`

Classe: `MemGPTStrategy`

Esta classe ainda nao implementa uma estrategia de memoria. Ela retorna apenas:

```text
pi_memgpt(T, q) = "(TODO: Implementar logica de Memoria Hierarquica)"
```

Portanto, matematicamente ela nao resolve o problema `|T| > |C|`; ela apenas
marca um ponto futuro para uma possivel estrategia com memoria externa:

```text
memoria externa M
estado h_t
recuperacao r_t = retrieve(M, q, h_t)
T' = assemble(r_t, q)
```

No estado atual do codigo, essa estrategia deve ser lida como placeholder, nao
como resultado experimental.

## Comparacao das Estrategias

| Estrategia      |        Forma de `pi_s(T, q)` | Usa `q`? |   Garante reducao? | Risco principal                             |
| --------------- | -----------------------------: | ---------: | -----------------: | ------------------------------------------- |
| Raw             |        identidade ou truncagem |        nao |                nao | estourar contexto ou perder fim do texto    |
| Semantica       |              reescrita por LLM |        nao |         aproximada | alucinacao ou perda silenciosa              |
| Perplexidade    | selecao de tokens por surpresa |        nao |                sim | texto menos fluente                         |
| Sliding window  |             chunks com overlap |        nao | depende da selecao | baixa relevancia para perguntas especificas |
| Parallel window |             chunks sem overlap |        nao | depende da selecao | perda de continuidade                       |
| RIG             |        top-k por score hibrido |        sim |     sim, via top-k | perda de continuidade global                |
| MCKP            | alocacao exata por particao    |        sim |    sim, por budget | custo de gerar e avaliar opcoes              |
| RSAW            |       score + tiers + montagem |        sim |    sim, por budget | custo maior e dependencia dos compressores  |
| Mock            |     sentenca com maior overlap |        sim |                sim | baseline lexical fraca                      |
| MemGPT          |                    placeholder |        nao |                nao | nao implementado                            |

## Algoritmo de Chamada ao Modelo

Os benchmarks transformam uma estrategia de contexto em uma funcao completa:

```text
F_s(T, q) = LLM(P(pi_s(T, q), q))
```

onde o prompt final tem a forma:

```text
Baseado no contexto abaixo, responda a pergunta de forma direta e concisa.

CONTEXTO:
T'

PERGUNTA:
q

RESPOSTA:
```

As chamadas sao feitas a modelos locais via API nativa do Ollama, com
`num_ctx` explicito por execucao. O limite de caracteres do contexto e
derivado da janela configurada. Antes da chamada:

```text
se len(T') > L_modelo:
    T' = T'[1:L_modelo]
```

Runs historicos via APIs externas (Groq, Gemini, Cerebras), com retry por
rate limit, estao preservados em `tests/old-api/`.

## Algoritmos de Avaliacao

Cada benchmark gera casos:

```text
z = (T, q, a)
```

onde `a` e a resposta esperada. Para uma estrategia `s`:

```text
y = F_s(T, q)
```

e o benchmark calcula:

```text
score_b(y, a, z) in [0, 1]
```

O runner agrega os resultados por media:

```text
ScoreMedio(s, b) = (1/N) * sum_i score_b(y_i, a_i, z_i)
```

Tambem mede latencia:

```text
latencia_i = tempo_fim_i - tempo_inicio_i
```

### Needle-in-a-Haystack

Arquivo: `benchmarks/needle_haystack.py`

O benchmark cria um texto com muitos distratores e insere um fato-alvo em uma
posicao `p`:

```text
T = haystack com needle(f, p)
```

A pergunta `q` pede o valor escondido `a`.

Score:

```text
score = 1, se a aparece em y
score = palavras_de_a_encontradas(y) / palavras(a), caso contrario
score = 0, se nada relevante aparece
```

Ele testa se a estrategia preserva informacao pontual em posicoes diferentes do
contexto.

### RULER

Arquivo: `benchmarks/ruler.py`

O RULER gera contextos com varios fatos artificiais distribuidos por posicao.
Cada caso pergunta por um fato especifico.

Score:

```text
score = 1.0, se a aparece exatamente em y
score = 0.5, se um componente significativo de a aparece em y
score = 0.0, caso contrario
```

O benchmark tambem estima tamanho efetivo de contexto:

```text
effective_size = max tamanho tal que media_score(tamanho) >= 0.8
```

e mede uma aproximacao do efeito "lost in the middle":

```text
lost_middle = media_score_inicio - media_score_meio
```

### LongBench sintetico

Arquivo: `benchmarks/longbench.py`

O modulo gera tarefas de QA e sumarizacao.

Para QA:

```text
score = 1, se a aparece em y
score = recall_palavras(a, y), caso contrario
```

Para sumarizacao:

```text
score = topicos_esperados_presentes(y) / topicos_esperados
```

### BABILong

Arquivo: `benchmarks/babilong.py`

O benchmark carrega exemplos de tarefas BABILong quando disponiveis e monta
casos de QA com contextos longos.

Score:

```text
score = 1, se a aparece em y
score = recall_palavras(a, y), caso contrario
```

### NarrativeQA

Arquivo: `benchmarks/narrativeqa.py`

Cada pergunta pode ter multiplas respostas de referencia:

```text
A = {a_1, a_2, ..., a_m}
```

Score:

```text
score = 1, se algum a_j aparece em y ou y aparece em a_j
score = recall_palavras(a_maior, y), caso contrario
```

### QASPER

Arquivo: `benchmarks/qasper.py`

Segue a mesma logica de multiplas respostas do NarrativeQA, aplicada a perguntas
sobre artigos cientificos.

```text
score = 1, se houver match direto com alguma resposta
score = recall_palavras(resposta_mais_longa, y), caso contrario
```

### InfiniteBench

Arquivo: `benchmarks/infinitebench.py`

Gera casos de QA longa a partir de exemplos do InfiniteBench.

Score:

```text
score = 1, se a aparece em y
score = recall_palavras(a, y), caso contrario
```

## Estrutura Principal

```text
.
|-- 01-context-extension-comparison/   # codigo do metodo
|   |-- context_strategies.py          # sliding window, parallel window, RIG wrapper
|   |-- prompt_compression.py          # OllamaSemanticCompressor, PerplexityCompressor
|   |-- run_benchmarks.py              # runner principal (Ollama local)
|   |-- benchmarks/                    # needle, RULER, LongBench, BABILong, etc.
|   |-- validators/                    # metricas de custo (L) e qualidade (Q)
|   |-- rig/                           # Dartboard ranking
|   |-- mckp/                          # particionamento, opcoes, DP e reconstrucao
|   `-- rsaw/                          # segmenter, scorer, assembler, strategy
|-- scripts/
|   |-- run_ollama_benchmark_matrix.py # matriz modelo x num_ctx via Ollama
|   |-- calibrate_from_runs.py         # calibracao a partir dos runs
|   `-- download_benchmark_datasets.py
|-- data/benchmarks/                   # datasets baixados (jsonl)
|-- tests/
|   |-- unit/                          # testes unitarios (pytest)
|   |-- old-api/                       # runs antigos via Groq/Cerebras/Gemini (maio/2026)
|   `-- ollama-local/                  # runs atuais via Ollama local
|-- spec-driven/                       # specs do RSAW + rascunho do capitulo 3
|-- ref_docs/                          # PDFs de referencia e bibliografia
|-- legacy/                            # Docker, chat interativo Groq, arquivos antigos
|-- requirements.txt
`-- README.md
```

## Como Rodar

Os benchmarks rodam contra modelos locais servidos pelo [Ollama](https://ollama.com)
(`http://localhost:11434`). Instale as dependencias:

```bash
pip install -r requirements.txt
```

Para rodar os benchmarks gerais:

```bash
python 01-context-extension-comparison/run_benchmarks.py --quick
```

Para rodar apenas uma estrategia:

```bash
python 01-context-extension-comparison/run_benchmarks.py \
  --models ollama/llama3.1:8b-instruct-q8_0 \
  --strategies raw,rig,rsaw \
  --benchmarks ruler \
  --quick
```

Para rodar somente mock, sem modelo:

```bash
python 01-context-extension-comparison/run_benchmarks.py --mock-only --quick
```

Para rodar a matriz modelo x janela de contexto (gera `tests/ollama-local/ollama_benchmark_runs/`):

```bash
python scripts/run_ollama_benchmark_matrix.py --contexts 8192,32768
```

Para avaliar o MCKP nos dois contextos sem sobrescrever a matriz das baselines:

```bash
.venv/bin/python scripts/run_ollama_benchmark_matrix.py \
  --models llama3.1:8b-instruct-q8_0,gemma4:26b-mlx,qwen3:30b-a3b,deepseek-r1:32b,gpt-oss:20b \
  --contexts 8192,32768 \
  --strategies mckp \
  --mckp-mu 0.1 \
  --mckp-budget-bucket 1 \
  --benchmarks longbench,zeroscrolls,naturalquestions,triviaqa,hotpotqa,musique,meeting_summarization \
  --output-root tests/ollama-local/benchmark_mckp_artigo
```

As ablações de `mu` devem usar outros diretórios de saída, por exemplo
`benchmark_mckp_mu0` e `benchmark_mckp_mu05`, para preservar a execução principal.

Para calibrar a partir dos runs:

```bash
python scripts/calibrate_from_runs.py
```

## Testes

Os testes unitarios cobrem compressores, estrategias de contexto, RIG, MCKP e
benchmarks.

```bash
pytest tests/unit -v
pytest 01-context-extension-comparison/mckp/tests -v
```

Os resultados dos benchmarks ficam versionados em `tests/`:

- `tests/old-api/` — runs feitos via APIs externas (Groq, Cerebras, Gemini),
  abandonados por causa de rate limits.
- `tests/ollama-local/` — runs atuais com modelos locais via Ollama
  (smoke tests, matriz de contexto e calibracao ampla).

## Leitura do Projeto

Uma forma curta de ler o trabalho e:

```text
1. Raw mostra o problema: T pode ser grande demais para C.
2. Compressao tenta reduzir T sem usar q.
3. Janelamento tenta cobrir T por partes.
4. RIG usa q para recuperar os chunks mais relevantes.
5. MCKP aloca o budget global entre opcoes locais de compressao.
6. RSAW combina relevancia, continuidade e budget por uma heuristica gulosa.
7. Os benchmarks medem quanto da informacao esperada sobrevive em T'.
```

Assim, a pergunta matematica da dissertacao pode ser resumida como:

```text
Qual estrategia pi_s gera o melhor T'
quando |T| > |C|,
mantendo |T'| <= |C|
e maximizando a capacidade do LLM de responder q?
```
