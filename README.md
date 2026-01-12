# Repositório de Código da Dissertação de Mestrado

Este repositório centraliza os códigos desenvolvidos para a dissertação de mestrado. O projeto está organizado em módulos que refletem as etapas da pesquisa, desde a comparação de técnicas de extensão de contexto até a proposta do framework socrático.

## Estrutura do Projeto

### 📂 `01-context-extension-comparison/`
Este módulo foca na análise comparativa de **Estratégias Não-Invasivas para Extensão de Contexto**. Ele implementa um chat modular que permite testar diferentes técnicas de pré-processamento de prompt antes de enviar os dados ao LLM.

**Estratégias Implementadas:**
- **Compressão Semântica (Semantic Compression):** Utiliza um LLM (via Groq API) para reescrever o contexto de forma concisa, preservando entidades e relações (Baseado em *Gilbert et al., 2023*).
- **Compressão por Perplexidade (Perplexity-based):** Utiliza um modelo proxy pequeno (ex: GPT-2) para remover tokens de baixa entropia/informação (Baseado em *LLMLingua, Jiang et al., 2023*).
- **Janela Deslizante (Sliding Window):** Segmentação do texto em blocos com sobreposição para processamento sequencial ou paralelo.
- **RIG - Dartboard Ranking:** Combina três sinais para ranquear chunks: similaridade semântica (embeddings), similaridade lexical (TF-IDF) e importância do documento. Baseado na estrutura Dartboard.

**Arquivos:**
- `chat.py`: Orquestrador central. Carrega o texto da dissertação e inicia o loop de chat.
- `prompt_compression.py`: Módulo contendo as classes de compressão.
- `context_strategies.py`: Módulo contendo lógicas de janelamento e segmentação.
- `rig/`: Submódulo com o processador Dartboard para RAG híbrido.

---

### 📂 `02-socratic-framework/`
*(Em desenvolvimento)*
Este módulo conterá a implementação do **Framework Socrático com Contra-exemplos**. O objetivo é criar um agente que utiliza RAG e Memória Externa para gerar refutações (Elenchos) e validar a consistência das respostas do modelo.

---


## Como Executar com Docker (recomendado)

### Pré-requisitos
- Docker + Docker Compose
- Chave da Groq (env `GROQ_API_KEY`)

### Execução (chat interativo)
A partir da pasta `master-dissertation-code/`:

Garanta que o arquivo `.env` contenha (sem espaços/aspas):

```bash
GROQ_API_KEY=sua_chave_aqui
```

```bash
docker compose run --rm chat
```

- O texto da dissertação é montado como volume em `/data` (ver `docker-compose.yml`).
- O cache do HuggingFace fica em `./.cache/` para evitar baixar o GPT-2 toda vez.

### Executar apontando para outro arquivo de contexto

```bash
docker compose run --rm chat \
  python 01-context-extension-comparison/chat.py \
  --context-file /data/Chapters/002Revision/Revision.tex
```

### Rodar Testes

```bash
docker compose run --rm test
```

Os testes usam mocks e **não precisam de API key real** nem de GPU.

---

## Como Executar (Módulo 01)

### Pré-requisitos
- Python 3.8+
- Conta na Groq Cloud (para a API Llama 3)
- Bibliotecas: `groq`, `torch`, `transformers`, `numpy`

### Instalação
```bash
pip install -r requirements.txt
```

### Execução
1. Defina sua chave da API Groq:
   ```bash
   export GROQ_API_KEY="sua_chave_aqui"
   ```

2. Navegue até o módulo de comparação:
   ```bash
   cd master-dissertation-code/01-context-extension-comparison
   ```

3. Execute o chat:
   ```bash
   python chat.py
   ```

O sistema carregará automaticamente o texto base (Capítulo de Revisão da Dissertação) e oferecerá um menu para escolher a estratégia de contexto desejada.
