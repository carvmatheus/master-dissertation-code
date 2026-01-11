# Repositório de Código da Dissertação de Mestrado

Este repositório centraliza os códigos desenvolvidos para a dissertação de mestrado. O projeto está organizado em módulos que refletem as etapas da pesquisa, desde a comparação de técnicas de extensão de contexto até a proposta do framework socrático.

## Estrutura do Projeto

### 📂 `01-context-extension-comparison/`
Este módulo foca na análise comparativa de **Estratégias Não-Invasivas para Extensão de Contexto**. Ele implementa um chat modular que permite testar diferentes técnicas de pré-processamento de prompt antes de enviar os dados ao LLM.

**Estratégias Implementadas:**
- **Compressão Semântica (Semantic Compression):** Utiliza um LLM (via Groq API) para reescrever o contexto de forma concisa, preservando entidades e relações (Baseado em *Gilbert et al., 2023*).
- **Compressão por Perplexidade (Perplexity-based):** Utiliza um modelo proxy pequeno (ex: GPT-2) para remover tokens de baixa entropia/informação (Baseado em *LLMLingua, Jiang et al., 2023*).
- **Janela Deslizante (Sliding Window):** Segmentação do texto em blocos com sobreposição para processamento sequencial ou paralelo.

**Arquivos:**
- `chat.py`: Orquestrador central. Carrega o texto da dissertação e inicia o loop de chat.
- `prompt_compression.py`: Módulo contendo as classes de compressão.
- `context_strategies.py`: Módulo contendo lógicas de janelamento e segmentação.

---

### 📂 `02-socratic-framework/`
*(Em desenvolvimento)*
Este módulo conterá a implementação do **Framework Socrático com Contra-exemplos**. O objetivo é criar um agente que utiliza RAG e Memória Externa para gerar refutações (Elenchos) e validar a consistência das respostas do modelo.

---

## Como Executar (Módulo 01)

### Pré-requisitos
- Python 3.8+
- Conta na Groq Cloud (para a API Llama 3)
- Bibliotecas: `groq`, `torch`, `transformers`, `numpy`

### Instalação
```bash
pip install groq torch transformers numpy
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
