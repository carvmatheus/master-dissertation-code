# TokenSaver — PoC

PoC de middleware + showcase web para compressão de contexto de LLMs, baseada nas estratégias **MCKP** e **Sliding Window** da dissertação.

## Estrutura

```
poc-token-saver/
├── backend/          # FastAPI (Python)
│   ├── app/
│   │   ├── main.py           # endpoints /compress, /ask, /health
│   │   └── compression.py    # orquestração das estratégias
│   ├── requirements.txt
│   └── .env.example
└── frontend/         # Next.js + Tailwind CSS
    └── app/page.tsx  # UI com upload, query e split-screen
```

## Pré-requisitos

- Python 3.12+
- Node.js 18+
- Conta OpenAI (opcional, apenas para o botão "Perguntar ao LLM")

## Como rodar

### 1. Backend

```bash
cd poc-token-saver/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# (opcional) Configure sua chave da OpenAI
cp .env.example .env
# edite .env e coloque OPENAI_API_KEY=sk-...

.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

O backend estará em: http://localhost:8000

### 2. Frontend

```bash
cd poc-token-saver/frontend
npm install
npm run dev
```

O frontend estará em: http://localhost:3000

### 3. Script único

```bash
cd poc-token-saver
./start.sh
```

## Endpoints da API

### `POST /api/v1/compress`

Comprime um documento dada uma query.

```json
{
  "document": "texto longo...",
  "query": "pergunta",
  "strategy": "MCKP",
  "budget": 4000
}
```

Resposta:

```json
{
  "compressed_context": "...",
  "original_tokens": 120000,
  "final_tokens": 3800,
  "savings_percentage": 96.8,
  "execution_time_ms": 320,
  "strategy": "MCKP",
  "budget": 4000
}
```

### `POST /api/v1/ask`

Envia o contexto comprimido + query para a OpenAI.

```json
{
  "compressed_context": "...",
  "query": "pergunta",
  "model": "gpt-4o-mini"
}
```

## Observações

- A primeira chamada ao MCKP pode demorar alguns segundos por causa do carregamento dos modelos de embeddings.
- Apenas arquivos `.txt` e `.md` são aceitos no upload por enquanto. PDF pode ser adicionado com `PyMuPDF`/`pdfplumber`.
- A estratégia `SLIDING_WINDOW` está disponível como alternativa mais simples.
