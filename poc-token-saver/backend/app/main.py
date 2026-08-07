import os
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .compression import compress_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TokenSaver API",
    description="PoC de compressão de contexto para LLMs baseada na dissertação.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja ao domínio do frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompressRequest(BaseModel):
    document: str = Field(..., description="Texto completo do documento")
    query: str = Field(..., description="Pergunta do usuário sobre o documento")
    strategy: str = Field(default="MCKP", description="MCKP ou SLIDING_WINDOW")
    budget: int = Field(default=4000, ge=100, le=128000, description="Orçamento de tokens")


class CompressResponse(BaseModel):
    compressed_context: str
    original_tokens: int
    final_tokens: int
    savings_percentage: float
    execution_time_ms: float
    strategy: str
    budget: int


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "token-saver"}


@app.post("/api/v1/compress", response_model=CompressResponse)
def compress(payload: CompressRequest):
    """
    Comprime um documento para caber no orçamento de tokens usando MCKP
    ou Sliding Window.
    """
    try:
        result = compress_document(
            document=payload.document,
            query=payload.query,
            strategy=payload.strategy,
            budget=payload.budget,
        )
        return CompressResponse(**result)
    except ValueError as e:
        logger.warning("Erro de validação: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Falha na compressão")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")


@app.post("/api/v1/compress/file", response_model=CompressResponse)
def compress_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    strategy: str = Form(default="MCKP"),
    budget: int = Form(default=4000),
):
    """
    Endpoint alternativo que aceita upload de arquivo de texto (.txt, .md).
    PDF pode ser adicionado posteriormente com PyMuPDF/pdfplumber.
    """
    try:
        content = file.file.read().decode("utf-8", errors="ignore")
        result = compress_document(
            document=content,
            query=query,
            strategy=strategy,
            budget=budget,
        )
        return CompressResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Falha na compressão de arquivo")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")


class AskRequest(BaseModel):
    compressed_context: str
    query: str
    model: str = "gpt-4o-mini"


@app.post("/api/v1/ask")
def ask(payload: AskRequest):
    """
    Envia a query + contexto comprimido para a OpenAI.
    Requer OPENAI_API_KEY no ambiente.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY não configurada no backend.",
        )

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        system_prompt = (
            "Você é um assistente especializado em responder com base "
            "estritamente no contexto fornecido."
        )
        user_prompt = f"""CONTEXTO:
{payload.compressed_context}

PERGUNTA: {payload.query}

Responda de forma direta e concisa."""

        response = client.chat.completions.create(
            model=payload.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            "answer": response.choices[0].message.content,
            "model": payload.model,
            "usage": response.usage.model_dump() if response.usage else None,
        }
    except Exception as e:
        logger.exception("Erro na chamada OpenAI")
        raise HTTPException(status_code=502, detail=f"Erro OpenAI: {e}")
