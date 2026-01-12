import argparse
import os
import sys

from groq import Groq
from dotenv import load_dotenv

# Repo root is one level above this module directory
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

# Load .env if present (works both locally and in Docker)
load_dotenv(os.path.join(REPO_ROOT, ".env"))

# Import local modules
try:
    from prompt_compression import GroqSemanticCompressor, PerplexityCompressor
    from context_strategies import SlidingWindowStrategy, ParallelWindowStrategy
except ImportError as e:
    print(f"Erro ao importar módulos locais: {e}")
    sys.exit(1)

DEFAULT_MODEL = "llama3-70b-8192"

# Legacy default (works when you have both repos side-by-side on host)
LEGACY_REVISION_FILE = "../../00-master-thesis-overleaf-help/Chapters/002Revision/Revision.tex"

# Docker-friendly default when mounting the thesis repo as /data
DOCKER_REVISION_FILE = "/data/Chapters/002Revision/Revision.tex"


def resolve_context_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    # Interpret relative paths from this script's directory
    return os.path.normpath(os.path.join(os.path.dirname(__file__), path))


def pick_default_context_file() -> str:
    env_path = os.environ.get("CONTEXT_FILE")
    candidates = [p for p in [env_path, DOCKER_REVISION_FILE, LEGACY_REVISION_FILE] if p]

    for c in candidates:
        if os.path.exists(resolve_context_path(c)):
            return c

    # Fall back to legacy path (still useful as a hint)
    return LEGACY_REVISION_FILE


def load_context_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado: {path}")
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat demo para comparar estratégias de extensão de contexto")
    parser.add_argument(
        "--context-file",
        default=pick_default_context_file(),
        help=(
            "Caminho para o arquivo de contexto (ex: /data/Chapters/002Revision/Revision.tex). "
            "Também aceita CONTEXT_FILE no ambiente."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        help="Modelo Groq a usar (também aceita GROQ_MODEL no ambiente).",
    )
    return parser.parse_args()


def chat_loop(context_file: str, model: str) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERRO: Variável de ambiente GROQ_API_KEY não definida.")
        print("Defina GROQ_API_KEY no ambiente, no .env, ou via docker compose --env-file.")
        return

    client = Groq(api_key=api_key)

    resolved_context_file = resolve_context_path(context_file)

    print("=== CARREGANDO CONTEXTO ===")
    print(f"Arquivo: {resolved_context_file}")

    full_context = load_context_file(resolved_context_file)
    if not full_context:
        print("Abortando: Contexto vazio.")
        return
    print(f"Contexto carregado: {len(full_context)} caracteres.")

    while True:
        print("\n" + "=" * 50)
        print("Escolha a Estratégia de Contexto:")
        print("1. Nenhuma (Contexto Bruto - Cuidado com limite)")
        print("2. Compressão Semântica (Groq Llama3)")
        print("3. Compressão via Perplexidade (Local GPT2)")
        print("4. Janela Deslizante (Simples)")
        print("q. Sair")

        choice = input("\nOpção: ").strip().lower()

        if choice == "q":
            break

        processed_context = ""
        strategy_name = ""

        # --- APLICAÇÃO DA ESTRATÉGIA ---
        if choice == "1":
            strategy_name = "Bruto"
            processed_context = full_context

        elif choice == "2":
            strategy_name = "Compressão Semântica"
            print(">> Comprimindo contexto... (pode levar alguns segundos)")
            compressor = GroqSemanticCompressor(model_name=model)
            processed_context = compressor.compress(full_context, compression_ratio=0.4)
            print(
                f">> Comprimido! Tamanho original: {len(full_context)} -> Novo: {len(processed_context)}"
            )

        elif choice == "3":
            strategy_name = "Compressão Perplexidade"
            print(">> Comprimindo localmente...")
            compressor = PerplexityCompressor()
            processed_context = compressor.compress(full_context, compression_ratio=0.5)
            print(f">> Comprimido! Novo tamanho: {len(processed_context)}")

        elif choice == "4":
            strategy_name = "Janela Deslizante"
            slider = SlidingWindowStrategy(chunk_size=500, overlap=50)
            chunks = slider.process(full_context, "")
            processed_context = "\n---\n".join(chunks[:3])
            print(
                f">> Contexto segmentado em {len(chunks)} partes. Usando as 3 primeiras para o chat."
            )

        else:
            print("Opção inválida.")
            continue

        # --- INTERAÇÃO COM O USUÁRIO ---
        query = input("\nSua Pergunta sobre o texto: ")

        system_msg = (
            "Você é um assistente útil. "
            f"Responda com base no contexto fornecido usando a estratégia: {strategy_name}."
        )
        final_prompt = f"CONTEXTO:\n{processed_context}\n\nPERGUNTA DO USUÁRIO:\n{query}"

        try:
            print("\n>> Enviando para Groq...")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": final_prompt},
                ],
                model=model,
                temperature=0.5,
            )
            print("\nRESPOSTA:")
            print(response.choices[0].message.content)

        except Exception as e:
            print(f"Erro na chamada da API: {e}")


if __name__ == "__main__":
    args = parse_args()
    chat_loop(context_file=args.context_file, model=args.model)
