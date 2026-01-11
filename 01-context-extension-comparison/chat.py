import os
import sys
from groq import Groq
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Importa nossos módulos locais
try:
    from prompt_compression import GroqSemanticCompressor, PerplexityCompressor
    from context_strategies import SlidingWindowStrategy, ParallelWindowStrategy
except ImportError as e:
    print(f"Erro ao importar módulos locais: {e}")
    sys.exit(1)

# Configuração
REVISION_FILE = "../../00-master-thesis-overleaf-help/Chapters/002Revision/Revision.tex"
DEFAULT_MODEL = "llama3-70b-8192"

def load_context_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado: {path}")
        return ""

def chat_loop():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERRO: Variável de ambiente GROQ_API_KEY não definida.")
        print("Execute: export GROQ_API_KEY='sua_chave'")
        return

    client = Groq(api_key=api_key)
    
    print("=== CARREGANDO CONTEXTO (Revision.tex) ===")
    full_context = load_context_file(REVISION_FILE)
    if not full_context:
        print("Abortando: Contexto vazio.")
        return
    print(f"Contexto carregado: {len(full_context)} caracteres.")
    
    while True:
        print("\n" + "="*50)
        print("Escolha a Estratégia de Contexto:")
        print("1. Nenhuma (Contexto Bruto - Cuidado com limite)")
        print("2. Compressão Semântica (Groq Llama3)")
        print("3. Compressão via Perplexidade (Local GPT2)")
        print("4. Janela Deslizante (Simples)")
        print("q. Sair")
        
        choice = input("\nOpção: ").strip().lower()
        
        if choice == 'q':
            break
            
        processed_context = ""
        strategy_name = ""

        # --- APLICAÇÃO DA ESTRATÉGIA ---
        if choice == '1':
            strategy_name = "Bruto"
            processed_context = full_context
            
        elif choice == '2':
            strategy_name = "Compressão Semântica"
            print(">> Comprimindo contexto... (pode levar alguns segundos)")
            compressor = GroqSemanticCompressor(model_name=DEFAULT_MODEL)
            # Comprime para 40% do tamanho original
            processed_context = compressor.compress(full_context, compression_ratio=0.4)
            print(f">> Comprimido! Tamanho original: {len(full_context)} -> Novo: {len(processed_context)}")
            
        elif choice == '3':
            strategy_name = "Compressão Perplexidade"
            print(">> Comprimindo localmente...")
            compressor = PerplexityCompressor()
            processed_context = compressor.compress(full_context, compression_ratio=0.5)
            print(f">> Comprimido! Novo tamanho: {len(processed_context)}")

        elif choice == '4':
            strategy_name = "Janela Deslizante"
            # Na janela deslizante, a lógica de chat é um pouco diferente (precisa iterar sobre chunks)
            # Para simplificar este chat demo, pegaremos apenas os primeiros 2 chunks
            slider = SlidingWindowStrategy(chunk_size=500, overlap=50)
            chunks = slider.process(full_context, "")
            processed_context = "\n---\n".join(chunks[:3]) # Pega os 3 primeiros chunks
            print(f">> Contexto segmentado em {len(chunks)} partes. Usando as 3 primeiras para o chat.")
            
        else:
            print("Opção inválida.")
            continue

        # --- INTERAÇÃO COM O USUÁRIO ---
        query = input("\nSua Pergunta sobre o texto: ")
        
        # Montagem do Prompt Final
        system_msg = f"Você é um assistente útil. Responda com base no contexto fornecido usando a estratégia: {strategy_name}."
        final_prompt = f"CONTEXTO:\n{processed_context}\n\nPERGUNTA DO USUÁRIO:\n{query}"
        
        try:
            print("\n>> Enviando para Groq...")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": final_prompt}
                ],
                model=DEFAULT_MODEL,
                temperature=0.5
            )
            print("\nRESPOSTA:")
            print(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Erro na chamada da API: {e}")

if __name__ == "__main__":
    chat_loop()
