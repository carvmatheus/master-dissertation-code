import os
import sys
from typing import List, Tuple

# Tenta importar Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PromptCompressor:
    """
    Classe Base para estratégias de compressão.
    """
    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        raise NotImplementedError

class GroqSemanticCompressor(PromptCompressor):
    """
    Implementa Compressão Semântica usando a API da Groq (Llama 3, Mixtral, etc).
    O LLM reescreve o texto mantendo as entidades e relações.
    """
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: str = None):
        if not GROQ_AVAILABLE:
            raise ImportError("Biblioteca 'groq' necessária. Rode: pip install groq")
        
        # Tenta carregar do .env se não estiver no ambiente
        if not os.environ.get("GROQ_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
            
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API Key da Groq não encontrada. Defina GROQ_API_KEY no arquivo .env ou no sistema.")
            
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not text:
            return ""

        target_words = int(len(text.split()) * compression_ratio)
        if target_words < 10: target_words = 10 
        
        system_prompt = "You are an expert editor designed to compress texts for LLM context windows."
        
        user_message = (
            f"Compress the following text to approximately {target_words} words ({int(compression_ratio*100)}% of original). "
            "Maintain ALL key entities, relationships, technical terms, and the core logic. "
            "Remove only fluff, redundant adjectives, and repetitive examples. "
            "Output ONLY the compressed text.\n\n"
            f"TEXT TO COMPRESS:\n{text}"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model_name,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[Erro na API Groq: {str(e)}]"

class PerplexityCompressor(PromptCompressor):
    """
    Compressão baseada em Entropia/Perplexidade (Estilo LLMLingua).
    Remove tokens que o modelo (ex: GPT-2) consegue prever facilmente.
    """
    def __init__(self, model_name: str = "gpt2", device: str = None):
        if not TORCH_AVAILABLE:
            print("Aviso: Torch/Transformers não instalados. PerplexityCompressor não funcionará.")
            return
            
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Erro carregando modelo local: {e}")

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not TORCH_AVAILABLE: return text
        if not text: return ""

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids[0]
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[0]

        shift_logits = logits[:-1, :]
        shift_labels = input_ids[1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits, shift_labels)
        token_losses = torch.cat([torch.tensor([token_losses.mean()]).to(self.device), token_losses])

        num_keep = int(len(input_ids) * compression_ratio)
        top_k_indices = torch.topk(token_losses, k=num_keep).indices
        keep_indices = torch.sort(top_k_indices).values
        
        compressed_ids = input_ids[keep_indices]
        return self.tokenizer.decode(compressed_ids, skip_special_tokens=True)
