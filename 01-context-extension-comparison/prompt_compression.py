import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

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


class OllamaSemanticCompressor(PromptCompressor):
    """
    Implementa Compressão Semântica usando um modelo local via Ollama.
    O LLM reescreve o texto mantendo as entidades e relações.
    """
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q8_0", base_url: str = None):
        self.model_name = model_name.removeprefix("ollama/")
        base = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = base.removesuffix("/v1").rstrip("/")

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not text:
            return ""

        target_words = int(len(text.split()) * compression_ratio)
        if target_words < 10:
            target_words = 10

        system_prompt = "You are an expert editor designed to compress texts for LLM context windows."

        user_message = (
            f"Compress the following text to approximately {target_words} words ({int(compression_ratio*100)}% of original). "
            "Maintain ALL key entities, relationships, technical terms, and the core logic. "
            "Remove only fluff, redundant adjectives, and repetitive examples. "
            "Output ONLY the compressed text.\n\n"
            f"TEXT TO COMPRESS:\n{text}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
                "temperature": 0.2,
                "seed": 42,
            },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            logger.error(f"[{self.model_name}] Erro na API Ollama: {type(e).__name__}: {e}")
            return f"[Erro na API Ollama: {e}]"


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


class LLMLinguaCompressor(PromptCompressor):
    """LLMLingua v1 (Jiang et al., 2023).

    Remove tokens de baixa perplexidade usando um LM causal como scorer, num
    esquema coarse-to-fine. Usado como baseline de compressão da literatura.
    O scorer padrão é o GPT-2, escolhido por ser leve e rápido em CPU.
    """

    def __init__(self, model_name: str = "gpt2", device_map: str = "cpu"):
        from llmlingua import PromptCompressor as _LLMLingua

        self._impl = _LLMLingua(
            model_name=model_name,
            use_llmlingua2=False,
            device_map=device_map,
        )

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not text:
            return ""
        try:
            out = self._impl.compress_prompt(text, rate=float(compression_ratio))
            return out.get("compressed_prompt", text) if isinstance(out, dict) else out
        except Exception as e:
            raise RuntimeError(
                f"[llmlingua] erro na compressão: {type(e).__name__}: {e}"
            ) from e


class LLMLingua2Compressor(PromptCompressor):
    """LLMLingua-2 (Pan et al., 2024).

    Compressão por classificação de tokens com um encoder bidirecional
    (XLM-RoBERTa) destilado do GPT-4. Rápido e agnóstico à tarefa.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "cpu",
    ):
        from llmlingua import PromptCompressor as _LLMLingua

        self._impl = _LLMLingua(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not text:
            return ""
        try:
            out = self._impl.compress_prompt(text, rate=float(compression_ratio))
            return out.get("compressed_prompt", text) if isinstance(out, dict) else out
        except Exception as e:
            raise RuntimeError(
                f"[llmlingua2] erro na compressão: {type(e).__name__}: {e}"
            ) from e


class SelectiveContextCompressor(PromptCompressor):
    """Selective Context (Li et al., 2023).

    Remove unidades lexicais de baixa auto-informação, medida por um LM causal
    (GPT-2). O parâmetro reduce_ratio do pacote é a fração removida, então é
    derivado da taxa de retenção alvo.
    """

    def __init__(self, model_type: str = "gpt2", lang: str = "en"):
        from selective_context import SelectiveContext

        self._impl = SelectiveContext(model_type=model_type, lang=lang)

    def compress(self, text: str, compression_ratio: float = 0.5) -> str:
        if not text:
            return ""
        try:
            reduce_ratio = max(0.0, min(1.0, 1.0 - float(compression_ratio)))
            res = self._impl(text, reduce_ratio=reduce_ratio)
            ctx = res[0] if isinstance(res, tuple) else res
            return ctx or text
        except Exception as e:
            raise RuntimeError(
                f"[selective_context] erro na compressão: {type(e).__name__}: {e}"
            ) from e
