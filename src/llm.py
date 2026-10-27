"""Módulo de integração com LLMs e modelos de Embedding.

Oferece suporte unificado via Ollama com stubs para NVIDIA e Hugging Face.
Inclui tratamento amigável de erros de conexão e modelos não encontrados.
"""

import logging
from typing import Any, Dict, List, Optional
import ollama
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cliente Ollama com host customizado
_ollama_client = ollama.Client(host=config.OLLAMA_HOST)


def _handle_ollama_error(e: Exception, model_name: str) -> None:
    """Gera mensagens de erro amigáveis para falhas comuns com o Ollama."""
    err_msg = str(e).lower()
    if "connection refused" in err_msg or "connecterror" in err_msg or "failed to connect" in err_msg:
        raise RuntimeError(
            f"Não foi possível conectar ao Ollama em '{config.OLLAMA_HOST}'. "
            "Certifique-se de que o Ollama está instalado e rodando. "
            "Abra o app do Ollama ou execute 'ollama serve' no terminal."
        ) from e
    if "not found" in err_msg or "try pulling it first" in err_msg:
        raise RuntimeError(
            f"O modelo '{model_name}' não foi encontrado no Ollama. "
            f"Baixe-o executando no terminal: ollama pull {model_name}"
        ) from e
    raise RuntimeError(f"Erro ao comunicar com Ollama ({model_name}): {e}") from e


def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    json_mode: bool = False,
    model: Optional[str] = None,
) -> str:
    """Envia mensagens de chat para o LLM e retorna o texto gerado.

    Args:
        messages: Lista de dicionários no formato [{"role": "user"|"system"|"assistant", "content": "..."}].
        temperature: Temperatura de amostragem (0.0 a 2.0).
        json_mode: Se True, força o LLM a retornar JSON válido.
        model: Nome do modelo (opcional, padrão definido em config.LLM_MODEL).

    Returns:
        String gerada pelo modelo.
    """
    model_name = model or config.LLM_MODEL
    provider = config.LLM_PROVIDER

    if provider == "ollama":
        try:
            format_opt = "json" if json_mode else ""
            response = _ollama_client.chat(
                model=model_name,
                messages=messages,
                options={"temperature": float(temperature)},
                format=format_opt if format_opt else None,
            )
            return response["message"]["content"]
        except Exception as e:
            _handle_ollama_error(e, model_name)

    elif provider == "nvidia":
        # TODO: Implementar integração com a API da NVIDIA (OpenAI-compatible)
        # Exemplo:
        # from openai import OpenAI
        # client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=config.NVIDIA_API_KEY)
        # response = client.chat.completions.create(model=model_name, messages=messages, temperature=temperature)
        # return response.choices[0].message.content
        raise NotImplementedError("Provedor 'nvidia' ainda não implementado. Use LLM_PROVIDER=ollama.")

    elif provider == "huggingface":
        # TODO: Implementar integração com Hugging Face Inference API / Router
        # Exemplo:
        # headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        # requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=...)
        raise NotImplementedError("Provedor 'huggingface' ainda não implementado. Use LLM_PROVIDER=ollama.")

    else:
        raise ValueError(f"Provedor desconhecido: '{provider}'. Opções válidas: ollama, nvidia, huggingface.")


def embed(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 32,
) -> List[List[float]]:
    """Gera vetores de embeddings para uma lista de textos em lotes (batches).

    Args:
        texts: Lista de strings a serem convertidas em vetores.
        model: Nome do modelo de embedding (opcional, padrão config.EMBED_MODEL_A).
        batch_size: Quantidade de textos processados por requisição.

    Returns:
        Lista de vetores (cada um sendo list[float]).
    """
    if not texts:
        return []

    model_name = model or config.EMBED_MODEL_A
    provider = config.LLM_PROVIDER
    all_embeddings: List[List[float]] = []

    if provider == "ollama":
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # A API embeddings/embed do Ollama suporta 'input' como string ou lista de strings
                # Dependendo da versão do pacote python ollama, usamos .embed() ou .embeddings()
                try:
                    res = _ollama_client.embed(model=model_name, input=batch)
                    embeddings = res.get("embeddings", [])
                    all_embeddings.extend(embeddings)
                except AttributeError:
                    # Fallback para versões mais antigas da lib ollama com client.embeddings
                    for text in batch:
                        single_res = _ollama_client.embeddings(model=model_name, prompt=text)
                        all_embeddings.append(single_res["embedding"])
            return all_embeddings
        except Exception as e:
            _handle_ollama_error(e, model_name)

    elif provider == "nvidia":
        # TODO: Implementar geração de embeddings via NVIDIA API
        raise NotImplementedError("Provedor 'nvidia' para embeddings ainda não implementado.")

    elif provider == "huggingface":
        # TODO: Implementar geração de embeddings via Hugging Face API
        raise NotImplementedError("Provedor 'huggingface' para embeddings ainda não implementado.")

    else:
        raise ValueError(f"Provedor desconhecido: '{provider}'. Opções válidas: ollama, nvidia, huggingface.")
