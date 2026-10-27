"""Script rápido de teste (smoke test) para validar a configuração do LLM e dos Embeddings."""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para permitir importações de src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.llm import chat, embed


def main() -> None:
    print("=" * 60)
    print("  GameFinder RAG - Smoke Test (Fase 1)")
    print("=" * 60)
    print(f"Provedor configurado: {config.LLM_PROVIDER}")
    print(f"LLM Model:            {config.LLM_MODEL}")
    print(f"Embedding Model A:    {config.EMBED_MODEL_A}")
    print(f"Embedding Model B:    {config.EMBED_MODEL_B}")
    print(f"Ollama Host:          {config.OLLAMA_HOST}")
    print("-" * 60)

    # 1. Teste do Chat
    print("\n[1/2] Testando chamada ao LLM (chat)...")
    messages = [
        {"role": "user", "content": "Diga em uma única frase: o que é um jogo indie?"}
    ]
    try:
        response = chat(messages=messages, temperature=0.7)
        print("Resposta do LLM:")
        print(f"   -> {response.strip()}")
    except Exception as e:
        print(f"[ERRO no Chat]: {e}")

    # 2. Teste do Embedding
    print("\n[2/2] Testando geração de embedding...")
    test_texts = ["Hollow Knight é um metroidvania desafiador e imersivo."]
    try:
        vectors = embed(texts=test_texts, model=config.EMBED_MODEL_A)
        if vectors and len(vectors) > 0:
            dim = len(vectors[0])
            print(f"Embedding gerado com sucesso!")
            print(f"   -> Modelo: {config.EMBED_MODEL_A}")
            print(f"   -> Quantidade de vetores: {len(vectors)}")
            print(f"   -> Dimensão do vetor: {dim}")
        else:
            print("[AVISO]: Nenhum vetor retornado.")
    except Exception as e:
        print(f"[ERRO no Embedding]: {e}")

    print("\n" + "=" * 60)
    print("Smoke test concluído.")
    print("=" * 60)


if __name__ == "__main__":
    main()
