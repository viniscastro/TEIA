"""Módulo de contagem, inspeção e manipulação de tokens.

Utiliza o tokenizer real cl100k_base (do tiktoken), que reflete a tokenização
de modelos modernos baseados em BPE (Byte Pair Encoding).

DECISÃO DE DESIGN - CHUNKING & LIMITES:
----------------------------------------
Por que escolhemos 1 documento = 1 jogo (textos curtos) em vez de chunks menores?
1. Coesão Semântica: Um jogo na Steam é uma unidade conceitual atômica. Se quebrássemos
   a descrição curta em pedaços menores (chunks de 50 tokens), poderíamos separar o Nome e
   Gênero das principais mecânicas (ex: 'cooperativo', 'roguelike'), gerando vetores incompletos.
2. Tamanho dos Textos: A média do doc_text fica em torno de 60 a 120 tokens, com o P95
   ficando bem abaixo de 250 tokens. Textos dessa escala cabem com folga dentro da janela
   de contexto dos modelos de embedding utilizados (nomic-embed-text suporta até 2048/8192 tokens
   e bge-m3 até 8192 tokens).
3. Relação 1:1 no ChromaDB: Permite mapeamento direto entre 1 embedding no banco vetorial
   e 1 registro na loja Steam com seus metadados (preço, plataformas, tags).

Limite máximo adotado: 512 tokens (margem de segurança ampla que garante integridade sem truncamento
indesejado para mais de 99.9% dos jogos).
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import tiktoken

# Tokenizer padrão compatível com modelos modernos
_ENCODING_NAME = "cl100k_base"
_encoder = tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Conta a quantidade exata de tokens de um texto usando o tokenizer cl100k_base.

    Args:
        text: Texto a ser avaliado.

    Returns:
        Número inteiro de tokens.
    """
    if not text:
        return 0
    return len(_encoder.encode(text))


def tokenize_preview(text: str) -> List[Dict[str, Any]]:
    """Gera visualização detalhada de tokens com representação em texto e ID numérico.

    Útil para exibir na interface como o BPE decompõe palavras (especialmente em português).

    Args:
        text: Texto a ser decomposto.

    Returns:
        Lista de dicionários: [{"token": "...", "id": 12345}, ...]
    """
    if not text:
        return []

    token_ids = _encoder.encode(text)
    preview = []
    for token_id in token_ids:
        raw_bytes = _encoder.decode_single_token_bytes(token_id)
        # Decodifica bytes para string tratando caracteres incompletos
        token_str = raw_bytes.decode("utf-8", errors="replace")
        preview.append({
            "token": token_str,
            "id": token_id,
        })
    return preview


def truncate_to_tokens(text: str, max_tokens: int = 512) -> str:
    """Trunca o texto para garantir que não ultrapasse max_tokens.

    Args:
        text: Texto de entrada.
        max_tokens: Quantidade máxima de tokens permitida (padrão 512).

    Returns:
        Texto truncado caso exceda o limite, ou o texto original.
    """
    if not text:
        return ""

    token_ids = _encoder.encode(text)
    if len(token_ids) <= max_tokens:
        return text

    truncated_ids = token_ids[:max_tokens]
    return _encoder.decode(truncated_ids)


def stats_doc_text(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula estatísticas de contagem de tokens na coluna doc_text do DataFrame.

    Args:
        df: DataFrame contendo a coluna 'doc_text'.

    Returns:
        Dicionário com média, mediana e p95 de tokens.
    """
    if "doc_text" not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'doc_text'.")

    # Contagem de tokens para cada linha
    token_counts = np.array([count_tokens(text) for text in df["doc_text"]])

    return {
        "count": float(len(token_counts)),
        "mean": float(np.mean(token_counts)),
        "median": float(np.median(token_counts)),
        "p95": float(np.percentile(token_counts, 95)),
        "min": float(np.min(token_counts)),
        "max": float(np.max(token_counts)),
    }


if __name__ == "__main__":
    from src import config

    print("=" * 65)
    print("        TESTE DIDÁTICO DE TOKENIZAÇÃO (Fase 3 - BPE)        ")
    print("=" * 65)

    # 1. Demonstração de tokenização em português
    frase_pt = "jogo cooperativo relaxante para jogar no fim de semana"
    tokens_info = tokenize_preview(frase_pt)
    print(f"Frase original: '{frase_pt}'")
    print(f"Total de tokens ({_ENCODING_NAME}): {len(tokens_info)}\n")
    print("Tokens individuais gerados (repare na divisão subpalavra):")
    for t in tokens_info:
        repr_str = repr(t['token'])
        print(f"  ID {t['id']:<7} -> {repr_str:<15}")

    # 2. Estatísticas do dataset processado
    parquet_path = config.PROCESSED_DATA_DIR / "games.parquet"
    if parquet_path.exists():
        print("\n" + "-" * 65)
        print("Estatísticas de Tokens na coluna 'doc_text' (games.parquet):")
        df_games = pd.read_parquet(parquet_path)
        stats = stats_doc_text(df_games)
        print(f"  - Total de documentos: {int(stats['count'])}")
        print(f"  - Média de tokens:     {stats['mean']:.1f}")
        print(f"  - Mediana de tokens:   {stats['median']:.1f}")
        print(f"  - P95 (95% dos jogos): {stats['p95']:.1f} tokens")
        print(f"  - Mínimo de tokens:    {int(stats['min'])}")
        print(f"  - Máximo de tokens:    {int(stats['max'])}")
        print("-" * 65)
    print("=" * 65)
