"""Módulo de indexação vetorial e gerenciamento de coleções no ChromaDB.

Lê o dataset limpo (games.parquet), gera embeddings do doc_text em lotes
usando EMBED_MODEL_A (nomic-embed-text) e EMBED_MODEL_B (bge-m3), e armazena em coleções
persistentes separadas no ChromaDB (data/chroma/).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.llm import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Nomes padronizados das coleções
COLLECTION_NOMIC = "games_nomic"
COLLECTION_BGE = "games_bge"


def get_chroma_client() -> chromadb.PersistentClient:
    """Retorna cliente persistente do ChromaDB configurado para data/chroma/."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _format_metadata(row: pd.Series) -> Dict[str, Any]:
    """Prepara os metadados para o ChromaDB (tipos primitivos permitidos: str, int, float, bool)."""
    tags_val = row.get("tags", [])
    if isinstance(tags_val, (list, np.ndarray)):
        tags_str = ", ".join(str(t) for t in tags_val)
    elif isinstance(tags_val, str):
        tags_str = tags_val
    else:
        tags_str = ""

    release_year = row.get("release_year")
    if pd.isna(release_year) or release_year is None:
        release_year = 0
    else:
        release_year = int(release_year)

    return {
        "name": str(row.get("name", "")),
        "price": float(row.get("price", 0.0)),
        "windows": bool(row.get("windows", True)),
        "mac": bool(row.get("mac", False)),
        "linux": bool(row.get("linux", False)),
        "multiplayer": bool(row.get("multiplayer", False)),
        "coop": bool(row.get("coop", False)),
        "release_year": release_year,
        "total_reviews": int(row.get("total_reviews", 0)),
        "tags": tags_str,
        "short_description": str(row.get("short_description", "")),
    }


def index_collection(
    df: pd.DataFrame,
    collection_name: str,
    embed_model: str,
    batch_size: int = 64,
    force: bool = False,
) -> None:
    """Indexa uma coleção no ChromaDB com barra de progresso.

    Args:
        df: DataFrame com jogos limpos.
        collection_name: Nome da coleção no ChromaDB (ex: games_nomic).
        embed_model: Modelo de embedding a ser usado (ex: nomic-embed-text).
        batch_size: Tamanho do lote para geração de embeddings e inserção.
        force: Se True, recria a coleção do zero mesmo se já existir.
    """
    client = get_chroma_client()
    total_docs = len(df)

    # Verifica se a coleção já existe
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        col = client.get_collection(collection_name)
        current_count = col.count()
        if current_count == total_docs and not force:
            logger.info(
                f"Coleção '{collection_name}' já existe com {current_count} itens. "
                "Pulando indexação (use --force para refazer)."
            )
            return
        if force:
            logger.info(f"Removendo coleção anterior '{collection_name}' para recriação (--force)...")
            client.delete_collection(collection_name)

    logger.info(f"Criando coleção '{collection_name}' usando modelo '{embed_model}'...")
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embed_model": embed_model},
    )

    docs = df["doc_text"].tolist()
    ids = [str(app_id) for app_id in df["app_id"]]
    metadatas = [_format_metadata(row) for _, row in df.iterrows()]

    logger.info(f"Iniciando indexação de {total_docs} jogos em lotes de {batch_size}...")

    with tqdm(total=total_docs, desc=f"Indexando [{collection_name}]", unit="jogos") as pbar:
        for i in range(0, total_docs, batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            # Gera embeddings do lote
            batch_embeddings = embed(texts=batch_docs, model=embed_model, batch_size=batch_size)

            # Insere no ChromaDB
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_meta,
            )
            pbar.update(len(batch_docs))

    logger.info(f"Coleção '{collection_name}' indexada com sucesso! Total: {collection.count()} registros.")


def search_raw(
    query: str,
    collection_name: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Realiza busca vetorial bruta sem filtros adicionais para teste e comparação.

    Args:
        query: Texto de busca (ex: 'jogo de fazenda relaxante').
        collection_name: Nome da coleção (games_nomic ou games_bge).
        k: Número de resultados a retornar.

    Returns:
        Lista com os top-k jogos encontrados contendo id, nome, distância e metadados.
    """
    client = get_chroma_client()
    col = client.get_collection(collection_name)
    embed_model = col.metadata.get("embed_model", config.EMBED_MODEL_A)

    query_vec = embed(texts=[query], model=embed_model)[0]

    results = col.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    print("\n" + "=" * 70)
    print(f"  BUSCA VETORIAL BRUTA | Coleção: {collection_name} | k={k}")
    print(f"  Query: '{query}'")
    print("=" * 70)

    ids = results["ids"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    for rank, (doc_id, dist, meta) in enumerate(zip(ids, distances, metas), start=1):
        item = {
            "rank": rank,
            "app_id": doc_id,
            "name": meta.get("name"),
            "distance": float(dist),
            "price": meta.get("price"),
            "tags": meta.get("tags"),
        }
        formatted.append(item)
        print(f"#{rank} [Distância Cosseno: {dist:.4f}] - {item['name']} (AppID: {doc_id})")
        print(f"    Preço: R$/$ {item['price']} | Tags: {item['tags'][:80]}...")

    print("=" * 70)
    return formatted


def index_all(force: bool = False, limit_samples: Optional[int] = None) -> None:
    """Executa a indexação nas duas coleções (games_nomic e games_bge)."""
    parquet_path = config.PROCESSED_DATA_DIR / "games.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Arquivo {parquet_path} não encontrado. Execute a Fase 2 (data_prep.py) primeiro."
        )

    df = pd.read_parquet(parquet_path)
    if limit_samples:
        logger.info(f"Limitando dataset a {limit_samples} amostras para teste rápido...")
        df = df.head(limit_samples)

    # 1. Indexa Coleção Nomic (Embedding A) - Lote de 128
    index_collection(
        df=df,
        collection_name=COLLECTION_NOMIC,
        embed_model=config.EMBED_MODEL_A,
        batch_size=128,
        force=force,
    )

    # 2. Indexa Coleção BGE (Embedding B) - Lote de 64
    index_collection(
        df=df,
        collection_name=COLLECTION_BGE,
        embed_model=config.EMBED_MODEL_B,
        batch_size=64,
        force=force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexador ChromaDB do GameFinder RAG.")
    parser.add_argument("--force", action="store_true", help="Força a recriação das coleções.")
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade de jogos para teste rápido.")
    parser.add_argument("--test-query", type=str, default="jogo de fazenda relaxante", help="Query para testar após indexação.")
    args = parser.parse_args()

    index_all(force=args.force, limit_samples=args.limit)

    # Executa busca de teste nas duas coleções
    search_raw(args.test_query, COLLECTION_NOMIC, k=5)
    search_raw(args.test_query, COLLECTION_BGE, k=5)
