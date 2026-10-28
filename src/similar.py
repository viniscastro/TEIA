"""Módulo para buscar jogos similares (Fase 7).

Busca os jogos mais parecidos vetorialmente com base em um app_id,
utilizando as mesmas coleções do ChromaDB.
"""

import logging
from typing import Any, Dict, List

from src import config
from src.indexer import COLLECTION_NOMIC, get_chroma_client
from src.llm import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_similar(app_id: str, collection_name: str = COLLECTION_NOMIC, k: int = 8) -> List[Dict[str, Any]]:
    """Encontra jogos mais parecidos com base em um app_id."""
    client = get_chroma_client()
    app_id_str = str(app_id).strip()

    # Tentar coleção solicitada primeiro
    col = None
    try:
        col = client.get_collection(collection_name)
    except Exception:
        pass

    # Se não encontrar na coleção solicitada, tentar fallback
    if col is None:
        for fallback_col in [COLLECTION_NOMIC, COLLECTION_BGE]:
            try:
                col = client.get_collection(fallback_col)
                collection_name = fallback_col
                logger.warning(f"Usando coleção fallback: {fallback_col}")
                break
            except Exception:
                continue

    if col is None:
        raise RuntimeError("Nenhuma coleção encontrada. Execute o indexador primeiro: python src/indexer.py")

    try:
        result = col.get(ids=[app_id_str], include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        logger.error(f"Erro ao buscar jogo {app_id_str}: {e}")
        raise

    if not result or not result.get("ids") or len(result["ids"]) == 0:
        # Tentar na outra coleção
        other_col_name = COLLECTION_BGE if collection_name == COLLECTION_NOMIC else COLLECTION_NOMIC
        try:
            col2 = client.get_collection(other_col_name)
            result = col2.get(ids=[app_id_str], include=["documents", "metadatas", "embeddings"])
            if result and result.get("ids"):
                col = col2
                collection_name = other_col_name
                logger.warning(f"Jogo encontrado na coleção alternativa: {other_col_name}")
            else:
                raise ValueError(f"Jogo com app_id {app_id_str} não encontrado em nenhuma coleção")
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"Jogo com app_id {app_id_str} não encontrado na coleção {collection_name}")

    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        try:
            meta = result.get("metadatas", [{}])[0] if result.get("metadatas") else {}
            name = meta.get("name", "")
            if name:
                from src.llm import embed
                embed_model = col.metadata.get("embed_model") or config.EMBED_MODEL_A
                ref_embedding = embed([name], model=embed_model)[0]
            else:
                raise ValueError(f"Não foi possível obter embedding do jogo {app_id_str}")
        except Exception:
            raise ValueError(f"Não foi possível obter embedding do jogo {app_id_str}")
    else:
        ref_embedding = embeddings[0]

    try:
        similar_results = col.query(
            query_embeddings=[ref_embedding],
            n_results=k + 1,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Erro ao buscar similares: {e}")
        raise

    similar = []
    ids = similar_results.get("ids", [[]])[0]
    docs = similar_results.get("documents", [[]])[0]
    metas = similar_results.get("metadatas", [[]])[0]
    dists = similar_results.get("distances", [[]])[0]

    for i in range(len(ids)):
        sim_id = str(ids[i])
        if sim_id == app_id_str:
            continue
        similar.append(
            {
                "app_id": sim_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
        if len(similar) >= k:
            break

    return similar


def get_all_games_for_dropdown(collection_name: str = COLLECTION_NOMIC) -> List[Dict[str, str]]:
    """Retorna lista de jogos para popular dropdown (com nome e app_id)."""
    parquet_path = config.PROCESSED_DATA_DIR / "games.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            games = [
                {"label": f"{str(row['name'])} ({row['app_id']})", "value": str(row['app_id'])}
                for _, row in df.iterrows()
            ]
            games.sort(key=lambda x: x["label"])
            return games
        except Exception as e:
            logger.warning(f"Não foi possível carregar dropdown de games.parquet: {e}")

    client = get_chroma_client()
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    try:
        result = col.get(include=["metadatas"])
        games = []
        if result and result.get("ids"):
            for i, game_id in enumerate(result["ids"]):
                meta = result["metadatas"][i] if result.get("metadatas") and i < len(result["metadatas"]) else {}
                name = meta.get("name", f"Jogo {game_id}")
                games.append({"label": f"{name} ({game_id})", "value": str(game_id)})
        games.sort(key=lambda x: x["label"])
        return games
    except Exception as e:
        logger.error(f"Erro ao obter lista de jogos do ChromaDB: {e}")
        return []
