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
    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"Coleção '{collection_name}' não encontrada. Execute o indexador primeiro: python src/indexer.py") from e

    app_id_str = str(app_id).strip()

    try:
        result = col.get(ids=[app_id_str], include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        logger.error(f"Erro ao buscar jogo {app_id_str}: {e}")
        raise

    if not result or not result.get("ids") or len(result["ids"]) == 0:
        raise ValueError(f"Jogo com app_id {app_id_str} não encontrado na coleção {collection_name}")

    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        # Tentar buscar por query com metadados se embeddings não vierem
        try:
            meta = result.get("metadatas", [{}])[0] if result.get("metadatas") else {}
            name = meta.get("name", "")
            if name:
                from src.llm import embed
                ref_embedding = embed([name], model=col.metadata.get("embed_model"))[0]
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
    client = get_chroma_client()
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    try:
        # Pega todos os metadados
        result = col.get(include=["metadatas"])
        games = []
        if result and result.get("ids"):
            for i, game_id in enumerate(result["ids"]):
                meta = result["metadatas"][i] if result.get("metadatas") and i < len(result["metadatas"]) else {}
                name = meta.get("name", f"Jogo {game_id}")
                games.append({"label": f"{name} ({game_id})", "value": str(game_id)})
        # Ordena por nome
        games.sort(key=lambda x: x["label"])
        return games
    except Exception as e:
        logger.error(f"Erro ao obter lista de jogos: {e}")
        return []
