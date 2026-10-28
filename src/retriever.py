"""Módulo de recuperação vetorial com filtros (Fase 6).

Realiza busca semântica no ChromaDB respeitando filtros extraídos pelo
query_parser, com relaxamento automático quando não há resultados.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import chromadb

from src import config
from src.indexer import COLLECTION_BGE, COLLECTION_NOMIC, get_chroma_client
from src.llm import embed
from src.query_parser import ParsedQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_where_clause(parsed: Union[ParsedQuery, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Monta cláusula where do ChromaDB a partir dos filtros extraídos."""
    if isinstance(parsed, ParsedQuery):
        data = parsed.model_dump()
    else:
        data = parsed

    conditions = []

    preco_max = data.get("preco_max")
    if preco_max is not None:
        try:
            conditions.append({"price": {"$lte": float(preco_max)}})
        except (TypeError, ValueError):
            pass

    plataforma = data.get("plataforma")
    if plataforma == "windows":
        conditions.append({"windows": True})
    elif plataforma == "mac":
        conditions.append({"mac": True})
    elif plataforma == "linux":
        conditions.append({"linux": True})

    coop = data.get("coop")
    if coop is True:
        conditions.append({"coop": True})

    multiplayer = data.get("multiplayer")
    if multiplayer is True and coop is not True:
        conditions.append({"multiplayer": True})

    ano_min = data.get("ano_min")
    if ano_min is not None:
        try:
            conditions.append({"release_year": {"$gte": int(ano_min)}})
        except (TypeError, ValueError):
            pass

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _search_with_where(
    query: str,
    collection_name: str,
    k: int,
    where: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Executa busca vetorial com filtros."""
    client = get_chroma_client()
    col = None
    try:
        col = client.get_collection(collection_name)
    except Exception:
        # Fallback para outra coleção
        for fb in [COLLECTION_NOMIC, COLLECTION_BGE]:
            try:
                col = client.get_collection(fb)
                collection_name = fb
                logger.warning(f"Usando coleção fallback {fb}")
                break
            except Exception:
                continue
    if col is None:
        raise RuntimeError(f"Coleção '{collection_name}' não encontrada. Execute o indexador primeiro: python src/indexer.py")

    embed_model = col.metadata.get("embed_model") or (
        config.EMBED_MODEL_A if collection_name == COLLECTION_NOMIC else config.EMBED_MODEL_B
    )

    query_vec = embed(texts=[query], model=embed_model)[0]

    try:
        results = col.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results
    except Exception as e:
        logger.warning(f"Erro na busca com filtros ({where}): {e}")
        # Tenta sem filtros como fallback
        results = col.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return results


def retrieve_candidates(
    parsed_query: Union[ParsedQuery, Dict[str, Any]],
    collection_name: str = COLLECTION_NOMIC,
    k: int = 20,
    relax_on_empty: bool = True,
) -> Dict[str, Any]:
    """Recupera candidatos do ChromaDB com filtros e relaxamento automático.

    Returns:
        Dict com keys: candidates, relaxed (bool), applied_where, message
    """
    if isinstance(parsed_query, ParsedQuery):
        data = parsed_query.model_dump()
        busca_semantica = parsed_query.busca_semantica
    else:
        data = parsed_query
        busca_semantica = data.get("busca_semantica", "")

    if not busca_semantica:
        busca_semantica = ""

    applied_where = _build_where_clause(data)
    results = _search_with_where(busca_semantica, collection_name, k, applied_where)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates = []
    for i in range(len(ids)):
        candidates.append(
            {
                "app_id": ids[i],
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )

    relaxed = False
    message = None

    if len(candidates) == 0 and relax_on_empty and applied_where is not None:
        logger.info("Nenhum resultado com filtros aplicados. Relaxando filtros...")
        # Tentar relaxar progressivamente
        # 1. Remover filtros de plataforma/coop/multiplayer restritivos, manter apenas preco_max se muito permissivo
        relaxed_data = dict(data)
        # Remover restrições mais duras primeiro
        relaxed_data["plataforma"] = None
        relaxed_data["coop"] = None
        relaxed_data["multiplayer"] = None
        relaxed_data["ano_min"] = None
        # Tentar manter preco_max apenas se for razoável
        where2 = _build_where_clause(relaxed_data)
        results2 = _search_with_where(busca_semantica, collection_name, k, where2)
        ids2 = results2.get("ids", [[]])[0]
        docs2 = results2.get("documents", [[]])[0]
        metas2 = results2.get("metadatas", [[]])[0]
        dists2 = results2.get("distances", [[]])[0]
        candidates2 = []
        for i in range(len(ids2)):
            candidates2.append(
                {
                    "app_id": ids2[i],
                    "document": docs2[i] if i < len(docs2) else "",
                    "metadata": metas2[i] if i < len(metas2) else {},
                    "distance": dists2[i] if i < len(dists2) else None,
                }
            )
        if len(candidates2) > 0:
            candidates = candidates2
            applied_where = where2
            relaxed = True
            message = "Filtros relaxados automaticamente (plataforma/modos/ano removidos) por falta de resultados."

        # 2. Se ainda vazio, remover também preco_max
        if len(candidates) == 0:
            relaxed_data["preco_max"] = None
            where3 = _build_where_clause(relaxed_data)
            results3 = _search_with_where(busca_semantica, collection_name, k, where3)
            ids3 = results3.get("ids", [[]])[0]
            docs3 = results3.get("documents", [[]])[0]
            metas3 = results3.get("metadatas", [[]])[0]
            dists3 = results3.get("distances", [[]])[0]
            candidates3 = []
            for i in range(len(ids3)):
                candidates3.append(
                    {
                        "app_id": ids3[i],
                        "document": docs3[i] if i < len(docs3) else "",
                        "metadata": metas3[i] if i < len(metas3) else {},
                        "distance": dists3[i] if i < len(dists3) else None,
                    }
                )
            if len(candidates3) > 0:
                candidates = candidates3
                applied_where = where3
                relaxed = True
                message = "Todos os filtros foram relaxados por falta de resultados. Exibindo resultados puramente semânticos."

    return {
        "candidates": candidates,
        "relaxed": relaxed,
        "applied_where": applied_where,
        "message": message,
        "busca_semantica": busca_semantica,
    }
