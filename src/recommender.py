"""Módulo de recomendação RAG completo (Fase 6).

Orquestra parser -> retriever -> LLM com validação anti-alucinação,
métricas (tokens/tempo) e modo use_rag=False.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from src import config
from src.indexer import COLLECTION_NOMIC
from src.llm import chat
from src.query_parser import ParsedQuery, parse_query
from src.retriever import retrieve_candidates
from src.tokens import count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_recommend_prompt() -> str:
    prompt_path = config.PROMPTS_DIR / "recommend.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _format_context(candidates: List[Dict[str, Any]]) -> str:
    """Formata candidatos numerados para o contexto."""
    lines = []
    for idx, c in enumerate(candidates, start=1):
        meta = c.get("metadata", {}) or {}
        name = meta.get("name") or c.get("name") or ""
        price = meta.get("price", 0.0)
        tags = meta.get("tags", "")
        total_reviews = meta.get("total_reviews", 0)
        short_desc = meta.get("short_description") or c.get("document") or ""
        if len(short_desc) > 200:
            short_desc = short_desc[:197] + "..."
        lines.append(
            f"[{idx}] app_id: {c['app_id']} | Nome: {name} | Preço: ${price:.2f} | "
            f"Avaliações: {total_reviews} | Tags: {tags} | Descrição: {short_desc}"
        )
    return "\n".join(lines)


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extrai array JSON da resposta (aceita array ou objeto único)."""
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "recommendations" in data and isinstance(data["recommendations"], list):
                return data["recommendations"]
            # objeto único -> wrap
            return [data]
    except json.JSONDecodeError:
        pass

    # Tenta array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Tenta objeto único
    match_obj = re.search(r'\{.*\}', text, re.DOTALL)
    if match_obj:
        try:
            data = json.loads(match_obj.group(0))
            if isinstance(data, dict):
                if "recommendations" in data and isinstance(data["recommendations"], list):
                    return data["recommendations"]
                return [data]
        except json.JSONDecodeError:
            pass
    return []


def recommend(
    user_query: str,
    collection_name: str = COLLECTION_NOMIC,
    use_rag: bool = True,
    use_fewshot_parser: bool = True,
    temperature: float = 0.2,
    k_candidates: int = 20,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Pipeline completo de recomendação RAG.

    Args:
        user_query: Pedido em linguagem natural
        collection_name: games_nomic ou games_bge
        use_rag: Se False, faz recomendação sem contexto vetorial (modo didático)
        use_fewshot_parser: Usar few-shot no parser
        temperature: Temperatura do LLM para recomendação
        k_candidates: Número de candidatos a recuperar
        max_results: Máximo de recomendações a retornar

    Returns:
        Dict com recomendações validadas, métricas, filtros, contexto etc.
    """
    timings = {}
    t_start = time.time()

    # 1. Parser
    t0 = time.time()
    parsed = parse_query(user_query, use_fewshot=use_fewshot_parser, temperature=0.0)
    if isinstance(parsed, ParsedQuery):
        parsed_dict = parsed.model_dump()
    else:
        parsed_dict = parsed
    timings["parser"] = time.time() - t0

    candidates = []
    retrieval_info = {}

    if use_rag:
        # 2. Retriever
        t0 = time.time()
        retrieval_info = retrieve_candidates(parsed, collection_name=collection_name, k=k_candidates)
        candidates = retrieval_info.get("candidates", [])
        timings["retriever"] = time.time() - t0
    else:
        timings["retriever"] = 0.0

    # 3. Montagem do contexto e chamada ao LLM
    t0 = time.time()
    prompt_template = _load_recommend_prompt()

    if use_rag and candidates:
        context_list = _format_context(candidates)
        parsed_filters_str = json.dumps(parsed_dict, ensure_ascii=False)
        prompt = prompt_template.format(
            USER_QUERY=user_query.strip(),
            PARSED_FILTERS=parsed_filters_str,
            CONTEXT_LIST=context_list,
        )
    else:
        # Modo sem RAG - contexto vazio, mas mantém estrutura
        parsed_filters_str = json.dumps(parsed_dict, ensure_ascii=False)
        prompt = prompt_template.format(
            USER_QUERY=user_query.strip(),
            PARSED_FILTERS=parsed_filters_str,
            CONTEXT_LIST="[Nenhum jogo disponível no contexto. Siga estritamente as regras - não invente jogos.]",
        )

    context_tokens = count_tokens(prompt)

    messages = [
        {"role": "system", "content": "Você segue rigorosamente as instruções. Retorne apenas JSON válido."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = chat(messages=messages, temperature=temperature, json_mode=True)
    except Exception as e:
        logger.error(f"Erro na geração de recomendação: {e}")
        response = "[]"

    timings["llm_generation"] = time.time() - t0
    timings["total"] = time.time() - t_start

    # 4. Validação cruzada de app_ids
    raw_recs = _extract_json_array(response)
    valid_app_ids = set(str(c["app_id"]) for c in candidates) if candidates else set()

    validated = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        app_id = rec.get("app_id") or rec.get("id") or rec.get("game_id")
        motivo = rec.get("motivo") or rec.get("reason") or rec.get("justificativa") or ""
        if app_id is None:
            continue
        app_id_str = str(app_id)
        if use_rag and candidates:
            if app_id_str not in valid_app_ids:
                logger.warning(f"app_id {app_id_str} não encontrado nos candidatos - descartado (possível alucinação)")
                continue
        # Montar recomendação com metadados
        meta = {}
        doc = ""
        dist = None
        for c in candidates:
            if str(c["app_id"]) == app_id_str:
                meta = c.get("metadata", {}) or {}
                doc = c.get("document", "")
                dist = c.get("distance")
                break
        validated.append(
            {
                "app_id": app_id_str,
                "name": meta.get("name", ""),
                "price": meta.get("price"),
                "tags": meta.get("tags", ""),
                "short_description": meta.get("short_description", doc[:200] if doc else ""),
                "distance": dist,
                "motivo": motivo,
            }
        )
        if len(validated) >= max_results:
            break

    return {
        "user_query": user_query,
        "parsed": parsed_dict,
        "use_rag": use_rag,
        "collection_name": collection_name,
        "candidates_count": len(candidates),
        "retrieval": retrieval_info,
        "context_tokens": context_tokens,
        "timings": timings,
        "raw_response": response,
        "recommendations": validated,
        "recommendations_count": len(validated),
    }
