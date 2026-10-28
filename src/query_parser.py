"""Módulo de extração de filtros a partir de pedidos em linguagem natural (Fase 5).

Converte a query do usuário em um JSON estruturado com filtros e termos
semânticos para busca vetorial, com validação Pydantic, retry e fallback.
"""

import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from src import config
from src.llm import chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ParsedQuery(BaseModel):
    """Schema estrito para os filtros extraídos do pedido do usuário."""

    busca_semantica: str = Field(..., description="Termos para busca vetorial semântica")
    preco_max: Optional[float] = Field(default=None, ge=0)
    plataforma: Optional[Literal["windows", "mac", "linux"]] = None
    multiplayer: Optional[bool] = None
    coop: Optional[bool] = None
    ano_min: Optional[int] = Field(default=None, ge=1970, le=2100)


def _load_prompt(use_fewshot: bool = True) -> str:
    """Carrega o prompt de parsing apropriado."""
    if use_fewshot:
        prompt_path = config.PROMPTS_DIR / "parse_query.txt"
    else:
        prompt_path = config.PROMPTS_DIR / "parse_query_zeroshot.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Tenta extrair JSON válido de um texto que pode conter markdown ou ruído."""
    # Tenta parse direto primeiro
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Procura por bloco JSON entre chaves
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _safe_fallback(query: str) -> dict:
    """Fallback seguro quando parsing falha."""
    return {
        "busca_semantica": query.strip() if query.strip() else "",
        "preco_max": None,
        "plataforma": None,
        "multiplayer": None,
        "coop": None,
        "ano_min": None,
    }


def parse_query(user_query: str, use_fewshot: bool = True, temperature: float = 0.0) -> Union[ParsedQuery, dict]:
    """Extrai filtros estruturados do pedido do usuário.

    Args:
        user_query: Pedido em linguagem natural do usuário.
        use_fewshot: Se True usa prompt com exemplos (padrão), senão usa zero-shot.
        temperature: Temperatura do LLM (padrão 0.0 para maior determinismo).

    Returns:
        ParsedQuery validado ou dict de fallback.
    """
    if not user_query or not user_query.strip():
        logger.warning("Query vazia recebida. Retornando fallback.")
        return _safe_fallback(user_query)

    system_prompt = _load_prompt(use_fewshot=use_fewshot)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query.strip()},
    ]

    # Primeira tentativa
    try:
        response = chat(messages=messages, temperature=temperature, json_mode=True)
        parsed = _extract_json_from_text(response)

        if parsed:
            try:
                validated = ParsedQuery.model_validate(parsed)
                return validated
            except ValidationError as e:
                logger.warning(f"Falha na validação Pydantic: {e}")
        else:
            logger.warning("Não foi possível extrair JSON da resposta do LLM.")
    except Exception as e:
        logger.warning(f"Erro na primeira tentativa de parsing: {e}")

    # Segunda tentativa com prompt de correção
    try:
        correction_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query.strip()},
            {"role": "assistant", "content": "Retorne apenas JSON válido."},
            {"role": "user", "content": "Corrija sua resposta anterior para retornar APENAS o JSON no esquema exato, sem texto, explicações ou markdown."},
        ]
        response = chat(messages=correction_messages, temperature=temperature, json_mode=True)
        parsed = _extract_json_from_text(response)

        if parsed:
            try:
                validated = ParsedQuery.model_validate(parsed)
                logger.info("Parsing bem-sucedido na segunda tentativa.")
                return validated
            except ValidationError as e:
                logger.warning(f"Falha na validação Pydantic na segunda tentativa: {e}")
    except Exception as e:
        logger.warning(f"Erro na segunda tentativa de parsing: {e}")

    # Fallback
    logger.warning("Usando fallback seguro - utilizando query original como busca_semantica")
    return _safe_fallback(user_query)


def parse_query_dict(user_query: str, use_fewshot: bool = True, temperature: float = 0.0) -> dict:
    """Versão que sempre retorna dict (útil para integração)."""
    result = parse_query(user_query, use_fewshot=use_fewshot, temperature=temperature)
    if isinstance(result, ParsedQuery):
        return result.model_dump()
    return result
