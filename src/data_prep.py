"""Módulo de limpeza e preparação do dataset de jogos da Steam.

Lê o arquivo bruto em data/raw/ (Parquet ou CSV), normaliza os tipos e metadados,
aplica filtros de qualidade (avaliações mínimas, remoção de registros sem descrição),
seleciona os top N jogos mais relevantes e constrói a coluna doc_text para embeddings.
Salva o resultado limpo em data/processed/games.parquet.
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from src import config
from src.tokens import truncate_to_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _to_list_of_strings(val: Any) -> List[str]:
    """Converte valores variados (ndarray, lista, string separada por vírgula) em list[str]."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, np.ndarray)):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str):
        cleaned = val.strip("[]'\"")
        if not cleaned:
            return []
        # Pode estar separado por vírgula ou aspas
        items = [item.strip(" '\"") for item in cleaned.split(",")]
        return [item for item in items if item]
    return [str(val).strip()]


def _extract_release_year(val: Any) -> Optional[int]:
    """Extrai o ano (4 dígitos) de uma data de lançamento."""
    if not val or pd.isna(val):
        return None
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(val))
    if match:
        return int(match.group(1))
    return None


def _is_multiplayer(categories: List[str], genres: List[str], tags: List[str]) -> bool:
    """Detecta se o jogo tem suporte a modo multiplayer a partir de categorias e tags."""
    keywords = {"multi-player", "multiplayer", "pvp", "online pvp", "mmo", "massively multiplayer", "lan pvp"}
    all_terms = [t.lower() for t in categories + genres + tags]
    return any(any(k in term for k in keywords) for term in all_terms)


def _is_coop(categories: List[str], genres: List[str], tags: List[str]) -> bool:
    """Detecta se o jogo tem suporte a modo cooperativo a partir de categorias e tags."""
    keywords = {"co-op", "coop", "online co-op", "lan co-op", "shared/split screen co-op"}
    all_terms = [t.lower() for t in categories + genres + tags]
    return any(any(k in term for k in keywords) for term in all_terms)


def load_raw_dataframe() -> pd.DataFrame:
    """Carrega o dataset bruto de data/raw/, priorizando Parquet sobre CSV."""
    raw_dir = config.RAW_DATA_DIR
    parquet_files = list(raw_dir.glob("*.parquet"))
    csv_files = list(raw_dir.glob("*.csv"))

    if parquet_files:
        chosen = parquet_files[0]
        logger.info(f"Carregando dataset bruto de: {chosen}")
        return pd.read_parquet(chosen)

    if csv_files:
        chosen = csv_files[0]
        logger.info(f"Carregando dataset bruto de: {chosen}")
        return pd.read_csv(chosen, low_memory=False)

    raise FileNotFoundError(f"Nenhum arquivo Parquet ou CSV encontrado em {raw_dir}")


def clean_and_prepare_dataset(
    top_n: int = 8000,
    min_reviews: int = 50,
) -> pd.DataFrame:
    """Executa o pipeline completo de limpeza, padronização e estruturação dos jogos.

    Args:
        top_n: Quantidade máxima de jogos mais bem avaliados a manter (padrão 8000).
        min_reviews: Quantidade mínima de avaliações para manter um jogo.

    Returns:
        DataFrame padronizado e limpo.
    """
    df = load_raw_dataframe()
    total_raw = len(df)
    logger.info(f"Registros brutos carregados: {total_raw}")

    # Normaliza nomes de colunas para identificação flexível (lowercase)
    col_map = {col.lower(): col for col in df.columns}

    # 1. Identifica colunas de ID e Nome
    id_col = col_map.get("appid", col_map.get("app_id", "appID"))
    name_col = col_map.get("name", "name")

    # 2. Descrições
    short_desc_col = col_map.get("short_description")
    detailed_desc_col = col_map.get("detailed_description", col_map.get("about the game", "about_the_game"))

    # 3. Avaliações (positive e negative)
    pos_col = col_map.get("positive")
    neg_col = col_map.get("negative")

    # 4. Outras colunas
    price_col = col_map.get("price")
    release_date_col = col_map.get("release_date", col_map.get("release date", "release_date"))
    win_col = col_map.get("windows")
    mac_col = col_map.get("mac")
    lin_col = col_map.get("linux")
    cat_col = col_map.get("categories")
    gen_col = col_map.get("genres")
    tags_col = col_map.get("tags")

    # Processamento linha a linha estruturado
    cleaned_rows = []

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() in ("nan", "none", "unknown"):
            continue

        # ID
        try:
            app_id = int(row.get(id_col, 0))
        except (ValueError, TypeError):
            continue

        if app_id <= 0:
            continue

        # Descrição
        short_desc = str(row.get(short_desc_col, "")).strip() if short_desc_col else ""
        if not short_desc or short_desc.lower() == "nan":
            detail = str(row.get(detailed_desc_col, "")).strip() if detailed_desc_col else ""
            if detail and detail.lower() != "nan":
                # Pega as primeiras frases até 300 caracteres
                short_desc = detail[:300].strip()

        # Remove HTML tags simples se houver
        short_desc = re.sub(r"<[^>]+>", " ", short_desc)
        short_desc = re.sub(r"\s+", " ", short_desc).strip()

        if not short_desc:
            continue

        # Reviews
        pos = int(row.get(pos_col, 0)) if pos_col and not pd.isna(row.get(pos_col)) else 0
        neg = int(row.get(neg_col, 0)) if neg_col and not pd.isna(row.get(neg_col)) else 0
        total_rev = pos + neg

        if total_rev < min_reviews:
            continue

        # Preço
        raw_price = row.get(price_col, 0.0)
        try:
            price = float(raw_price) if not pd.isna(raw_price) else 0.0
        except (ValueError, TypeError):
            price = 0.0

        # Categorias, Gêneros e Tags
        categories = _to_list_of_strings(row.get(cat_col)) if cat_col else []
        genres = _to_list_of_strings(row.get(gen_col)) if gen_col else []
        tags = _to_list_of_strings(row.get(tags_col)) if tags_col else []
        if not tags and genres:
            tags = list(genres)

        # Modos de jogo
        multiplayer = _is_multiplayer(categories, genres, tags)
        coop = _is_coop(categories, genres, tags)

        # Plataformas
        windows = bool(row.get(win_col, True)) if win_col else True
        mac = bool(row.get(mac_col, False)) if mac_col else False
        linux = bool(row.get(lin_col, False)) if lin_col else False

        # Ano de lançamento
        release_year = _extract_release_year(row.get(release_date_col))

        # Síntese do doc_text para o embedding
        genres_str = ", ".join(genres) if genres else "Geral"
        tags_str = ", ".join(tags[:10]) if tags else genres_str
        doc_text = (
            f"Nome: {name} | "
            f"Gêneros: {genres_str} | "
            f"Tags: {tags_str} | "
            f"Descrição: {short_desc}"
        )
        # Garante que nenhum documento exceda a janela de tokens do embedding
        doc_text = truncate_to_tokens(doc_text, max_tokens=512)

        cleaned_rows.append({
            "app_id": app_id,
            "name": name,
            "short_description": short_desc,
            "tags": tags,
            "genres": genres,
            "price": price,
            "windows": windows,
            "mac": mac,
            "linux": linux,
            "multiplayer": multiplayer,
            "coop": coop,
            "positive_reviews": pos,
            "total_reviews": total_rev,
            "release_year": release_year,
            "doc_text": doc_text,
        })

    clean_df = pd.DataFrame(cleaned_rows)
    logger.info(f"Jogos após filtros de descrição e mínimo de {min_reviews} reviews: {len(clean_df)}")

    # Ordena pelos mais avaliados e mantém os Top N
    clean_df = clean_df.sort_values(by="total_reviews", ascending=False).head(top_n).reset_index(drop=True)
    logger.info(f"Mantidos os Top {len(clean_df)} jogos mais avaliados.")

    # Salva no formato Parquet processado
    output_path = config.PROCESSED_DATA_DIR / "games.parquet"
    clean_df.to_parquet(output_path, index=False)
    logger.info(f"Dataset processado salvo com sucesso em: {output_path}")

    return clean_df


def print_stats(df: pd.DataFrame) -> None:
    """Imprime estatísticas do dataset limpo."""
    print("=" * 70)
    print("           ESTATÍSTICAS DO DATASET LIMPO (games.parquet)           ")
    print("=" * 70)
    print(f"Total de jogos selecionados: {len(df)}")
    print(f"Preço médio:                 R$/$ {df['price'].mean():.2f}")
    print(f"Jogos gratuitos (preço 0):   {(df['price'] == 0).sum()} ({(df['price'] == 0).mean() * 100:.1f}%)")
    print(f"Suporte a Windows:           {df['windows'].sum()} ({df['windows'].mean() * 100:.1f}%)")
    print(f"Suporte a Mac:               {df['mac'].sum()} ({df['mac'].mean() * 100:.1f}%)")
    print(f"Suporte a Linux:             {df['linux'].sum()} ({df['linux'].mean() * 100:.1f}%)")
    print(f"Jogos Multiplayer:           {df['multiplayer'].sum()} ({df['multiplayer'].mean() * 100:.1f}%)")
    print(f"Jogos Cooperativos (Co-op):  {df['coop'].sum()} ({df['coop'].mean() * 100:.1f}%)")
    print(f"Média de avaliações/jogo:    {df['total_reviews'].mean():.0f}")
    print(f"Mediana de avaliações/jogo:  {df['total_reviews'].median():.0f}")
    print("-" * 70)
    print("Exemplo do doc_text gerado (primeiro jogo):")
    print(df["doc_text"].iloc[0][:150] + "...")
    print("=" * 70)


if __name__ == "__main__":
    df_clean = clean_and_prepare_dataset(top_n=8000, min_reviews=50)
    print_stats(df_clean)
