"""Módulo de visualização vetorial com UMAP (Fase 8).

Gera projeção 2D dos embeddings das coleções ChromaDB, adiciona coluna
'genre' e salva cache em data/processed/ para uso na interface Gradio.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import umap
from tqdm import tqdm

from src import config
from src.indexer import COLLECTION_BGE, COLLECTION_NOMIC, get_chroma_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_games_metadata() -> pd.DataFrame:
    """Carrega metadados do games.parquet para enriquecer com gêneros."""
    parquet_path = config.PROCESSED_DATA_DIR / "games.parquet"
    if not parquet_path.exists():
        logger.warning("games.parquet não encontrado. Coluna 'genre' ficará vazia.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_path)
        return df
    except Exception as e:
        logger.warning(f"Erro ao carregar games.parquet: {e}")
        return pd.DataFrame()


def _extract_main_genre(row: pd.Series) -> str:
    """Extrai o gênero principal a partir de genres, tags ou metadados."""
    # Tentar coluna genres (lista)
    genres = row.get("genres")
    if isinstance(genres, (list, np.ndarray)) and len(genres) > 0:
        genre = str(genres[0]).strip()
        if genre and genre.lower() != "nan":
            return genre

    # Tentar coluna genre direta
    genre_direct = row.get("genre")
    if isinstance(genre_direct, str) and genre_direct.strip() and genre_direct.lower() != "nan":
        return genre_direct.strip()

    # Fallback: tentar extrair primeira tag relevante
    tags = row.get("tags")
    if isinstance(tags, (list, np.ndarray)) and len(tags) > 0:
        # Pegar primeira tag como fallback
        tag = str(tags[0]).strip()
        if tag and tag.lower() != "nan":
            return tag
    elif isinstance(tags, str) and tags.strip():
        # Se for string com vírgulas
        parts = [p.strip() for p in tags.split(",")]
        if parts and parts[0]:
            return parts[0]

    return "Outros"


def generate_umap(
    collection_name: str = COLLECTION_NOMIC,
    max_samples: int = 3000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    """Gera projeção UMAP 2D para uma coleção e salva em cache.

    Args:
        collection_name: Nome da coleção (games_nomic ou games_bge)
        max_samples: Máximo de amostras para plotar (padrão 3000)
        n_neighbors: Parâmetro UMAP
        min_dist: Parâmetro UMAP
        random_state: Semente para reprodutibilidade
        force: Forçar regeneração mesmo se cache existir

    Returns:
        DataFrame com colunas x,y,name,app_id,price,tags,genre,...
    """
    # Definir caminho do cache
    if collection_name == COLLECTION_NOMIC:
        cache_path = config.PROCESSED_DATA_DIR / "umap_games_nomic.parquet"
    elif collection_name == COLLECTION_BGE:
        cache_path = config.PROCESSED_DATA_DIR / "umap_games_bge.parquet"
    else:
        cache_path = config.PROCESSED_DATA_DIR / f"umap_{collection_name}.parquet"

    if cache_path.exists() and not force:
        logger.info(f"Carregando cache existente: {cache_path}")
        try:
            df_cache = pd.read_parquet(cache_path)
            logger.info(f"Cache carregado com {len(df_cache)} pontos")
            return df_cache
        except Exception as e:
            logger.warning(f"Erro ao carregar cache, regenerando: {e}")

    logger.info(f"Gerando projeção UMAP para coleção: {collection_name}")
    client = get_chroma_client()

    try:
        col = client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Coleção '{collection_name}' não encontrada. "
            "Execute o indexador primeiro: python src/indexer.py"
        ) from e

    # Carregar todos os dados da coleção
    logger.info("Extraindo embeddings e metadados do ChromaDB...")
    try:
        result = col.get(include=["embeddings", "metadatas", "documents"])
    except Exception as e:
        logger.error(f"Erro ao extrair dados: {e}")
        raise

    ids = result.get("ids", [])
    embeddings = result.get("embeddings")
    metadatas = result.get("metadatas", [])

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Nenhum embedding encontrado na coleção")

    logger.info(f"Total de embeddings extraídos: {len(embeddings)}")

    # Converter embeddings para array numpy
    embeddings_array = np.array(embeddings)

    # Carregar metadados enriquecidos do games.parquet
    games_df = _load_games_metadata()
    games_meta_dict = {}
    if not games_df.empty and "app_id" in games_df.columns:
        for _, row in games_df.iterrows():
            games_meta_dict[str(row["app_id"])] = row.to_dict()

    # Construir DataFrame
    data_rows = []
    for i, app_id in enumerate(ids):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        games_meta = games_meta_dict.get(str(app_id), {})

        # Mesclar metadados
        name = meta.get("name") or games_meta.get("name", f"Jogo {app_id}")
        price = meta.get("price", games_meta.get("price", 0.0))
        tags = meta.get("tags", games_meta.get("tags", ""))
        short_desc = meta.get("short_description", games_meta.get("short_description", ""))

        # Extrair gênero
        genre_row = pd.Series({**games_meta, **meta})
        genre = _extract_main_genre(genre_row)

        data_rows.append(
            {
                "app_id": str(app_id),
                "name": name,
                "price": float(price) if price is not None else 0.0,
                "tags": str(tags),
                "short_description": str(short_desc),
                "genre": genre,
                "embedding_idx": i,
            }
        )

    df = pd.DataFrame(data_rows)
    logger.info(f"DataFrame construído com {len(df)} registros")

    # Amostragem para melhor performance
    if len(df) > max_samples:
        logger.info(f"Amostrando {max_samples} jogos de {len(df)} para visualização...")
        df_sample = df.sample(n=max_samples, random_state=random_state).copy()
        embeddings_sample = embeddings_array[df_sample["embedding_idx"].values]
    else:
        df_sample = df.copy()
        embeddings_sample = embeddings_array

    logger.info(f"Executando UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric=cosine)...")
    logger.info("Isso pode levar alguns minutos...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )

    embedding_2d = reducer.fit_transform(embeddings_sample)

    df_sample["x"] = embedding_2d[:, 0]
    df_sample["y"] = embedding_2d[:, 1]

    # Remover coluna auxiliar
    df_sample = df_sample.drop(columns=["embedding_idx"])

    # Salvar cache
    logger.info(f"Salvando cache em: {cache_path}")
    df_sample.to_parquet(cache_path, index=False)
    logger.info(f"Cache salvo com sucesso! {len(df_sample)} pontos gerados.")

    return df_sample


def generate_all(force: bool = False, max_samples: int = 3000) -> None:
    """Gera projeções UMAP para ambas as coleções."""
    logger.info("=" * 60)
    logger.info("GERANDO MAPA VETORIAL 2D COM UMAP (FASE 8)")
    logger.info("=" * 60)

    try:
        generate_umap(COLLECTION_NOMIC, max_samples=max_samples, force=force)
    except Exception as e:
        logger.error(f"Erro ao gerar UMAP para games_nomic: {e}")

    try:
        generate_umap(COLLECTION_BGE, max_samples=max_samples, force=force)
    except Exception as e:
        logger.error(f"Erro ao gerar UMAP para games_bge: {e}")

    logger.info("=" * 60)
    logger.info("Processo concluído!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gerador de mapa vetorial UMAP (Fase 8)")
    parser.add_argument("--force", action="store_true", help="Força regeneração do cache")
    parser.add_argument("--max-samples", type=int, default=3000, help="Máximo de pontos a plotar")
    parser.add_argument(
        "--collection",
        type=str,
        default="all",
        choices=["all", "nomic", "bge"],
        help="Qual coleção gerar",
    )
    args = parser.parse_args()

    if args.collection == "all":
        generate_all(force=args.force, max_samples=args.max_samples)
    elif args.collection == "nomic":
        generate_umap(COLLECTION_NOMIC, max_samples=args.max_samples, force=args.force)
    elif args.collection == "bge":
        generate_umap(COLLECTION_BGE, max_samples=args.max_samples, force=args.force)
