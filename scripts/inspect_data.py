"""Script para inspecionar o arquivo bruto em data/raw/ (Parquet, CSV ou JSON).

Exibe colunas, tipos de dados, 3 linhas de exemplo e porcentagem de nulos por coluna.
"""

import sys
import io
from pathlib import Path
import pandas as pd

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def load_raw_data() -> tuple[pd.DataFrame, str]:
    """Carrega uma amostra do dataset bruto em data/raw/."""
    raw_dir = config.RAW_DATA_DIR
    parquet_files = list(raw_dir.glob("*.parquet"))
    csv_files = list(raw_dir.glob("*.csv"))
    json_files = list(raw_dir.glob("*.json"))

    # Prioriza parquet por ser mais rápido e já tipado, depois CSV e JSON
    if parquet_files:
        chosen = parquet_files[0]
        print(f"Carregando amostra de arquivo Parquet: {chosen.name}")
        df = pd.read_parquet(chosen)
        return df, chosen.name

    if csv_files:
        chosen = csv_files[0]
        print(f"Carregando amostra de arquivo CSV: {chosen.name}")
        # Trata o caso comum no CSV do FronkonGames onde o cabeçalho tem 'DiscountDLC count'
        with open(chosen, "r", encoding="utf-8", errors="ignore") as f:
            header = next(f)
            if "DiscountDLC count" in header:
                header = header.replace("DiscountDLC count", "Discount,DLC count")
            lines = [header] + [next(f) for _ in range(5000)]
            df = pd.read_csv(io.StringIO("".join(lines)))
            return df, chosen.name

    if json_files:
        chosen = json_files[0]
        print(f"Carregando amostra de arquivo JSON: {chosen.name}")
        df = pd.read_json(chosen)
        return df, chosen.name

    return pd.DataFrame(), ""


def inspect() -> None:
    df, filename = load_raw_data()

    if df.empty:
        print("=" * 70)
        print("Nenhum arquivo encontrado em data/raw/!")
        print("=" * 70)
        return

    print("=" * 70)
    print(f"Inspecionando arquivo: {filename}")
    print(f"Total de registros na amostra: {len(df)}")
    print(f"Total de colunas: {len(df.columns)}")
    print("=" * 70)

    print("\n1. LISTA DE COLUNAS E TIPOS:")
    print("-" * 70)
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print("\n2. PORCENTAGEM DE NULOS:")
    print("-" * 70)
    null_pct = (df.isnull().sum() / len(df)) * 100
    for col, pct in null_pct.items():
        if pct > 0:
            print(f"  - {col}: {pct:.2f}% nulos")

    print("\n3. EXEMPLO DE JOGOS REAIS NA BASE:")
    print("-" * 70)
    # Seleciona jogos conhecidos com bastante reviews para demonstrar
    rev_col = "positive" if "positive" in df.columns else ("Positive" if "Positive" in df.columns else None)
    if rev_col:
        sample_df = df.sort_values(by=rev_col, ascending=False).head(3)
    else:
        sample_df = df.head(3)

    cols_to_show = [
        col for col in [
            "appID", "AppID", "name", "Name", "release_date", "Release date",
            "price", "Price", "positive", "Positive", "negative", "Negative",
            "windows", "Windows", "genres", "Genres", "tags", "Tags",
            "short_description", "detailed_description", "About the game"
        ] if col in df.columns
    ]

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        print(f"\n--- [Jogo de Exemplo {idx + 1}] ---")
        for col in cols_to_show:
            val_str = str(row[col])
            if len(val_str) > 100:
                val_str = val_str[:97] + "..."
            print(f"  {col}: {val_str}")

    print("\n" + "=" * 70)
    print("Inspeção concluída com sucesso.")
    print("=" * 70)


if __name__ == "__main__":
    inspect()
