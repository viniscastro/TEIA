"""Script para inspecionar o arquivo bruto em data/raw/ (CSV ou JSON).

Exibe colunas, tipos de dados, 3 linhas de exemplo e porcentagem de nulos por coluna.
"""

import sys
from pathlib import Path
import pandas as pd

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def inspect() -> None:
    raw_dir = config.RAW_DATA_DIR
    files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.json"))

    if not files:
        print("=" * 70)
        print("Nenhum arquivo CSV ou JSON encontrado em data/raw/!")
        print("Por favor, baixe um Steam Games Dataset (Kaggle) e salve o arquivo em:")
        print(f"  {raw_dir.resolve()}")
        print("=" * 70)
        return

    file_path = files[0]
    print("=" * 70)
    print(f"Inspecionando arquivo: {file_path.name}")
    print(f"Caminho completo:     {file_path.resolve()}")
    print("=" * 70)

    try:
        if file_path.suffix.lower() == ".csv":
            # Tenta ler com utf-8, fallback para latin-1 se necessário
            try:
                df = pd.read_csv(file_path, nrows=5000)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1", nrows=5000)
        else:
            df = pd.read_json(file_path)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    print(f"\nDimensões da amostra lida: {df.shape[0]} linhas x {df.shape[1]} colunas\n")

    print("-" * 70)
    print("1. LISTA DE COLUNAS E TIPOS:")
    print("-" * 70)
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print("\n" + "-" * 70)
    print("2. PORCENTAGEM DE VALORES NULOS:")
    print("-" * 70)
    null_pct = (df.isnull().sum() / len(df)) * 100
    for col, pct in null_pct.items():
        print(f"  - {col}: {pct:.2f}% nulos")

    print("\n" + "-" * 70)
    print("3. EXEMPLO DE 3 LINHAS (Amostra):")
    print("-" * 70)
    sample_df = df.head(3)
    for idx, row in sample_df.iterrows():
        print(f"\n--- [Linha {idx + 1}] ---")
        for col in df.columns:
            val_str = str(row[col])
            if len(val_str) > 120:
                val_str = val_str[:117] + "..."
            print(f"  {col}: {val_str}")

    print("\n" + "=" * 70)
    print("Inspeção concluída com sucesso.")
    print("=" * 70)


if __name__ == "__main__":
    inspect()
