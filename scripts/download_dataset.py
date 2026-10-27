"""Script para baixar o dataset da Steam (FronkonGames) do Hugging Face diretamente para data/raw/."""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Adiciona a raiz do projeto ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def download_dataset() -> None:
    print("=" * 60)
    print("Baixando dataset FronkonGames/steam-games-dataset...")
    print(f"Destino: {config.RAW_DATA_DIR.resolve()}")
    print("=" * 60)

    try:
        downloaded_path = hf_hub_download(
            repo_id="FronkonGames/steam-games-dataset",
            filename="games.csv",
            repo_type="dataset",
            local_dir=str(config.RAW_DATA_DIR),
        )
        print("\nDownload concluído com sucesso!")
        print(f"Arquivo salvo em: {downloaded_path}")
    except Exception as e:
        print(f"\nErro durante o download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
