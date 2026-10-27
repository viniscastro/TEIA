"""Configurações centralizadas do GameFinder RAG.

Carrega variáveis de ambiente do arquivo .env e expõe constantes tipadas
e caminhos de diretórios do projeto.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Carrega o .env da raiz do projeto
load_dotenv(ROOT_DIR / ".env")

# Provedor e modelos
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL_A: str = os.getenv("EMBED_MODEL_A", "nomic-embed-text")
EMBED_MODEL_B: str = os.getenv("EMBED_MODEL_B", "bge-m3")

# Tokens / APIs externas
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

# Configurações do servidor web (Gradio)
SERVER_NAME: str = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))

# Diretórios padrão
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CHROMA_DIR: Path = DATA_DIR / "chroma"
PROMPTS_DIR: Path = ROOT_DIR / "src" / "prompts"
EVAL_DIR: Path = ROOT_DIR / "eval"

# Garante que os diretórios existam
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
