# GameFinder RAG 🎮🔍

Recomendador de jogos da Steam a partir de pedidos em linguagem natural, utilizando LLMs e RAG (Retrieval-Augmented Generation) com busca semântica sobre dataset público da Steam.

Projeto desenvolvido para a disciplina **TEIA (1ª Unidade) - IFPE**.

---

## 🎯 Objetivos do Projeto
- **Busca Semântica:** Permitir que o usuário faça pedidos em linguagem natural (ex.: *"jogo cooperativo curto e relaxante para jogar com amigos, até R$ 50"*).
- **Extração de Filtros com LLM:** Identificar preço máximo, plataformas (Windows, Mac, Linux), modos (coop, multiplayer) e ano de lançamento.
- **RAG com ChromaDB:** Recuperação vetorial de jogos no banco vetorial utilizando embeddings (`nomic-embed-text` e `bge-m3`).
- **Prevenção de Alucinação:** LLM recomenda e justifica apenas os jogos recuperados no contexto.
- **Interface Interativa (Gradio):** Demonstração didática dos conceitos de tokenização, espaço vetorial (mapa UMAP 2D), temperatura e comparação Com RAG vs Sem RAG.

---

## 🏗️ Estrutura do Projeto

```text
gamefinder-rag/
├── app.py                 # Interface Gradio (ponto de entrada)
├── requirements.txt       # Dependências do projeto
├── .env.example           # Modelo de configuração de ambiente
├── .env                   # Configurações locais (ignorado no git)
├── .gitignore
├── README.md
├── data/
│   ├── raw/               # CSV original do Kaggle (ignorado no git)
│   ├── processed/         # games.parquet limpo
│   └── chroma/            # Banco vetorial ChromaDB (ignorado no git)
├── src/
│   ├── config.py          # Configurações centralizadas (.env)
│   ├── data_prep.py       # Limpeza e preparação do dataset
│   ├── indexer.py         # Geração de embeddings e indexação
│   ├── tokens.py          # Contagem e inspeção de tokens
│   ├── llm.py             # Cliente de LLM unificado (Ollama / APIs)
│   ├── query_parser.py    # Extração de filtros via LLM
│   ├── retriever.py       # Busca vetorial + filtros
│   ├── recommender.py     # Pipeline RAG completo
│   ├── similar.py         # "Jogos parecidos com X"
│   ├── viz.py             # Projeção UMAP 2D com Plotly
│   └── prompts/           # Prompts versionados (.txt)
├── eval/
│   ├── queries.json       # Pedidos de teste com jogos esperados
│   ├── evaluate.py        # Benchmark hit@5: Sem RAG vs RAG (A e B)
│   └── results.md         # Tabela de resultados
└── scripts/
    ├── smoke_test.py      # Teste rápido do LLM e embeddings
    └── inspect_data.py    # Inspeção inicial do dataset bruto
```

---

## 🚀 Como Executar

### 1. Pré-requisitos
- **Python 3.10 a 3.12**
- **Ollama** ([ollama.com](https://ollama.com)) instalado com os modelos:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull nomic-embed-text
  ollama pull bge-m3
  ```

### 2. Configuração do Ambiente
```bash
# Clone o repositório
git clone https://github.com/viniscastro/TEIA.git
cd TEIA

# Crie e ative o ambiente virtual
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
```

### 3. Validação Inicial
Execute o teste de conectividade:
```bash
python scripts/smoke_test.py
```
