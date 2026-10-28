"""Interface Gradio do GameFinder RAG (Fase 7)."""

import json
import os
import sys
from pathlib import Path

import gradio as gr
import plotly.express as px
import pandas as pd

# Adiciona raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config
from src.indexer import COLLECTION_BGE, COLLECTION_NOMIC
from src.llm import chat, embed
from src.recommender import recommend
from src.similar import find_similar, get_all_games_for_dropdown
from src.tokens import count_tokens, tokenize_preview, stats_doc_text

# Carrega dados para dropdown
try:
    games_dropdown = get_all_games_for_dropdown(COLLECTION_NOMIC)
    if not games_dropdown:
        games_dropdown = get_all_games_for_dropdown(COLLECTION_BGE)
except Exception as e:
    print(f"Erro ao carregar jogos para dropdown: {e}")
    games_dropdown = []


def recomendar(
    query,
    preco_max_slider,
    plataforma,
    embed_model_choice,
    temperatura,
    usar_rag,
):
    if not query or not query.strip():
        return "<p>Digite um pedido para receber recomendações.</p>", "", "", "", ""

    # Mapear escolha de modelo
    if embed_model_choice == "A - nomic-embed-text":
        collection = COLLECTION_NOMIC
    else:
        collection = COLLECTION_BGE

    # Ajustar parsed query com slider de preço se fornecido
    try:
        res = recommend(
            user_query=query.strip(),
            collection_name=collection,
            use_rag=usar_rag,
            temperature=float(temperatura),
            k_candidates=20,
            max_results=5,
        )
    except Exception as e:
        return f"<p>Erro: {e}</p>", "", "", "", ""

    # Montar cards HTML
    cards_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;'>"
    if not res["recommendations"]:
        cards_html += "<p>Nenhuma recomendação válida encontrada.</p>"
    else:
        for r in res["recommendations"]:
            app_id = r["app_id"]
            name = r.get("name", "Nome não disponível")
            price = r.get("price", 0.0)
            tags = r.get("tags", "")
            motivo = r.get("motivo", "")
            steam_url = f"https://store.steampowered.com/app/{app_id}"
            cards_html += f"""
            <div style='border: 1px solid #333; border-radius: 8px; padding: 16px; background: #1e1e1e;'>
                <h3 style='margin: 0 0 8px 0;'>{name}</h3>
                <p style='margin: 4px 0;'><strong>Preço:</strong> ${price:.2f}</p>
                <p style='margin: 4px 0;'><strong>Tags:</strong> {tags[:100]}</p>
                <p style='margin: 8px 0;'><strong>Motivo:</strong> {motivo}</p>
                <a href='{steam_url}' target='_blank' style='color: #66c0f4;'>Ver na Steam →</a>
            </div>
            """
    cards_html += "</div>"

    # JSON de filtros
    filtros_json = json.dumps(res["parsed"], indent=2, ensure_ascii=False)

    # Candidatos recuperados
    candidatos_txt = ""
    if res.get("retrieval") and res["retrieval"].get("candidates"):
        for c in res["retrieval"]["candidates"][:10]:
            meta = c.get("metadata", {})
            dist = c.get("distance")
            candidatos_txt += f"- {meta.get('name')} (ID: {c['app_id']}) | Dist: {dist:.4f} | ${meta.get('price'):.2f}\n"
        if len(res["retrieval"]["candidates"]) > 10:
            candidatos_txt += f"\n... e mais {len(res['retrieval']['candidates']) - 10} candidatos\n"
    if res.get("retrieval", {}).get("relaxed"):
        candidatos_txt += f"\n⚠️ {res['retrieval'].get('message', 'Filtros relaxados')}\n"

    # Métricas
    metricas = f"Tokens do contexto: {res['context_tokens']}\n"
    metricas += f"Parser: {res['timings'].get('parser', 0):.3f}s | "
    metricas += f"Retriever: {res['timings'].get('retriever', 0):.3f}s | "
    metricas += f"LLM: {res['timings'].get('llm_generation', 0):.3f}s | "
    metricas += f"Total: {res['timings'].get('total', 0):.3f}s\n"
    metricas += f"Modo RAG: {res['use_rag']} | Coleção: {res['collection_name']}"

    return cards_html, filtros_json, candidatos_txt, metricas, ""


def buscar_similares(jogo_selecionado, embed_model_choice_sim):
    if not jogo_selecionado:
        return "<p>Selecione um jogo.</p>"

    app_id = None
    try:
        if isinstance(jogo_selecionado, dict):
            app_id = jogo_selecionado.get("value") or jogo_selecionado.get("label")
        else:
            val = str(jogo_selecionado)
            if "(" in val and ")" in val:
                app_id = val.split("(")[-1].replace(")", "").strip()
            else:
                app_id = val.strip()
    except Exception:
        app_id = str(jogo_selecionado).strip()

    if not app_id:
        return "<p>Erro: Não foi possível extrair o ID do jogo.</p>"

    if embed_model_choice_sim == "A - nomic-embed-text":
        collection = COLLECTION_NOMIC
    else:
        collection = COLLECTION_BGE

    try:
        similares = find_similar(app_id, collection_name=collection, k=8)
    except Exception as e:
        return f"<p>Erro: {e}</p>"

    if not similares:
        return "<p>Nenhum jogo similar encontrado.</p>"

    cards_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px;'>"
    for s in similares:
        meta = s.get("metadata", {}) or {}
        name = meta.get("name", "N/A")
        price = meta.get("price", 0.0)
        dist = s.get("distance")
        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
        app_id_s = s["app_id"]
        steam_url = f"https://store.steampowered.com/app/{app_id_s}"
        cards_html += f"""
        <div style='border: 1px solid #333; border-radius: 8px; padding: 12px; background: #1e1e1e;'>
            <h4 style='margin: 0 0 6px 0;'>{name}</h4>
            <p style='margin: 2px 0; font-size: 0.9em;'>Preço: ${price:.2f} | Dist: {dist_str}</p>
            <a href='{steam_url}' target='_blank' style='color: #66c0f4; font-size: 0.9em;'>Ver na Steam →</a>
        </div>
        """
    cards_html += "</div>"
    return cards_html


def tokenizar_texto(texto):
    if not texto:
        return "Digite um texto para visualizar a tokenização.", ""
    tokens = tokenize_preview(texto)
    html = "<div style='display: flex; flex-wrap: wrap; gap: 4px;'>"
    for t in tokens:
        html += f"<span style='background: #4a4a4a; padding: 2px 6px; border-radius: 4px; font-family: monospace;' title='ID: {t['id']}'>{t['token']}</span>"
    html += "</div>"
    info = f"Total de tokens: {len(tokens)} (cl100k_base)"
    return html, info


def comparar_embeddings(frase1, frase2, modelo):
    if not frase1 or not frase2:
        return "Digite ambas as frases."
    try:
        if modelo == "A - nomic-embed-text":
            v1 = embed([frase1], model=config.EMBED_MODEL_A)[0]
            v2 = embed([frase2], model=config.EMBED_MODEL_A)[0]
        else:
            v1 = embed([frase1], model=config.EMBED_MODEL_B)[0]
            v2 = embed([frase2], model=config.EMBED_MODEL_B)[0]
        # Cosine similarity
        import numpy as np
        v1n = np.array(v1) / np.linalg.norm(v1)
        v2n = np.array(v2) / np.linalg.norm(v2)
        sim = np.dot(v1n, v2n)
        return f"Similaridade de Cosseno ({modelo}): {sim:.4f}"
    except Exception as e:
        return f"Erro: {e}"


def testar_temperatura(query):
    if not query:
        return "Digite um pedido.", "Digite um pedido."
    try:
        messages = [{"role": "user", "content": f"Recomende 2 jogos para: {query}. Responda em português."}]
        r1 = chat(messages, temperature=0.0)
        r2 = chat(messages, temperature=1.2)
        return r1, r2
    except Exception as e:
        return f"Erro: {e}", f"Erro: {e}"


# Interface Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="GameFinder RAG") as demo:
    gr.Markdown("# 🎮 GameFinder RAG - Recomendador de Jogos Steam")
    gr.Markdown("Busca semântica com RAG, filtros extraídos por LLM e prevenção de alucinações")

    with gr.Tabs():
        # ABA 1: Recomendar
        with gr.Tab("🎮 Recomendar"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Seu pedido",
                        placeholder="Ex: jogo cooperativo relaxante para jogar com amigos até R$ 50",
                        lines=2,
                    )
                    with gr.Accordion("Opções Avançadas", open=False):
                        preco_slider = gr.Slider(0, 200, value=100, label="Preço máximo (referência)")
                        plataforma_dd = gr.Dropdown(["Todas", "windows", "mac", "linux"], value="Todas", label="Plataforma")
                        embed_choice = gr.Radio(["A - nomic-embed-text", "B - bge-m3"], value="A - nomic-embed-text", label="Modelo de Embedding")
                        temp_slider = gr.Slider(0.0, 1.5, value=0.2, step=0.1, label="Temperatura do LLM")
                        usar_rag_cb = gr.Checkbox(value=True, label="Usar RAG (contexto vetorial)")
                    btn_recomendar = gr.Button("Buscar Recomendações", variant="primary")
                with gr.Column(scale=1):
                    metricas_out = gr.Textbox(label="Métricas", lines=6, interactive=False)

            cards_out = gr.HTML(label="Recomendações")
            
            with gr.Accordion("Por baixo dos panos", open=False):
                with gr.Row():
                    filtros_out = gr.Code(label="Filtros extraídos (JSON)", language="json")
                    candidatos_out = gr.Textbox(label="Candidatos recuperados (top 10)", lines=12, interactive=False)
                debug_out = gr.Textbox(label="Debug", visible=False)

            btn_recomendar.click(
                recomendar,
                inputs=[query_input, preco_slider, plataforma_dd, embed_choice, temp_slider, usar_rag_cb],
                outputs=[cards_out, filtros_out, candidatos_out, metricas_out, debug_out],
            )

        # ABA 2: Parecidos com...
        with gr.Tab("🔗 Parecidos com..."):
            gr.Markdown("Encontre jogos vetorialmente similares a um jogo escolhido")
            with gr.Row():
                if games_dropdown:
                    jogo_dropdown = gr.Dropdown(
                        choices=games_dropdown,
                        label="Selecione um jogo",
                        interactive=True,
                        value=games_dropdown[0]["value"] if games_dropdown else None,
                    )
                else:
                    jogo_dropdown = gr.Dropdown(
                        choices=[],
                        label="Selecione um jogo",
                        interactive=True,
                    )
                embed_sim = gr.Radio(["A - nomic-embed-text", "B - bge-m3"], value="A - nomic-embed-text", label="Modelo")
            btn_similares = gr.Button("Buscar Similares", variant="primary")
            similares_out = gr.HTML(label="Jogas Similares")

            btn_similares.click(
                buscar_similares,
                inputs=[jogo_dropdown, embed_sim],
                outputs=[similares_out],
            )

        # ABA 3: Como Funciona
        with gr.Tab("💡 Como Funciona"):
            gr.Markdown("## Painel Didático")
            
            with gr.Accordion("Tokenizador Interativo (cl100k_base)", open=True):
                token_input = gr.Textbox(label="Digite um texto", placeholder="Ex: jogo cooperativo relaxante")
                btn_tokenizar = gr.Button("Tokenizar")
                token_html = gr.HTML()
                token_info = gr.Textbox(label="Informação", interactive=False)
                btn_tokenizar.click(tokenizar_texto, inputs=[token_input], outputs=[token_html, token_info])

            with gr.Accordion("Comparador de Embeddings (Similaridade de Cosseno)", open=True):
                with gr.Row():
                    frase1 = gr.Textbox(label="Frase 1", placeholder="Hollow Knight é um metroidvania desafiador")
                    frase2 = gr.Textbox(label="Frase 2", placeholder="Celeste é um platformer difícil e recompensador")
                modelo_emb = gr.Radio(["A - nomic-embed-text", "B - bge-m3"], value="A - nomic-embed-text")
                btn_comp = gr.Button("Comparar")
                comp_out = gr.Textbox(label="Resultado", interactive=False)
                btn_comp.click(comparar_embeddings, inputs=[frase1, frase2, modelo_emb], outputs=[comp_out])

            with gr.Accordion("Efeito da Temperatura", open=True):
                temp_query = gr.Textbox(label="Pedido", placeholder="Recomende jogos de fazenda relaxantes")
                btn_temp = gr.Button("Gerar com temperaturas diferentes")
                with gr.Row():
                    temp0 = gr.Textbox(label="Temperatura = 0.0 (Determinística)", lines=6, interactive=False)
                    temp12 = gr.Textbox(label="Temperatura = 1.2 (Criativa)", lines=6, interactive=False)
                btn_temp.click(testar_temperatura, inputs=[temp_query], outputs=[temp0, temp12])

            gr.Markdown("""
            ## Conceitos Didáticos
            
            **Tokenização (BPE)**: O modelo divide o texto em unidades menores (tokens). Usa-se `cl100k_base` (compatível com modelos modernos GPT).
            
            **Embeddings**: Representação vetorial numérica do significado textual em alta dimensão (768 ou 1024 dimensões). Jogos com significado similar ficam próximos no espaço vetorial.
            
            **Busca Semântica**: Ao invés de buscar por palavras exatas, busca por similaridade de cosseno entre vetores.
            
            **RAG (Retrieval-Augmented Generation)**: Combina busca vetorial com LLM. O contexto recuperado é fornecido ao modelo para evitar alucinações.
            
            **Temperatura**: Controla a aleatoriedade da geração. Valores baixos (0.0) geram respostas mais determinísticas. Valores altos aumentam a criatividade, mas também o risco de alucinações.
            """)

        # ABA 4: Mapa Vetorial 2D
        with gr.Tab("🗺️ Mapa Vetorial 2D"):
            gr.Markdown("## Projeção UMAP do Espaço Vetorial")
            gr.Markdown("Visualização 2D dos embeddings. Pontos próximos representam jogos semanticamente similares.")
            try:
                # Tentar carregar dados UMAP se existirem
                umap_path_a = config.PROCESSED_DATA_DIR / "umap_games_nomic.parquet"
                umap_path_b = config.PROCESSED_DATA_DIR / "umap_games_bge.parquet"
                if umap_path_a.exists():
                    df_umap = pd.read_parquet(umap_path_a)
                    if len(df_umap) > 3000:
                        df_umap = df_umap.sample(3000, random_state=42)
                    fig = px.scatter(
                        df_umap,
                        x="x",
                        y="y",
                        hover_name="name",
                        hover_data=["price", "tags"],
                        color="genre" if "genre" in df_umap.columns else None,
                        title="Mapa Vetorial 2D - Coleção Nomic",
                    )
                    plot_out = gr.Plot(value=fig)
                else:
                    plot_out = gr.Markdown("*Mapa UMAP não gerado ainda. Execute a Fase 8 para gerar as projeções.*")
            except Exception as e:
                plot_out = gr.Markdown(f"*Erro ao carregar mapa: {e}*")
            plot_out

# Lançar app
if __name__ == "__main__":
    demo.launch(server_name=config.SERVER_NAME, server_port=config.SERVER_PORT, share=False)
