"""
Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
Entrega 4 — app Streamlit
Aluno: Pedro Ernesto

Executar:  streamlit run app_ouvidoria.py
"""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# configuração geral
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

MODELOS = {
    "MiniLM-L12 (rápido, 384d)": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet-base (preciso, 768d)": "paraphrase-multilingual-mpnet-base-v2",
}
CORES_CATEGORIA = {
    "infraestrutura": "#4C78A8", "saúde": "#E45756", "segurança": "#F58518",
    "educação": "#72B7B2", "meio ambiente": "#54A24B",
}


def cor_score(s: float) -> str:
    return "🟢" if s > 0.7 else ("🟡" if s > 0.5 else "🔴")


@st.cache_data
def carregar_dados() -> pd.DataFrame:
    df = pd.read_json("manifestacoes.json")
    df["n_chars"] = df["texto"].str.len()
    return df


@st.cache_resource(show_spinner="Carregando modelo de embedding…")
def carregar_modelo(nome: str) -> SentenceTransformer:
    return SentenceTransformer(nome)


@st.cache_data(show_spinner="Calculando embeddings…")
def embeddings_base(nome_modelo: str, textos: tuple) -> np.ndarray:
    modelo = carregar_modelo(nome_modelo)
    return modelo.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)


def codificar(nome_modelo: str, textos: list) -> np.ndarray:
    return carregar_modelo(nome_modelo).encode(textos, normalize_embeddings=True, show_progress_bar=False)


@st.cache_data(show_spinner=False)
def projetar_2d(E: np.ndarray, metodo: str, perplexidade: int) -> np.ndarray:
    if metodo == "PCA":
        return PCA(n_components=2, random_state=0).fit_transform(E)
    perp = max(2, min(perplexidade, len(E) - 1))
    return TSNE(n_components=2, perplexity=perp, random_state=0, init="pca").fit_transform(E)


# ──────────────────────────────────────────────────────────────────────────────
# sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs")
    rotulo_modelo = st.selectbox("Modelo de embedding", list(MODELOS))
    nome_modelo = MODELOS[rotulo_modelo]
    top_k = st.slider("Top-k resultados", min_value=1, max_value=15, value=5)
    st.divider()
    st.markdown(
        "**Legenda de score**\n\n🟢 &gt; 0.70 muito similar  \n🟡 &gt; 0.50 relacionado  \n🔴 ≤ 0.50 pouco relacionado",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Aluno: Pedro Ernesto")

df = carregar_dados()
ids = df["id"].tolist()
E = embeddings_base(nome_modelo, tuple(df["texto"]))

aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)

# ──────────────────────────────────────────────────────────────────────────────
# 🔍 Busca Semântica
# ──────────────────────────────────────────────────────────────────────────────
with aba_busca:
    st.subheader("Buscar manifestações parecidas")
    st.write(
        "Descreva um problema com suas palavras. O sistema compara o significado do texto "
        "com as manifestações já registradas, mesmo que as palavras sejam diferentes."
    )
    exemplos = {
        "(escolha um exemplo ou escreva)": "",
        "rua com buraco": "tem um buraco grande na rua e os carros estão quebrando",
        "posto sem médico": "fui ao posto de saúde e não tinha médico para me atender",
        "lixo": "o lixo não está sendo recolhido na minha rua",
        "escola": "meu filho está sem professor na escola",
    }
    exemplo = st.selectbox("Exemplos rápidos", list(exemplos), label_visibility="collapsed")
    consulta = st.text_area("Sua manifestação", value=exemplos[exemplo], height=100,
                            placeholder="Ex.: a iluminação da minha rua está apagada há semanas…")
    col1, col2 = st.columns([1, 3])
    filtro_cat = col1.multiselect("Filtrar por categoria", sorted(df["categoria_oficial"].unique()))
    buscar = col2.button("Buscar", type="primary", width="stretch")

    if buscar and consulta.strip():
        q = codificar(nome_modelo, [consulta])
        scores = cosine_similarity(q, E)[0]
        res = df.assign(score=scores)
        if filtro_cat:
            res = res[res["categoria_oficial"].isin(filtro_cat)]
        res = res.sort_values("score", ascending=False).head(top_k)

        melhor = res.iloc[0]["score"] if len(res) else 0
        if melhor > 0.7:
            st.success(f"Provável duplicata: existe manifestação muito parecida (score {melhor:.2f}).")
        elif melhor > 0.5:
            st.info(f"Há manifestações relacionadas (melhor score {melhor:.2f}).")
        else:
            st.warning("Nenhuma manifestação parecida encontrada. Parece ser um problema novo.")

        for _, r in res.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([1, 6])
                c1.markdown(f"### {cor_score(r['score'])} {r['score']:.2f}")
                c1.caption(f"{r['id']} · {r['data']}")
                c1.markdown(
                    f"<span style='background:{CORES_CATEGORIA[r['categoria_oficial']]};color:white;"
                    f"padding:2px 8px;border-radius:10px;font-size:0.8em'>{r['categoria_oficial']}</span>",
                    unsafe_allow_html=True,
                )
                c2.write(r["texto"])
    elif buscar:
        st.warning("Escreva alguma coisa antes de buscar.")

# ──────────────────────────────────────────────────────────────────────────────
# 📋 Base Completa
# ──────────────────────────────────────────────────────────────────────────────
with aba_base:
    st.subheader("Todas as manifestações")
    c1, c2, c3 = st.columns(3)
    c1.metric("Manifestações", len(df))
    c2.metric("Categorias", df["categoria_oficial"].nunique())
    c3.metric("Textos longos (> 500 chars)", int((df["n_chars"] > 500).sum()))

    st.dataframe(
        df[["id", "data", "categoria_oficial", "n_chars", "texto"]],
        width="stretch", hide_index=True,
        column_config={"texto": st.column_config.TextColumn(width="large"), "n_chars": "chars"},
    )

    st.divider()
    cA, cB = st.columns([1, 2])
    limiar = cA.slider("Limiar de duplicata", 0.50, 0.95, 0.85, 0.01)
    if cA.button("Gerar matriz de similaridade", type="primary"):
        S = cosine_similarity(E)
        fig = go.Figure(data=go.Heatmap(
            z=S, x=ids, y=ids, colorscale="Magma", zmin=0, zmax=1,
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(height=700, width=760, yaxis_autorange="reversed",
                          title=f"Matriz de similaridade — {rotulo_modelo}")
        st.plotly_chart(fig, width="content")

        pares = [(ids[i], ids[j], S[i, j]) for i, j in combinations(range(len(ids)), 2) if S[i, j] >= limiar]
        pares = pd.DataFrame(pares, columns=["id_a", "id_b", "similaridade"]).sort_values("similaridade", ascending=False)
        st.markdown(f"**{len(pares)} pares com similaridade ≥ {limiar:.2f}** (candidatos a duplicata)")
        texto_de = dict(zip(ids, df["texto"]))
        pares["texto_a"] = pares["id_a"].map(texto_de).str[:90] + "…"
        pares["texto_b"] = pares["id_b"].map(texto_de).str[:90] + "…"
        st.dataframe(pares.round(3), width="stretch", hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🌐 Espaço Vetorial
# ──────────────────────────────────────────────────────────────────────────────
with aba_espaco:
    st.subheader("Mapa semântico das manifestações")
    c1, c2 = st.columns([1, 1])
    metodo = c1.radio("Redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)
    perp = c2.slider("Perplexidade (t-SNE)", 2, 20, 8, disabled=(metodo == "PCA"))

    P = projetar_2d(E, metodo, perp)
    plot_df = df.assign(x=P[:, 0], y=P[:, 1], resumo=df["texto"].str[:80] + "…")
    fig = px.scatter(
        plot_df, x="x", y="y", color="categoria_oficial", color_discrete_map=CORES_CATEGORIA,
        hover_data={"id": True, "resumo": True, "x": False, "y": False}, text="id",
        title=f"{metodo} dos embeddings ({rotulo_modelo}) — cor = categoria oficial",
    )
    fig.update_traces(textposition="top center", marker=dict(size=11), textfont_size=9)
    fig.update_layout(height=600, legend_title="categoria oficial")
    st.plotly_chart(fig, width="stretch")

    # medida quantitativa de coincidência entre clusters semânticos e categorias oficiais
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(E, df["categoria_oficial"], metric="cosine")
    st.metric("Silhueta por categoria oficial (espaço de embedding)", f"{sil:.3f}",
              help="Perto de 1: clusters semânticos coincidem com as categorias. Perto de 0: misturados.")

    with st.expander("Os clusters semânticos coincidem com as categorias oficiais? (comentário)", expanded=True):
        st.markdown(
            """
Em parte. Os embeddings agrupam manifestações pelo **assunto linguístico**, e as categorias oficiais seguem
a **divisão administrativa** da prefeitura; as duas nem sempre batem:

- **Saúde** e **educação** formam os grupos mais coesos: o vocabulário (posto, médico, consulta, remédio /
  escola, professor, aula, creche) é específico e pouco compartilhado com outras áreas.
- **Infraestrutura** e **segurança** se misturam: iluminação pública apagada é registrada como infraestrutura
  (M031, lâmpada da praça) ou como segurança (M005, M028, rua escura e assaltos) dependendo de como o
  cidadão enquadra o problema. Para o modelo é o mesmo assunto.
- **Meio ambiente** se espalha: lixo acumulado (M011, M035) fica perto de infraestrutura; queima de lixo com
  criança asmática (M023) e fábrica poluidora (M040) puxam para saúde.
- As manifestações **longas** ficam mais ao centro do mapa, porque tratam de vários temas ao mesmo tempo e o
  embedding vira uma "média" (motivo do chunking na aba seguinte).

A silhueta acima (baixa, mas positiva) quantifica isso: existe estrutura por categoria, mas com fronteiras
borradas. Para a triagem, o ganho prático é que o mapa revela **subtemas** que a categoria oficial esconde
(por exemplo, "falta de profissional" aparece tanto em saúde quanto em educação).
            """
        )

# ──────────────────────────────────────────────────────────────────────────────
# 🧩 Chunking
# ──────────────────────────────────────────────────────────────────────────────
with aba_chunk:
    st.subheader("Fatiar uma manifestação longa em chunks")
    longa_exemplo = df.sort_values("n_chars", ascending=False).iloc[0]["texto"]
    texto_longo = st.text_area("Cole uma manifestação longa", value=longa_exemplo, height=180)

    c1, c2, c3 = st.columns(3)
    estrategia = c1.selectbox("Estratégia", ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"])
    chunk_size = c2.slider("chunk_size", 50, 600, 200, 10)
    chunk_overlap = c3.slider("chunk_overlap", 0, 200, 50, 5)
    if chunk_overlap >= chunk_size:
        st.error("chunk_overlap precisa ser menor que chunk_size.")
        st.stop()

    if estrategia == "RecursiveCharacterTextSplitter":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
    else:
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")

    pedacos = splitter.split_text(texto_longo) if texto_longo.strip() else []
    if not pedacos:
        st.info("Cole um texto para ver os chunks.")
    else:
        st.markdown(f"**{len(pedacos)} chunks** gerados ({len(texto_longo)} caracteres no total)")
        E_c = codificar(nome_modelo, pedacos)
        E_doc = codificar(nome_modelo, [texto_longo])[0]
        fid = E_c @ E_doc
        coesao = [float(E_c[i] @ E_c[i + 1]) for i in range(len(E_c) - 1)]

        m1, m2 = st.columns(2)
        m1.metric("Fidelidade média chunk ↔ documento", f"{fid.mean():.3f}")
        m2.metric("Coesão média entre chunks consecutivos", f"{np.mean(coesao):.3f}" if coesao else "—")

        for i, (p, f) in enumerate(zip(pedacos, fid)):
            with st.container(border=True):
                a, b = st.columns([1, 7])
                a.markdown(f"**chunk {i}**")
                a.caption(f"{len(p)} chars")
                a.caption(f"fid. {f:.2f}")
                b.write(p)

        st.markdown("#### Embeddings dos chunks")
        st.caption("Cada linha é um chunk; as colunas são as primeiras dimensões do vetor. "
                   "Abaixo, a similaridade entre todos os chunks.")
        n_dim = min(24, E_c.shape[1])
        st.dataframe(pd.DataFrame(E_c[:, :n_dim], index=[f"chunk {i}" for i in range(len(pedacos))]).round(3),
                     width="stretch")
        S_c = cosine_similarity(E_c)
        figc = go.Figure(data=go.Heatmap(
            z=S_c, x=[f"c{i}" for i in range(len(pedacos))], y=[f"c{i}" for i in range(len(pedacos))],
            colorscale="Viridis", zmin=0, zmax=1, text=np.round(S_c, 2), texttemplate="%{text}",
        ))
        figc.update_layout(height=380, width=480, yaxis_autorange="reversed", title="Similaridade entre chunks")
        st.plotly_chart(figc, width="content")
