"""
Gera RELATORIO.pdf (até 5 páginas) a partir dos arquivos em resultados/ produzidos pelos notebooks.

Uso:
    python gerar_relatorio.py

Se algum notebook ainda não foi executado, os números correspondentes aparecem como
"[rodar notebook N]" no PDF. Os valores de BoW e TF-IDF são recalculados aqui mesmo,
porque não dependem de modelo baixado.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALUNO = "Pedro Ernesto"
R = Path("resultados")


def carregar(nome):
    p = R / nome
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


e1, e2, e3 = carregar("entrega1.json"), carregar("entrega2.json"), carregar("entrega3.json")


def v(obj, *chaves, fmt="{:.3f}", falta="[rodar notebook]"):
    """Navega em dicionários aninhados; devolve valor formatado ou marcador de ausência."""
    try:
        for k in chaves:
            obj = obj[k]
        return fmt.format(obj) if isinstance(obj, (int, float)) else str(obj)
    except (KeyError, TypeError, IndexError):
        return falta


# ── BoW / TF-IDF recalculados localmente (mesmo pré-processamento do notebook 1) ──
df = pd.read_json("manifestacoes.json")
STOP = set("a o os as um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre e ou mas que se ao aos à às pelo pela pelos pelas este esta isto esse essa isso aquele aquela aquilo eu tu ele ela nós vós eles elas me te lhe nos vos lhes meu minha meus minhas seu sua seus suas nosso nossa nossos nossas dele dela deles delas já não sim mais menos muito muitos muita muitas pouco pouca todo toda todos todas outro outra outros outras mesmo mesma ser estar ter haver é são está estão foi foram era eram tem têm há tinha tinham ser sendo sido como quando onde porque porquê qual quais quem cada também ainda até desde então aqui ali lá aí só apenas depois antes agora hoje ontem sempre nunca ha faz fazem há vem vai".split())


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(w for w in t.split() if w not in STOP and len(w) > 1)


tn = df["texto"].apply(norm)
S_bow = cosine_similarity(CountVectorizer().fit_transform(tn))
S_tfidf = cosine_similarity(TfidfVectorizer(sublinear_tf=True).fit_transform(tn))
idx = {m: i for i, m in enumerate(df["id"])}
PARES = [("M003", "M017"), ("M008", "M022"), ("M008", "M031")]


def emb(par, modelo):
    """Similaridade de embedding de um par do enunciado (notebook 1), com 2 casas."""
    return v(e1, "pares", par, modelo, fmt="{:.2f}", falta="[nb 1]")


def sim_gab(a, b):
    """Similaridade MiniLM de um par do gabarito (notebook 2), com 2 casas."""
    return v(e2, "sims_gabarito", f"{a}-{b}", fmt="{:.2f}", falta="[nb 2]")

# ── estilos ──────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
base = ParagraphStyle("base", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=2)
titulo = ParagraphStyle("titulo", parent=base, fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=1, spaceAfter=2)
sub = ParagraphStyle("sub", parent=base, fontSize=10, alignment=1, textColor=colors.HexColor("#555555"), spaceAfter=10)
cel = ParagraphStyle("cel", parent=base, fontSize=8.4, leading=10.2, alignment=0, spaceAfter=0)


def tabela(dados, larguras, cabecalho=True):
    dados = [[Paragraph(str(c), cel) for c in linha] for linha in dados]
    t = Table(dados, colWidths=larguras, hAlign="LEFT")
    est = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
           ("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if cabecalho:
        est += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5"))]
    t.setStyle(TableStyle(est))
    return t


P = lambda s, st=base: Paragraph(s, st)
story = []

# ── cabeçalho ────────────────────────────────────────────────────────────────
story += [P("Ouvidoria Inteligente: Triagem Semântica de Manifestações Cidadãs", titulo),
          P(f"Relatório técnico · Desafio prático de NLP aplicado · {ALUNO}", sub)]

# ── 1. contexto ──────────────────────────────────────────────────────────────
story += [P("1. Contexto e objetivo", h1), P(
    "A Ouvidoria de um município de médio porte recebe cerca de 4.000 manifestações por mês em texto livre e faz a triagem "
    "por palavras-chave. Isso gera três falhas: duplicatas com vocabulário diferente não são detectadas, temas relacionados "
    "não são agrupados e manifestações longas perdem contexto ao serem indexadas como um bloco só. Este trabalho constrói um "
    "protótipo de triagem semântica baseado em representações vetoriais, entregue em três notebooks e um app Streamlit. "
    "Este relatório resume as decisões tomadas, os resultados, as dificuldades e os aprendizados; os detalhes e gráficos "
    "completos estão nos notebooks.")]

# ── 2. dataset ───────────────────────────────────────────────────────────────
longas = df[df["texto"].str.len() > 500]["id"].tolist()
story += [P("2. Base de dados", h1), P(
    f"O corpus tem 40 manifestações (M001 a M040) distribuídas em cinco categorias oficiais "
    f"({', '.join(f'{c}: {n}' for c, n in df['categoria_oficial'].value_counts().items())}), com textos entre "
    f"{df['texto'].str.len().min()} e {df['texto'].str.len().max()} caracteres. Seis manifestações são duplicatas "
    "semânticas de outra (15% do corpus, seis pares no gabarito <i>duplicatas_gabarito.json</i>) e cinco têm mais de 500 "
    f"caracteres ({', '.join(longas)}), sendo o alvo do chunking. Como o arquivo <i>manifestacoes.json</i> não acompanhou o "
    "enunciado, ele foi reconstruído com o script <i>gerar_dataset.py</i> respeitando todas as restrições descritas, "
    "inclusive os pares M003/M017, M008/M022 e M008/M031 usados na Entrega 1. Os textos simulam manifestações reais de "
    "João Pessoa (bairros, avenidas e equipamentos públicos existentes), com variação de registro entre formal e coloquial.")]

# ── 3. entregas ──────────────────────────────────────────────────────────────
story += [P("3. Decisões técnicas e resultados", h1)]

# 3.1
story += [P("3.1 Entrega 1: comparação de representações", h2), P(
    "Para BoW e TF-IDF apliquei um pré-processamento leve (minúsculas, remoção de acentos e de stopwords do português) para "
    "que conectivos não inflassem a similaridade. Os embeddings recebem o texto original, já que os modelos foram treinados "
    "com linguagem natural. Comparei dois modelos multilíngues de <i>sentence-transformers</i>: "
    "<i>paraphrase-multilingual-MiniLM-L12-v2</i> (384 dimensões, rápido) e <i>paraphrase-multilingual-mpnet-base-v2</i> "
    "(768 dimensões, mais preciso). Os vetores são L2-normalizados, então produto interno = cosseno.")]
linhas = [["Par", "BoW", "TF-IDF", "Emb. MiniLM", "Emb. mpnet", "Tokens em comum (após pré-processamento)"]]
for a, b in PARES:
    chave = f"{a} × {b}"
    tok = e1["tokens_em_comum"].get(chave, "") if e1 else "[rodar notebook 1]"
    linhas.append([chave, f"{S_bow[idx[a], idx[b]]:.3f}", f"{S_tfidf[idx[a], idx[b]]:.3f}",
                   v(e1, "pares", chave, "Emb MiniLM-L12", falta="[nb 1]"), v(e1, "pares", chave, "Emb mpnet-base", falta="[nb 1]"), tok])
story += [tabela(linhas, [2.6 * cm, 1.4 * cm, 1.5 * cm, 2.1 * cm, 2.0 * cm, 7.3 * cm]), Spacer(1, 4)]
story += [P(
    "A leitura é direta: as representações esparsas só enxergam sobreposição lexical. No par M008 × M022 os dois textos "
    "descrevem o mesmo posto sem médico, mas com palavras diferentes (\"posto de saúde\" vs. \"PSF\", \"sem médico\" vs. "
    "\"falta atendimento\"), e a similaridade esparsa depende de poucos tokens coincidentes; no par M008 × M031 ela é "
    "próxima de zero pelo motivo errado (só a palavra genérica \"bairro\" em comum, não porque o modelo entenda que são "
    "assuntos diferentes). Os embeddings colocam os dois pares duplicados bem acima do par sem relação (MiniLM: "
    f"{emb('M003 × M017', 'Emb MiniLM-L12')} e {emb('M008 × M022', 'Emb MiniLM-L12')} contra {emb('M008 × M031', 'Emb MiniLM-L12')}; "
    f"mpnet: {emb('M003 × M017', 'Emb mpnet-base')} e {emb('M008 × M022', 'Emb mpnet-base')} contra {emb('M008 × M031', 'Emb mpnet-base')}), "
    "que é o comportamento desejado para triagem. Ainda assim, M008 × M022 fica abaixo do 0,85 sugerido no enunciado nos dois "
    "modelos, e no MiniLM M022 é apenas o segundo vizinho de M008, atrás de M002; esse par reaparece como falso negativo na "
    "Entrega 2. Na análise global (similaridade média intra vs. inter categoria), a razão intra/inter foi de "
    f"{v(e1, 'global', 'BoW', 'razão intra/inter', fmt='{:.2f}', falta='[nb 1]')} para BoW, "
    f"{v(e1, 'global', 'TF-IDF', 'razão intra/inter', fmt='{:.2f}', falta='[nb 1]')} para TF-IDF, "
    f"{v(e1, 'global', 'Emb MiniLM-L12', 'razão intra/inter', fmt='{:.2f}', falta='[nb 1]')} para MiniLM e "
    f"{v(e1, 'global', 'Emb mpnet-base', 'razão intra/inter', fmt='{:.2f}', falta='[nb 1]')} para mpnet. A razão das "
    "esparsas parece alta, mas vem de um piso quase zero (a maioria dos pares não compartilha token nenhum); nos embeddings "
    "a matriz ordenada por categoria exibe blocos visíveis na diagonal, com mistura entre infraestrutura e segurança.")]
story += [P(
    "<b>Limitações.</b> BoW ignora ordem e semântica e é sensível a tamanho. TF-IDF corrige o peso de termos genéricos, "
    "mas continua lexical e, num corpus de 40 documentos, o IDF é instável. Embeddings resolvem paráfrase e sinonímia, mas "
    "custam mais, dependem de modelo pré-treinado, são opacos (não dá para apontar \"a palavra X causou o agrupamento\"), "
    "representam mal nomes locais e siglas municipais e truncam textos acima de 128 tokens, o que motiva a Entrega 3.")]

# 3.2
story += [P("3.2 Entrega 2: detecção de duplicatas", h2), P(
    "A função <i>detectar_duplicatas(textos, limiar=0.85)</i> codifica os textos com o MiniLM, calcula a matriz de "
    "similaridade e devolve os pares acima do limiar, com opção de reaproveitar embeddings pré-calculados. O heatmap 40×40 "
    "(triângulo inferior) está no notebook e em <i>resultados/heatmap_similaridade.png</i>. Para justificar o limiar, "
    "olhei a distribuição das 780 similaridades entre pares e varri limiares de 0,50 a 0,95, medindo precisão, recall e F1 "
    "contra o gabarito. "
    f"A distribuição tem média {v(e2, 'distribuicao', 'media', falta='[nb 2]')} e percentis "
    f"p90 = {v(e2, 'percentis', 'p90', falta='[nb 2]')}, p95 = {v(e2, 'percentis', 'p95', falta='[nb 2]')} e "
    f"p99 = {v(e2, 'percentis', 'p99', falta='[nb 2]')}.")]
m = (e2 or {}).get("metricas_limiar_final", {})
m85 = (e2 or {}).get("metricas_085", {})
linhas = [["Limiar", "Precisão", "Recall", "F1", "VP", "FP", "FN"],
          ["0,85 (enunciado)", v(m85, "precisão", fmt="{:.2f}", falta="[nb 2]"), v(m85, "recall", fmt="{:.2f}", falta=""), v(m85, "F1", fmt="{:.2f}", falta=""), v(m85, "VP", fmt="{}", falta=""), v(m85, "FP", fmt="{}", falta=""), v(m85, "FN", fmt="{}", falta="")],
          [f"{v(e2, 'limiar_final', fmt='{:.2f}', falta='[nb 2]')} (escolhido, F1 máx.)", v(m, "precisão", fmt="{:.2f}", falta="[nb 2]"), v(m, "recall", fmt="{:.2f}", falta=""), v(m, "F1", fmt="{:.2f}", falta=""), v(m, "VP", fmt="{}", falta=""), v(m, "FP", fmt="{}", falta=""), v(m, "FN", fmt="{}", falta="")]]
story += [tabela(linhas, [4.2 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm]), Spacer(1, 4)]
fps = ", ".join("×".join(p) for p in (e2 or {}).get("falsos_positivos", [])) or ("nenhum" if e2 else "[nb 2]")
fns = ", ".join("×".join(p) for p in (e2 or {}).get("falsos_negativos", [])) or ("nenhum" if e2 else "[nb 2]")
faixa = (e2 or {}).get("faixa_f1_max")
if faixa and abs(faixa[0] - faixa[1]) < 1e-9:
    escolha = (f"O F1 máximo ({v(m, 'F1', fmt='{:.2f}')}) ocorre em um único ponto da grade, {faixa[1]:.2f}, então a regra de "
               "desempate que adotei (preferir o maior limiar entre os empatados) não chegou a ser acionada")
elif faixa:
    escolha = f"Escolhi o maior limiar dentro da faixa de F1 máximo ({faixa[0]:.2f} a {faixa[1]:.2f})"
else:
    escolha = "Escolhi o maior limiar dentro da faixa de F1 máximo ([nb 2])"
story += [P(
    f"<b>Justificativa do limiar.</b> {escolha}. A preferência pelo limiar mais alto, em caso de empate, é por ser a opção mais "
    "conservadora: para a Ouvidoria, fundir indevidamente duas manifestações diferentes pode deixar um cidadão sem resposta, "
    "enquanto uma duplicata não detectada custa apenas retrabalho. O valor 0,85 do enunciado é comparado na tabela: com ele o "
    f"MiniLM detecta só {v(m85, 'VP', fmt='{}', falta='[nb 2]')} dos 6 pares do gabarito (recall {v(m85, 'recall', fmt='{:.2f}', falta='[nb 2]')}), "
    "sem falsos positivos, o que é restritivo demais para a escala de similaridade deste modelo. "
    "Também avaliei o limiar dinâmico por percentil (dica do enunciado): o p90 pressupõe que 10% dos "
    "pares são duplicatas, muito acima da realidade (6 em 780, 0,8%), então o p99 é o percentil adequado. A vantagem do "
    "percentil é se adaptar automaticamente à escala de similaridade de cada modelo, que muda entre MiniLM e mpnet, como a "
    "comparação no notebook mostra.")]
story += [P(
    f"<b>Erros.</b> Falsos positivos: {fps}. Falsos negativos: {fns}. O falso positivo é o caso típico de manifestações sobre o "
    "mesmo tipo de problema em lugares diferentes (iluminação pública na rua Manoel Deodato e na praça da Torre): o embedding "
    "captura \"o que\" e não \"onde\". Uma correção prática é combinar o score semântico com um filtro de entidade (bairro/rua) "
    "via regex ou NER. Os falsos negativos são textos curtos e de tamanho parecido, com similaridades "
    f"{sim_gab('M008', 'M022')} e {sim_gab('M019', 'M038')} (acima do percentil 95, mas abaixo do limiar); o que muda entre as "
    "versões é o vocabulário (posto de saúde/PSF, médico/profissional; assaltada/roubos, ponto/parada) e o ponto de vista, um "
    "relato pessoal de um lado e a descrição de um problema recorrente do outro. Para esses casos o ganho vem de um modelo "
    f"maior (o mpnet dá {emb('M008 × M022', 'Emb mpnet-base')} a M008 × M022) ou de um limiar menor combinado com o filtro de "
    "local, e não do chunking, já que nenhum dos dois pares envolve texto longo.")]

# 3.3
toks = [x["n_tokens"] for x in (e3 or {}).get("manifestacoes_longas", [])]
faixa_tok = f"entre {min(toks)} e {max(toks)} tokens" if toks else "[nb 3]"
story += [P("3.3 Entrega 3: chunking das manifestações longas", h2), P(
    f"O MiniLM trunca a entrada em {v(e3, 'max_seq_length', fmt='{}', falta='128')} tokens; as cinco manifestações longas "
    f"passam desse limite ({faixa_tok}), então o final do texto (que costuma conter o pedido concreto) seria ignorado. "
    "Usei o <i>RecursiveCharacterTextSplitter</i> com três configurações: A (200, 0), B (200, 60) e C (350, 80). Para cada "
    "uma medi a coesão consecutiva (cosseno entre chunks vizinhos), a fidelidade média (cosseno entre cada chunk e o texto "
    "completo) e a cobertura (cosseno do melhor chunk com o texto completo), além da silhueta dos chunks agrupados pela "
    "manifestação de origem.")]
linhas = [["Config", "Chunks (total)", "Coesão consecutiva", "Fidelidade média", "Cobertura", "Silhueta"]]
for c, (cs, co) in {"A": (200, 0), "B": (200, 60), "C": (350, 80)}.items():
    linhas.append([f"{c} ({cs}, {co})", v(e3, "resumo_chunks", c, "n_chunks", fmt="{:.0f}", falta="[nb 3]"),
                   v(e3, "coesao", c, "coesão consecutiva", falta="[nb 3]"), v(e3, "coesao", c, "fidelidade média", falta="[nb 3]"),
                   v(e3, "coesao", c, "cobertura (melhor chunk)", falta="[nb 3]"), v(e3, "silhueta", c, falta="[nb 3]")])
story += [tabela(linhas, [2.4 * cm, 2.4 * cm, 3.0 * cm, 2.8 * cm, 2.2 * cm, 2.0 * cm]), Spacer(1, 4)]
story += [P(
    "<b>Overlap e coesão.</b> Entre A e B a única diferença é o overlap, e a coesão consecutiva sobe: chunks vizinhos passam a "
    "compartilhar ~60 caracteres idênticos, o que puxa seus vetores um para o outro e reduz o risco de uma frase importante "
    "ficar cortada ao meio sem estar inteira em nenhum chunk. O custo é redundância (mais chunks, trechos contados duas "
    "vezes no ranking da busca); não testei overlaps maiores, e a expectativa é que acima de ~30% do chunk_size o ganho diminua. "
    "<b>Configuração recomendada: C.</b> Chunks de ~350 caracteres ficam abaixo do limite de tokens (nada é truncado) e ainda "
    "contêm sujeito + problema + local; C obteve a maior fidelidade e cobertura. B é melhor para busca por detalhe pontual "
    "(\"falta de insulina\" cai exatamente num chunk). Em M006, por exemplo, A corta a frase dos atendentes (\"dizem que o "
    "estoque não chega e que | devemos voltar em uma semana\") entre dois chunks, e o overlap de B a devolve inteira no chunk "
    "seguinte; em C a lista de remédios (chunk 0) e a agenda de consultas com o pedido final (chunk 2) ficam em chunks "
    "distintos e inteiros. "
    "<b>Espaço 2D.</b> Nos gráficos PCA/t-SNE os chunks da mesma manifestação formam grupos, mas não compactos: os "
    "primeiros chunks (que apresentam o problema) ficam perto do centróide, e os últimos (o pedido genérico à prefeitura) "
    "se aproximam dos chunks finais de outras manifestações. Chunks maiores (C) agrupam melhor.")]

# 3.4
story += [P("3.4 Entrega 4: app Streamlit", h2), P(
    "<i>app_ouvidoria.py</i> tem as quatro abas pedidas. A sidebar seleciona o modelo (MiniLM ou mpnet) e o top-k. O modelo "
    "fica em <i>st.cache_resource</i> e os embeddings da base em <i>st.cache_data</i>, chaveados pelo nome do modelo, então "
    "trocar de modelo recalcula uma vez e depois é instantâneo. Na Busca Semântica, além da lista com score colorido "
    "(verde/amarelo/vermelho), o app resume o resultado em uma frase (\"provável duplicata\", \"há relacionadas\", "
    "\"problema novo\"), oferece exemplos prontos e filtro por categoria. A Base Completa tem métricas do corpus, tabela "
    "completa, heatmap interativo (Plotly, com hover mostrando o par e o valor) e a lista de candidatos a duplicata para um "
    "limiar ajustável. O Espaço Vetorial permite alternar PCA/t-SNE (com perplexidade ajustável), mostra a silhueta por "
    "categoria e traz o comentário pedido: saúde e educação formam clusters coesos; infraestrutura e segurança se misturam "
    "(iluminação pública aparece nas duas); meio ambiente se espalha entre infraestrutura e saúde; as manifestações longas "
    "ficam no centro do mapa por tratarem de vários temas. A aba Chunking aceita texto colado, permite escolher estratégia "
    "(Recursive ou Character) e parâmetros, e exibe cada chunk com sua fidelidade ao documento, as primeiras dimensões "
    "dos embeddings e a matriz de similaridade entre chunks.")]

# ── 4. dificuldades ──────────────────────────────────────────────────────────
story += [P("4. Dificuldades", h1), P(
    "<b>Dataset ausente.</b> O JSON não veio com o enunciado; reconstruí um corpus fiel à especificação, com gabarito de "
    "duplicatas explícito, o que na verdade tornou a avaliação da Entrega 2 mais rigorosa. "
    "<b>Overlap que desaparece.</b> Minha primeira versão do chunking usava separadores por frase (\". \", \", \") para cortes "
    "mais naturais. Descobri que o splitter só reaproveita pedaços inteiros para formar o overlap; como as frases têm 80 a "
    "150 caracteres, um overlap de 60 não cabia em nenhuma e o overlap efetivo caía para ~2 caracteres, tornando A e B "
    "idênticas. A solução foi voltar aos separadores padrão (nível de palavra) e registrar o achado no notebook. "
    "<b>Limiar não transferível.</b> A escala de similaridade muda entre modelos; o 0,85 do enunciado não significa a mesma "
    "coisa para MiniLM e mpnet, daí a importância da varredura de limiares e do limiar por percentil. "
    "<b>t-SNE em amostra pequena.</b> Com 40 pontos (ou 15 a 30 chunks) a perplexidade precisa ser baixa e o resultado "
    "varia com a semente; fixei <i>random_state</i> e tratei o t-SNE como leitura qualitativa, mantendo o PCA ao lado. "
    "<b>Tipos do pandas.</b> Colunas de string com backend Arrow não aceitam indexação 2D no NumPy; foi preciso converter "
    "explicitamente para <i>object</i> antes de construir a máscara de categorias.")]

# ── 5. aprendizados ──────────────────────────────────────────────────────────
story += [P("5. Aprendizados", h1), P(
    "(1) Representações esparsas falham exatamente no caso que motiva o projeto, a paráfrase; embeddings resolvem isso, mas "
    "trazem um novo tipo de erro: juntam o mesmo problema em lugares diferentes, porque capturam o assunto e não a entidade. "
    "O sistema final deve combinar os dois (score semântico + filtro lexical de local). (2) Um limiar não é uma constante: é "
    "uma decisão de negócio que depende do custo de cada erro e da escala do modelo, e deve ser calibrada com um gabarito e "
    "revisada a cada troca de modelo. (3) Chunking não é só \"caber no modelo\": o tamanho do chunk define a granularidade da "
    "busca, e a métrica de fidelidade/cobertura ajuda a escolher sem depender do olho. (4) Detalhes de implementação "
    "(separadores do splitter, cache no Streamlit, tipos do pandas) mudam resultados e merecem ser documentados, já que o "
    "código será lido por outra pessoa da equipe.")]

# ── 6. execução ──────────────────────────────────────────────────────────────
story += [P("6. Como executar", h1), P(
    "<i>pip install -r requirements.txt</i>, depois rodar os notebooks na ordem (1, 2, 3), que gravam seus resultados em "
    "<i>resultados/</i>; <i>streamlit run app_ouvidoria.py</i> abre o app; <i>python gerar_relatorio.py</i> regenera este PDF "
    "com os números atualizados. Os modelos são baixados do Hugging Face na primeira execução.")]

doc = SimpleDocTemplate("RELATORIO.pdf", pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Ouvidoria Inteligente - Relatório", author=ALUNO)
doc.build(story)

from pypdf import PdfReader
print(f"RELATORIO.pdf gerado com {len(PdfReader('RELATORIO.pdf').pages)} página(s)")
