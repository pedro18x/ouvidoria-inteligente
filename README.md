# Ouvidoria Inteligente: Triagem Semântica de Manifestações Cidadãs

Aluno: Pedro Ernesto

## Arquivos

| arquivo | entrega |
|---|---|
| `análise_comparativa.ipynb` | Entrega 1: BoW × TF-IDF × embeddings |
| `deteccao_duplicatas.ipynb` | Entrega 2: `detectar_duplicatas`, heatmap, limiar, FP/FN |
| `chunking_manifestacoes.ipynb` | Entrega 3: RecursiveCharacterTextSplitter, coesão, PCA/t-SNE |
| `app_ouvidoria.py` | Entrega 4: app Streamlit com 4 abas |
| `RELATORIO.pdf` | relatório de até 5 páginas (gerado por `gerar_relatorio.py`) |
| `manifestacoes.json` | corpus de 40 manifestações |
| `duplicatas_gabarito.json` | pares de duplicatas reais (para avaliar a Entrega 2) |
| `gerar_dataset.py` | recria o corpus (o JSON não veio com o enunciado) |
| `gerar_relatorio.py` | monta o PDF a partir de `resultados/` |

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt

# 1. notebooks, na ordem (cada um grava resultados/entregaN.json)
jupyter notebook   # ou: jupyter nbconvert --to notebook --execute --inplace *.ipynb

# 2. app
streamlit run app_ouvidoria.py

# 3. relatório com os números atualizados
python gerar_relatorio.py
```

Na primeira execução os modelos `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB)
e `paraphrase-multilingual-mpnet-base-v2` (~1.1 GB) são baixados do Hugging Face.

## Fluxo

Os notebooks salvam métricas em `resultados/`. O `gerar_relatorio.py` lê esses
arquivos e preenche as tabelas do PDF; enquanto um notebook não for executado, o
campo correspondente aparece como `[nb N]`. Os valores de BoW e TF-IDF são
recalculados pelo próprio script.

Os três notebooks estão no repositório já executados, com todas as saídas e
gráficos, e a pasta `resultados/` (JSONs de métricas, heatmap e projeções 2D)
também está versionada. Dá para ler tudo direto no GitHub, sem rodar nada; o
`RELATORIO.pdf` foi gerado a partir desses mesmos arquivos.
