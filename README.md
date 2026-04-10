# Churn Ops Intelligence

[![Demo ao vivo](https://img.shields.io/badge/Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://churn-prediction-kw68vghhsd95yci7ppu5cj.streamlit.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Luiz%20Nelson-0A66C2?logo=linkedin)](https://www.linkedin.com/in/luiznelson)
[![GitHub](https://img.shields.io/badge/GitHub-luizznelson-181717?logo=github)](https://github.com/luizznelson)

---

## O que este projeto faz

Empresas de telecom acumulam milhares de registros de clientes, mas muitas vezes não conseguem transformar esses dados em ações práticas de retenção. Este projeto constrói um pipeline completo de análise de churn sobre o dataset público **Telco Customer Churn** do Kaggle, preparando a base, treinando um modelo preditivo e entregando um dashboard executivo para priorização de clientes com maior risco de cancelamento.

Fluxo: **dados brutos → base tratada → modelo → outputs → dashboard**.

---

## Resultados concretos

| Resultado | Detalhe |
|-----------|---------|
| **Predição de churn com base real do Kaggle** | O modelo foi treinado diretamente com variáveis reais da base Telco, sem depender de dados sintéticos |
| **Modelo com boa capacidade discriminativa** | AUC-ROC acima de 0,80, indicando boa separação entre clientes com e sem risco de churn |
| **Leitura operacional dos principais riscos** | Contrato mês a mês, electronic check, menor tempo de casa e maior cobrança mensal aparecem entre os principais sinais de risco |
| **Dashboard pronto para demonstração** | Painel com KPIs, drivers de churn, simulador de risco, relatório executivo e assistente de retenção |

---

## Como funciona

### Etapa 1 — Preparação da base
A base original do Kaggle é lida, tratada e normalizada. O campo `TotalCharges` é convertido para numérico e a variável alvo `Churn` é transformada para binário.

### Etapa 2 — Treino do modelo
É treinado um **Random Forest** com pré-processamento para variáveis numéricas e categóricas via `ColumnTransformer`, permitindo trabalhar diretamente com os dados reais da base Telco.

### Etapa 3 — Geração de outputs
O pipeline salva automaticamente:
- métricas do modelo em JSON
- importância agregada das variáveis em CSV
- relatório executivo em Markdown gerado via Groq
- modelo treinado em `joblib`

### Dashboard interativo
O painel Streamlit reúne cinco abas:
- **Visão Executiva** — KPIs e principais cortes de churn por contrato, pagamento e tempo de casa
- **Drivers de Churn** — variáveis mais influentes no modelo, com tabela e gráfico ajustável
- **Simulador** — formulário para testar perfis de clientes e estimar risco individualmente
- **Relatório** — relatório executivo gerado por LLM, renderizado em Markdown
- **Assistente** — chat com contexto do projeto para perguntas sobre retenção

---

## Stack

Python · Pandas · Scikit-learn · Streamlit · Joblib · Groq

---

## Estrutura do projeto

```text
churn-ops-intelligence/
├── data/
│   ├── raw/                          # CSV original do Kaggle — não versionado
│   └── processed/                    # Base tratada para modelagem — não versionada
├── pipelines/
│   ├── 00_prepare_data.py            # Tratamento e preparação da base
│   ├── 01_treinar_modelo.py          # Treino do modelo e geração de métricas
│   └── 02_relatorio.py               # Geração do relatório executivo via Groq
├── app/
│   └── dashboard.py                  # Dashboard Streamlit (5 abas)
├── models/
│   └── modelo_churn.joblib           # Modelo treinado — não versionado
├── outputs/
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── churn_por_contrato.csv
│   ├── churn_por_pagamento.csv
│   ├── churn_por_faixa_tempo.csv
│   └── relatorio_executivo.md
├── .streamlit/
│   └── secrets.example               # Template para secrets.toml local
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Configure a chave Groq (necessária para 02_relatorio.py e o Assistente)
cp .streamlit/secrets.example .streamlit/secrets.toml
# edite secrets.toml e adicione sua GROQ_API_KEY (gratuita em console.groq.com)

# 3. Baixe o dataset em data/raw/
# kaggle.com/datasets/blastchar/telco-customer-churn
# Arquivo: WA_Fn-UseC_-Telco-Customer-Churn.csv

# 4. Execute os pipelines em ordem
python pipelines/00_prepare_data.py
python pipelines/01_treinar_modelo.py
python pipelines/02_relatorio.py

# 5. Suba o dashboard
streamlit run app/dashboard.py
```

---

## Sobre o autor

Analista com foco em automação de processos, dados e GenAI aplicada. Este projeto foi desenvolvido como demonstração prática de como transformar dados brutos em inteligência operacional de forma autônoma — da ingestão ao relatório executivo.

- [LinkedIn](https://www.linkedin.com/in/luiznelson)
- [GitHub](https://github.com/luizznelson)

