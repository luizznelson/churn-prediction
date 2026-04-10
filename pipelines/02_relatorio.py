import json
import os
import tomllib
from pathlib import Path

from groq import Groq

OUTPUTS = Path("outputs")
SECRETS = Path(".streamlit/secrets.toml")
OUT_FILE = OUTPUTS / "relatorio_executivo.md"


def load_api_key() -> str:
    if SECRETS.exists():
        with open(SECRETS, "rb") as f:
            return tomllib.load(f)["GROQ_API_KEY"]
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Groq API key not found. "
            "Add it to .streamlit/secrets.toml or set the GROQ_API_KEY env var."
        )
    return key


def build_prompt(metrics: dict, features: str, contrato: str, pagamento: str, faixa: str) -> str:
    churn_pct = round(metrics["churn_in_test"] / metrics["test_size"] * 100, 1)
    return f"""Você é um analista de dados sênior. Escreva um relatório executivo em português brasileiro sobre risco de churn para uma empresa de telecom.

Use APENAS os dados abaixo. Nunca invente ou suponha valores. Máximo 400 palavras.

## Dados do Modelo
- AUC-ROC: {metrics["auc_roc"]}
- Precisão: {metrics["precision"]}
- Recall: {metrics["recall"]}
- F1-Score: {metrics["f1"]}
- Acurácia: {metrics["accuracy"]}
- Tamanho do conjunto de teste: {metrics["test_size"]} clientes
- Clientes com churn no teste: {metrics["churn_in_test"]} ({churn_pct}%)

## Top 10 Variáveis por Importância
{features}

## Churn por Tipo de Contrato
{contrato}

## Churn por Método de Pagamento
{pagamento}

## Churn por Faixa de Tempo de Contrato
{faixa}

---

Escreva o relatório com exatamente estas três seções em Markdown:

# Resumo Executivo
(visão geral dos resultados e desempenho do modelo)

# Principais Riscos
(perfis de maior risco com base nos dados fornecidos)

# Recomendações
(ações práticas de retenção priorizadas pelos dados)

Seja direto, objetivo e use os números reais fornecidos acima."""


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((OUTPUTS / "metrics.json").read_text())

    import pandas as pd

    features_df = pd.read_csv(OUTPUTS / "feature_importance.csv").head(10)
    features_str = features_df.to_string(index=False)

    contrato_str = pd.read_csv(OUTPUTS / "churn_por_contrato.csv").to_string(index=False)
    pagamento_str = pd.read_csv(OUTPUTS / "churn_por_pagamento.csv").to_string(index=False)
    faixa_str = pd.read_csv(OUTPUTS / "churn_por_faixa_tempo.csv").to_string(index=False)

    prompt = build_prompt(metrics, features_str, contrato_str, pagamento_str, faixa_str)

    client = Groq(api_key=load_api_key())
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    report = response.choices[0].message.content
    OUT_FILE.write_text(report, encoding="utf-8")

    print(f"Saved: {OUT_FILE}\n")
    lines = report.splitlines()
    for line in lines[:20]:
        print(line)
    if len(lines) > 20:
        print(f"... (+{len(lines) - 20} more lines)")


if __name__ == "__main__":
    main()
