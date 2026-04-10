import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import altair as alt

# ── paths ──────────────────────────────────────────────────────────────────────
METRICS   = Path("outputs/metrics.json")
FEATURES  = Path("outputs/feature_importance.csv")
CONTRATO  = Path("outputs/churn_por_contrato.csv")
PAGAMENTO = Path("outputs/churn_por_pagamento.csv")
FAIXA     = Path("outputs/churn_por_faixa_tempo.csv")
RELATORIO = Path("outputs/relatorio_executivo.md")
MODEL_PATH = Path("models/modelo_churn.joblib")

st.set_page_config(page_title="Churn Ops Intelligence", layout="wide")

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))

@st.cache_data
def load_features():
    return pd.read_csv(FEATURES)

@st.cache_data
def load_contrato():
    return pd.read_csv(CONTRATO)

@st.cache_data
def load_pagamento():
    return pd.read_csv(PAGAMENTO)

@st.cache_data
def load_faixa():
    return pd.read_csv(FAIXA)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_assistant_context() -> str:
    parts = []
    if METRICS.exists():
        parts.append("## Métricas do Modelo\n" + METRICS.read_text(encoding="utf-8"))
    if FEATURES.exists():
        df = pd.read_csv(FEATURES).head(10)
        parts.append("## Top 10 Variáveis por Importância\n" + df.to_string(index=False))
    if RELATORIO.exists():
        parts.append("## Relatório Executivo\n" + RELATORIO.read_text(encoding="utf-8"))
    return "\n\n".join(parts)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Churn Ops Intelligence")
    st.caption("Pipeline: dados reais Telco · Random Forest · Groq")
    st.divider()
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visão Executiva", "Drivers de Churn", "Simulador", "Relatório", "Assistente",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Visão Executiva
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Visão Executiva")

    required = [METRICS, CONTRATO, PAGAMENTO, FAIXA]
    missing  = [str(p) for p in required if not p.exists()]
    if missing:
        st.warning(f"Arquivos ausentes: {', '.join(missing)}. Execute `01_treinar_modelo.py` primeiro.")
    else:
        m = load_metrics()
        churn_pct = round(m["churn_in_test"] / m["test_size"] * 100, 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC-ROC",   f"{m['auc_roc']:.4f}")
        c2.metric("Precisão",  f"{m['precision']:.4f}")
        c3.metric("Recall",    f"{m['recall']:.4f}")
        c4.metric("F1-Score",  f"{m['f1']:.4f}")
        c5.metric("Acurácia",  f"{m['accuracy']:.4f}")

        st.caption(
            f"Conjunto de teste: {m['test_size']:,} clientes · "
            f"Churn observado: {m['churn_in_test']:,} ({churn_pct}%)"
        )
        st.divider()

        ca, cb, cc = st.columns(3)

        with ca:
            st.subheader("Churn por Contrato")
            df_c = load_contrato().copy()
            df_c["churn_%"] = (df_c["churn_rate"] * 100).round(1)
            # ── gráfico contrato ─────────────────────────────
            chart_c = alt.Chart(df_c).mark_bar().encode(
                x=alt.X("Contract:N", title="Contrato"),
                y=alt.Y("churn_%:Q", title="Churn (%)"),
                tooltip=["Contract", "churn_%"]
            )
            st.altair_chart(chart_c, use_container_width=True)


        with cb:
            st.subheader("Churn por Pagamento")
            df_p = load_pagamento().copy()
            df_p["churn_%"] = (df_p["churn_rate"] * 100).round(1)
            # ── gráfico pagamento ───────────────────────────
            chart_p = alt.Chart(df_p).mark_bar().encode(
                x=alt.X("PaymentMethod:N", title="Pagamento"),
                y=alt.Y("churn_%:Q", title="Churn (%)"),
                tooltip=["PaymentMethod", "churn_%"]
            )
            st.altair_chart(chart_p, use_container_width=True)

        with cc:
            st.subheader("Churn por Faixa de Tempo")
            df_f = load_faixa().copy()
            df_f["churn_%"] = (df_f["churn_rate"] * 100).round(1)
            # ── gráfico tempo ───────────────────────────────
            chart_f = alt.Chart(df_f).mark_bar().encode(
                x=alt.X("faixa_tempo:N", title="Tempo"),
                y=alt.Y("churn_%:Q", title="Churn (%)"),
                tooltip=["faixa_tempo", "churn_%"]
            )
            st.altair_chart(chart_f, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Drivers de Churn
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Drivers de Churn")

    if not FEATURES.exists():
        st.warning("Execute `01_treinar_modelo.py` para gerar a importância das variáveis.")
    else:
        df_feat = load_features()
        top_n = st.slider("Número de variáveis exibidas", 5, len(df_feat), 15)
        df_top = df_feat.head(top_n).set_index("feature")[["importance"]]
        
        chart_feat = alt.Chart(df_top.reset_index()).mark_bar().encode(
            x=alt.X("importance:Q", title="Importância"),
            y=alt.Y("feature:N", sort="-x", title="Variável"),
            tooltip=["feature", "importance"]
        )

        st.altair_chart(chart_feat, use_container_width=True)

        with st.expander("Tabela completa de importância"):
            st.dataframe(df_feat, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Simulador
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Simulador de Risco Individual")

    if not MODEL_PATH.exists():
        st.warning("Execute `01_treinar_modelo.py` para gerar o modelo.")
    else:
        model = load_model()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Contrato e Cobrança")
            contract        = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
            tenure          = st.slider("Tempo de contrato (meses)", 0, 72, 12)
            monthly_charges = st.number_input("Cobrança mensal (R$)", 0.0, 200.0, 65.0, step=0.5)
            total_charges   = st.number_input(
                "Cobrança total (R$)", 0.0, 10000.0,
                float(round(monthly_charges * tenure, 2)), step=1.0,
            )
            payment_method  = st.selectbox("Método de pagamento", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            internet_service = st.selectbox("Serviço de internet", ["DSL", "Fiber optic", "No"])

        with col2:
            st.subheader("Perfil")
            gender      = st.selectbox("Gênero", ["Male", "Female"])
            senior      = st.checkbox("Cliente sênior (60+)")
            partner     = st.checkbox("Possui cônjuge/parceiro")
            dependents  = st.checkbox("Possui dependentes")
            paperless   = st.checkbox("Fatura digital")

            st.subheader("Serviços Adicionais")
            phone_service     = st.checkbox("Serviço de telefone", value=True)
            multiple_lines    = st.checkbox("Múltiplas linhas")
            online_security   = st.checkbox("Segurança online")
            online_backup     = st.checkbox("Backup online")
            device_protection = st.checkbox("Proteção de dispositivo")
            tech_support      = st.checkbox("Suporte técnico")
            streaming_tv      = st.checkbox("Streaming TV")
            streaming_movies  = st.checkbox("Streaming filmes")

        if st.button("Calcular risco de churn", type="primary"):
            row = pd.DataFrame([{
                "gender":           gender,
                "SeniorCitizen":    int(senior),
                "Partner":          int(partner),
                "Dependents":       int(dependents),
                "tenure":           tenure,
                "PhoneService":     int(phone_service),
                "MultipleLines":    int(multiple_lines),
                "InternetService":  internet_service,
                "OnlineSecurity":   int(online_security),
                "OnlineBackup":     int(online_backup),
                "DeviceProtection": int(device_protection),
                "TechSupport":      int(tech_support),
                "StreamingTV":      int(streaming_tv),
                "StreamingMovies":  int(streaming_movies),
                "Contract":         contract,
                "PaperlessBilling": int(paperless),
                "PaymentMethod":    payment_method,
                "MonthlyCharges":   monthly_charges,
                "TotalCharges":     total_charges,
            }])

            prob = model.predict_proba(row)[0][1]
            st.divider()

            if prob >= 0.6:
                st.error(f"Risco ALTO de churn: {prob:.1%}")
            elif prob >= 0.35:
                st.warning(f"Risco MÉDIO de churn: {prob:.1%}")
            else:
                st.success(f"Risco BAIXO de churn: {prob:.1%}")

            st.progress(prob, text=f"Probabilidade estimada: {prob:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Relatório
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Relatório Executivo")

    if not RELATORIO.exists():
        st.warning("Execute `02_relatorio.py` para gerar o relatório.")
    else:
        st.markdown(RELATORIO.read_text(encoding="utf-8"))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Assistente
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Assistente de Retenção")

    groq_key = st.secrets.get("GROQ_API_KEY", "")

    if not groq_key:
        st.warning(
            "Chave da API Groq não configurada. "
            "Adicione `GROQ_API_KEY` em `.streamlit/secrets.toml` para usar o assistente."
        )
    else:
        ctx_missing = [str(p) for p in [METRICS, FEATURES, RELATORIO] if not p.exists()]
        if ctx_missing:
            st.warning(
                f"Contexto incompleto — arquivos ausentes: {', '.join(ctx_missing)}. "
                "Execute os pipelines primeiro."
            )
        else:
            context = load_assistant_context()

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_input := st.chat_input("Faça uma pergunta sobre churn e retenção..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                try:
                    from groq import Groq

                    if not groq_key:
                        raise ValueError("API key não encontrada")

                    client = Groq(api_key=groq_key)

                except Exception as e:
                    st.error("Erro ao inicializar o assistente de IA.")
                    st.info("Verifique se a chave da API Groq está configurada corretamente.")
                    st.stop()

                system_prompt = (
                    "Você é um assistente especialista em retenção de clientes de telecom.\n"
                    "Responda APENAS com base no contexto abaixo. "
                    "Se a pergunta não puder ser respondida com os dados fornecidos, diga isso claramente.\n"
                    "Responda sempre em português brasileiro. "
                    "Foque em ações práticas de retenção.\n\n"
                    f"---\n{context}\n---"
                )

                api_messages = [{"role": "system", "content": system_prompt}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                with st.chat_message("assistant"):
                    with st.spinner("Analisando..."):
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            temperature=0.1,
                            messages=api_messages,
                        )
                        answer = response.choices[0].message.content
                    st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
