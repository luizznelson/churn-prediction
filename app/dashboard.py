import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📉 Churn Prediction Dashboard")

# Carrega dados e modelo treinado
df = pd.read_csv("data/churn_data.csv")
model = joblib.load("app/modelo_churn.pkl")

st.write("### Estatísticas da base")
st.write(df.describe())

fig, ax = plt.subplots()
df['churn'].value_counts().plot(kind='bar', ax=ax, color=['blue','red'])
ax.set_xticklabels(['Não Cancelou', 'Cancelou'], rotation=0)
ax.set_title('Distribuição de Churn')
st.pyplot(fig)

st.write("---\n### Prever Churn de um Cliente")
idade = st.number_input("Idade", 18, 100, 30)
tempo_cliente = st.slider("Tempo como cliente (meses)", 1, 72, 12)
mensalidade = st.number_input("Mensalidade", 60.0, 200.0, 100.0)
suporte_ligacoes = st.slider("Chamadas ao suporte no mês", 0, 10, 1)
tem_servico_extra = st.selectbox("Tem serviço extra?", ['Não', 'Sim']) == 'Sim'
pagamento_em_dia = st.selectbox("Pagamento em dia?", ['Sim', 'Não']) == 'Sim'

if st.button("Prever Churn"):
    dados = [[
        idade,
        tempo_cliente,
        mensalidade,
        suporte_ligacoes,
        int(tem_servico_extra),
        int(pagamento_em_dia)
    ]]
    pred = model.predict(dados)[0]
    prob = model.predict_proba(dados)[0][1]
    st.success(f"Probabilidade de Churn: {prob:.1%}")
    if pred == 1:
        st.error("⚠️ Este cliente tem ALTO risco de cancelamento!")
    else:
        st.info("✅ Este cliente tem BAIXO risco de cancelamento.")
