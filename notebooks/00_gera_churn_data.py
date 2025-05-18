import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'id': np.arange(1, n+1),
    'idade': np.random.randint(18, 75, n),
    'tempo_cliente': np.random.randint(1, 72, n),   # meses
    'mensalidade': np.random.uniform(60, 200, n).round(2),
    'suporte_ligacoes': np.random.randint(0, 10, n),
    'tem_servico_extra': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'pagamento_em_dia': np.random.choice([1, 0], n, p=[0.88, 0.12]),
})

# Agora simule o churn levando em conta as outras colunas
p = (
    0.12 + 0.12 * np.random.rand(n) + 
    0.2 * (df['suporte_ligacoes'] > 5) + 
    0.1 * (df['tem_servico_extra'] == 0) + 
    0.1 * (df['pagamento_em_dia'] == 0)
)
df['churn'] = np.random.binomial(1, p.clip(0, 1))  # Garante probabilidade no intervalo [0,1]

df.to_csv('data/churn_data.csv', index=False)