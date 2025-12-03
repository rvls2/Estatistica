import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

archive_path = 'Projeto_estatistica/cardio_data_processed.csv'
populacao = pd.read_csv(archive_path)
populacao['bmi'] = populacao['bmi'].clip(upper=60)
populacao['ap_hi'] = populacao['ap_hi'].clip(upper=200)


colunas = ['id','age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio','age_years','bmi','bp_category','bp_category_encoded']

# Definir colunas (ajustar conforme seu dataset)
binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'age_years', 'bmi']

# Separar features e target
# ATENÇÃO: Verifique o nome correto da coluna target no seu CSV
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded'], axis=1)  # Ajuste o nome da coluna target se necessário
y = populacao['cardio']  # Ajuste o nome da coluna target se necessário


print("=== GERANDO MATRIZ DE CORRELAÇÃO ===")

# 1. Calcular a correlação entre todas as colunas de X
# (O X deve conter apenas números. Se tiver colunas de texto, o .corr() pode dar erro ou ignorar)
matriz_correlacao = X.corr()

# 2. Criar uma máscara para esconder a parte de cima do triângulo
# (Porque a matriz é espelhada, a parte de cima é igual a de baixo)
mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool))

# 3. Configurar o tamanho da imagem
plt.figure(figsize=(18, 14))

# 4. Desenhar o Heatmap
sns.heatmap(
    matriz_correlacao,
    mask=mask,           # Aplica a máscara
    cmap='coolwarm',     # Cores: Vermelho (+) e Azul (-)
    vmax=1, vmin=-1,     # Limites da escala de cor
    center=0,            # Centro da escala (branco) no zero
    annot=True,          # Escrever os números dentro dos quadrados
    fmt='.2f',           # Formatar números com 2 casas decimais
    square=True,         # Forçar quadrados perfeitos
    linewidths=.9,       # Linhas brancas entre os quadrados
    cbar_kws={"shrink": .5} # Tamanho da barra lateral de legenda
)
# Eixo X: Rotacionar 45 graus e alinhar à direita para não sobrepor
plt.xticks(rotation=45, ha='right', fontsize=12)

# Eixo Y: Manter reto, mas garantir tamanho legível
plt.yticks(rotation=0, fontsize=12)

plt.title('Matriz de Correlação (Diagnóstico para Naive Bayes)', fontsize=16)
plt.show()