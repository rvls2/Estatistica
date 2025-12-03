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
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded'], axis=1)  
y = populacao['cardio']  


print("=== GERANDO MATRIZ DE CORRELAÇÃO ===")

# Calcular a correlação entre todas as colunas de X
# Correção de numeros
matriz_correlacao = X.corr()

# Criar uma máscara para esconder a parte de cima do triângulo
mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool))

# Configurar o tamanho da imagem
plt.figure(figsize=(18, 14))

# Desenhar o Heatmap
sns.heatmap(
    matriz_correlacao,
    mask=mask,          
    cmap='coolwarm',     # Cores: Vermelho (+) e Azul (-)
    vmax=1, vmin=-1,     
    center=0,            
    annot=True,          # Escrever os números dentro dos quadrados
    fmt='.2f',           
    square=True,         
    linewidths=.9,       
    cbar_kws={"shrink": .5} 
)
# Eixo X: Rotacionar 45 graus e alinhar à direita para não sobrepor
plt.xticks(rotation=45, ha='right', fontsize=12)

# Eixo Y: Manter reto, mas garantir tamanho legível
plt.yticks(rotation=0, fontsize=12)

plt.title('Matriz de Correlação (Diagnóstico para Naive Bayes)', fontsize=16)
plt.show()