import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

# Carrega dataset
df = pd.read_csv("cardio_data_processed.csv")

# Listas de colunas
continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'age_years', 'bmi']
binary_cols = ['gender', 'smoke', 'alco', 'active']
categorical_cols = ['cholesterol', 'gluc']

# Gráficos para CONTÍNUAS
for col in continuous_cols:
    valores = df[col].dropna()
    media = valores.mean()
    desvio = valores.std()
    
    plt.figure(figsize=(8, 5))
    plt.hist(valores, bins=40, density=True, alpha=0.6, edgecolor='black')
    
    x = np.linspace(valores.min(), valores.max(), 200)
    pdf = norm.pdf(x, media, desvio)
    plt.plot(x, pdf, linewidth=2)
    
    plt.title(f"Distribuição Normal - {col}\nMédia={media:.2f}, Desvio={desvio:.2f}")
    plt.xlabel(col)
    plt.ylabel("Densidade")

    plt.savefig(f"distribuicao_{col}.png", dpi=200)
    plt.show()

# Gráficos para BINÁRIAS
for col in binary_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts().sort_index().plot(kind="bar", edgecolor="black")

    plt.title(f"Distribuição da Variável Binária - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")

    plt.savefig(f"grafico_binario_{col}.png", dpi=200)
    plt.show()

# Gráficos para CATEGÓRICAS 
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts().sort_index().plot(kind="bar", edgecolor="black")

    plt.title(f"Distribuição da Variável Categórica - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")

    plt.savefig(f"grafico_categorico_{col}.png", dpi=200)
    plt.show()
