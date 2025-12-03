import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


archive_path = 'Projeto_estatistica/cardio_data_processed.csv'
populacao = pd.read_csv(archive_path)


colunas = ['id','age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio','age_years','bmi','bp_category','bp_category_encoded']

# Definir colunas (ajustar conforme seu dataset)
binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['ap_hi', 'age_years', 'bmi']



# 1. Lista de Numéricas
populacao['cardio'] = populacao['cardio'].astype(int)
features_continuas = ['age_years', 'bmi', 'ap_hi']

plt.figure(figsize=(18, 6))

for i, col in enumerate(features_continuas):
    plt.subplot(1, 3, i+1)
    
    # Boxplot: X = Doença, Y = Valor da Variável
    # FIX 2: Adicionar hue='cardio' e legend=False
    sns.boxplot(
        x='cardio', 
        y=col, 
        hue='cardio',  # <--- Obrigatório nas novas versões do Seaborn
        data=populacao, 
        palette={0: "#2ecc71", 1: "#e74c3c"}, 
        legend=False   # <--- Remove a legenda duplicada
    )
    
    plt.title(f'{col} vs Doença Cardíaca', fontsize=14)
    plt.xlabel('0 = Saudável | 1 = Doente')
    
    # AJUSTE DE ZOOM (Opcional, mas recomendado para limpar o gráfico)
    # Remove outliers extremos que "esmagam" o desenho da caixa
    if col == 'ap_hi': plt.ylim(80, 200)   # Foca na pressão entre 80 e 200
    if col == 'bmi': plt.ylim(15, 50)      # Foca no IMC comum
    if col == 'age_years': plt.ylim(30, 70)

plt.tight_layout()
plt.show()



features_categoricas = ['cholesterol', 'gluc', 'gender', 'smoke', 'alco', 'active']

# 1. FIGSIZE MENOR: Ajuste aqui para ficar do tamanho que você quer na tela
# (Largura=16, Altura=8) é mais compacto que o anterior
plt.figure(figsize=(16, 8)) 

# 2. ESPAÇAMENTO ENTRE OS GRÁFICOS
# wspace = espaço horizontal, hspace = espaço vertical
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for i, col in enumerate(features_categoricas):
    plt.subplot(2, 3, i+1)
    
    # Gráfico
    ax = sns.countplot(
        x=col, 
        hue='cardio', 
        data=populacao, 
        palette={0: "#2ecc71", 1: "#e74c3c"}
    )
    
    # 3. AJUSTAR FONTES E LEGENDAS
    plt.title(f'{col}', fontsize=11, fontweight='bold')
    plt.xlabel('') # Remove o nome do eixo X para limpar a visão
    plt.ylabel('Qtd', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    # Legenda Inteligente: Só mostra no primeiro gráfico para não poluir
    if i == 0:
        plt.legend(title='Doença', title_fontsize=9, fontsize=8, loc='upper right')
    else:
        ax.get_legend().remove() # Remove legenda dos outros 5 gráficos

plt.show()



