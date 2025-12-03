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


colunas = ['id','age','gender','height','weight',
           'ap_hi','ap_lo','cholesterol','gluc',
           'smoke','alco','active','cardio','age_years',
           'bmi','bp_category','bp_category_encoded']

binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['ap_hi', 'age_years', 'bmi']

# Lista de Numéricas
populacao['cardio'] = populacao['cardio'].astype(int)
features_continuas = ['age_years', 'bmi', 'ap_hi']

plt.figure(figsize=(18, 6))

for i, col in enumerate(features_continuas):
    plt.subplot(1, 3, i+1)
    
    sns.boxplot(
        x='cardio', 
        y=col, 
        hue='cardio',  
        data=populacao, 
        palette={0: "#2ecc71", 1: "#e74c3c"}, 
        legend=False   
    )
    
    plt.title(f'{col} vs Doença Cardíaca', fontsize=14)
    plt.xlabel('0 = Saudável | 1 = Doente')
    
    if col == 'ap_hi': plt.ylim(80, 200)   # Foca na pressão entre 80 e 200
    if col == 'bmi': plt.ylim(15, 50)      # Foca no IMC comum
    if col == 'age_years': plt.ylim(30, 70)

plt.tight_layout()
plt.show()



features_categoricas = ['cholesterol', 'gluc', 'gender', 'smoke', 'alco', 'active']

plt.figure(figsize=(16, 8)) 

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
    
    plt.title(f'{col}', fontsize=11, fontweight='bold')
    plt.xlabel('') 
    plt.ylabel('Qtd', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    if i == 0:
        plt.legend(title='Doença', title_fontsize=9, fontsize=8, loc='upper right')
    else:
        ax.get_legend().remove() 

plt.show()