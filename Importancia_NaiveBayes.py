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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_cols, categorical_cols, continuous_cols):
        self.binary_cols = binary_cols
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        
        self.bn = BernoulliNB()
        self.mn = MultinomialNB()
        self.gn = GaussianNB()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Verificar se as colunas existem no DataFrame
        for col in self.binary_cols + self.categorical_cols + self.continuous_cols:
            if col not in X.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
        
        self.bn.fit(X[self.binary_cols], y)
        self.mn.fit(X[self.categorical_cols], y)
        Xc = self.scaler.fit_transform(X[self.continuous_cols])
        self.gn.fit(Xc, y)
        return self

    def predict_proba(self, X):
        lp = self.bn.predict_log_proba(X[self.binary_cols])
        lp += self.mn.predict_log_proba(X[self.categorical_cols])
        lp += self.gn.predict_log_proba(self.scaler.transform(X[self.continuous_cols]))
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

archive_path = 'Projeto_estatistica/cardio_data_processed.csv'
populacao = pd.read_csv(archive_path)
populacao['bmi'] = populacao['bmi'].clip(upper=60)
populacao['ap_hi'] = populacao['ap_hi'].clip(upper=200)



colunas = ['id','age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio','age_years','bmi','bp_category','bp_category_encoded']

# Definir colunas (ajustar conforme seu dataset)
binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['ap_hi', 'age_years', 'bmi']

# Separar features e target
# ATENÇÃO: Verifique o nome correto da coluna target no seu CSV
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded', 'ap_lo', 'height','weight', 'age', 'id'], axis=1)  # Ajuste o nome da coluna target se necessário
y = populacao['cardio']  # Ajuste o nome da coluna target se necessário

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

# Aplicar SMOTE para oversampling
smote = SMOTE(sampling_strategy='auto', random_state=25)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("=== BALANCEAMENTO DE DADOS ===")
print(f"Distribuicao original: {np.bincount(y_train)}")
print(f"Distribuicao apos SMOTE: {np.bincount(y_train_res)}")

# Treinar o modelo
model = MixedNaiveBayes(
    binary_cols=binary_cols,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols
)

model.fit(X_train_res, y_train_res)

# Fazer previsões
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)


print("=== CALCULANDO IMPORTANCIA DAS FEATURES (PERMUTATION) ===")

# 1. É necessário ter o modelo treinado
# (Seu código já treinou o 'model' na linha: model.fit(X_train_res, y_train_res))

# 2. Calcular a importância usando os dados de TESTE (X_test, y_test)
# n_repeats=10: Ele vai embaralhar cada coluna 10 vezes para ter certeza estatística
result = permutation_importance(
    model, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1  # Usa todos os núcleos do PC para ir rápido
)

# 3. Organizar os dados em um DataFrame
feature_names = X_test.columns
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importancia_Media': result.importances_mean,
    'Desvio_Padrao': result.importances_std
})

# 4. Ordenar do mais importante para o menos importante
importances = importances.sort_values(by='Importancia_Media', ascending=False)

# 5. Plotar o Gráfico de Barras
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importancia_Media', 
    y='Feature', 
    hue='Feature',      # <--- ADICIONADO: A cor segue a Feature
    data=importances, 
    palette='viridis',
    legend=False        # <--- ADICIONADO: Remove a legenda duplicada
)

plt.title('Importância das Features - Naive Bayes (Queda na Acurácia)', fontsize=14)
plt.xlabel('O quanto a Acurácia cai se removermos essa feature?')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8) # Linha do zero
plt.tight_layout()
plt.show()

# Exibir tabela textual
print(importances)