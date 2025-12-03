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

class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_cols, categorical_cols, continuous_Scols):
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



colunas = ['id','age','gender','height',
           'weight','ap_hi','ap_lo','cholesterol',
           'gluc','smoke','alco','active','cardio',
           'age_years','bmi','bp_category','bp_category_encoded']

# Definir colunas (ajustar conforme seu dataset)
binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['ap_hi', 'age_years', 'bmi']

# Separar features e target
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded', 
                    'ap_lo', 'height','weight', 'age', 'id'], axis=1) 
y = populacao['cardio'] 

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

# Métricas de avaliação
print("\n=== METRICAS DO MODELO NAIVE BAYES ===")
print(f"Acuracia: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatorio de Classificacao:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusao:")
print(confusion_matrix(y_test, y_pred))

# Probabilidades por classe
print("\n=== PROBABILIDADES ===")
print("Probabilidades previstas (primeiras 10 amostras):")
for i in range(min(10, len(y_pred_proba))):
    print(f"Amostra {i+1}: {y_pred_proba[i]}")

# Estatísticas das previsões
print(f"\nDistribuicao das previsoes: {np.bincount(y_pred)}")
print(f"Distribuição real: {np.bincount(y_test)}")

# Verificar importância das features (exemplo simples)
print("\n=== INFORMACOES DO MODELO ===")
print("Modelo BernoulliNB (features binárias):")
print(f"Classes: {model.bn.classes_}")
print(f"Log probabilidades a priori: {model.bn.class_log_prior_}")

print("\nModelo GaussianNB (features continuas):")
print(f"Classes: {model.gn.classes_}")
print(f"Medias por classe: {model.gn.theta_.shape}")

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
print(f"\nValidação Cruzada (5 folds): {cv_scores}")
print(f"Media da validação cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n=== TESTE COM NOVOS DADOS ===")

nova_amostra = pd.DataFrame({
    'Gender': [1],
    'Family Heart Disease': [0],
    'Smoking': [1],
    'Diabetes': [0],
    'Low HDL Cholesterol': [1],
    'High LDL Cholesterol': [0],
    'High Blood Pressure': [1],
    'Age Group': [2],
    'BP Category': [1],
    'Exercise Habits': [3],
    'Alcohol Consumption': [1],
    'Stress Level': [2],
    'Sugar Consumption': [1],
    'Age': [45],
    'Sleep Hours': [7],
    'Blood Pressure': [130],
    'BMI': [25.5],
    'Cholesterol Level': [200],
    'Triglyceride Level': [150],
    'Fasting Blood Sugar': [95],
    'CRP Level': [2.1],
    'Homocysteine Level': [12.0]
})

# Garantir que a nova amostra tenha todas as colunas necessárias
for col in X.columns:
    if col not in nova_amostra.columns:
        nova_amostra[col] = 0  # ou valor padrão apropriado

nova_amostra = nova_amostra[X.columns]  # Manter mesma ordem das colunas

predicao_nova = model.predict(nova_amostra)
probabilidade_nova = model.predict_proba(nova_amostra)

print(f"Previsao para nova amostra: {predicao_nova[0]}")
print(f"Probabilidades: {probabilidade_nova[0]}")