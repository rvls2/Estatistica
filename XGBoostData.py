import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import compute_class_weight


archive_path = 'cardio_data_processed.csv'
populacao = pd.read_csv(archive_path)

colunas = ['id','age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio','age_years','bmi','bp_category','bp_category_encoded']

# Definir colunas (ajustar conforme seu dataset)
binary_cols = ['gender', 'smoke', 'alco', 'active', 'cardio']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 
                   'age_years', 'bmi']

# Separar features e target
# ATENÇÃO: Verifique o nome correto da coluna target no seu CSV
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded', 'age'], axis=1)  # Ajuste o nome da coluna target se necessário
y = populacao['cardio']  # Ajuste o nome da coluna target se necessário

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight_value = count_neg / count_pos

# 2. Inicializar o XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',     # Função de perda para classificação binária
    n_estimators=150,                # Número de árvores
    learning_rate=0.1,               # Taxa de aprendizado (ajusta a contribuição de cada árvore)
    max_depth=5,                     # Profundidade máxima das árvores
    scale_pos_weight=scale_pos_weight_value, # <--- O PONTO CRÍTICO PARA O RECALL
    random_state=42,
    use_label_encoder=False,         # Para suprimir avisos de versões futuras
    eval_metric='logloss'
)

# 3. Treinar o modelo
xgb_model.fit(X_train, y_train)

# 4. Fazer Previsões
y_pred_xgb = xgb_model.predict(X_test)


# 5. Avaliar o Desempenho
acuracia_xgb = accuracy_score(y_test, y_pred_xgb)
relatorio_xgb = classification_report(y_test, y_pred_xgb)

print("---")
print(f"Acurácia do XGBoost: {acuracia_xgb:.4f}")
print("---")
print("Relatório de Classificação (XGBoost com scale_pos_weight):")
print(relatorio_xgb)



importancia = xgb_model.feature_importances_
feature_names = X.columns

feature_importance = pd.DataFrame({
    'Feature':feature_names,
    'Importance':importancia
}).sort_values(by='Importance', ascending=False)

print("\n--- Ranking de Importância das Features (XGBoost) ---")
print(feature_importance.to_string())