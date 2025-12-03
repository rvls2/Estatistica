import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
populacao['bmi'] = populacao['bmi'].clip(upper=60)
populacao['ap_hi'] = populacao['ap_hi'].clip(upper=200)

accuracies = []
f1_scores = []

colunas = ['id','age','gender','height',
           'weight','ap_hi','ap_lo','cholesterol',
           'gluc','smoke','alco','active','cardio',
           'age_years','bmi','bp_category','bp_category_encoded']

binary_cols = ['gender', 'smoke', 'alco', 'active']

categorical_cols = ['cholesterol', 'gluc']

continuous_cols = ['ap_hi', 'age_years', 'bmi']

# Separar features e target
X = populacao.drop(['cardio', 'bp_category', 'bp_category_encoded',
                    'ap_lo', 'height','weight', 'age', 'id'], axis=1)  
y = populacao['cardio']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
scale_pos_weight_value = count_neg / count_pos

# Inicializar o XGBoost
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',     # Função de perda para classificação binária
    n_estimators=150,                # Número de árvores
    learning_rate=0.1,               # Taxa de aprendizado (ajusta a contribuição de cada árvore)
    max_depth=5,                     # Profundidade máxima das árvores
    scale_pos_weight=scale_pos_weight_value, # RECALL
    random_state=42,      
    eval_metric='logloss'
)

print("\n=== INICIANDO VALIDACAO CRUZADA (K-FOLD) COM SCALE_POS_WEIGHT - XGBOOST ===")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)

# MUDANÇA 1: Criar lista para guardar PROBABILIDADES, não apenas classes
y_real_total = []
y_proba_total = [] 

fold = 1
for train_index, val_index in kf.split(X, y):
    X_fold_train, X_fold_val = X.iloc[train_index], X.iloc[val_index]
    y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]
    
    # Modelo
    model_fold = clone(xgb_model)
    model_fold.fit(X_fold_train, y_fold_train)
    
    # MUDANÇA 2: Usar predict_proba em vez de predict
    proba_fold = model_fold.predict_proba(X_fold_val)[:, 1]
    y_fold_pred = model_fold.predict(X_fold_val)
    
    y_real_total.extend(y_fold_val)
    y_proba_total.extend(proba_fold) # Guardamos a probabilidade bruta

    acc = accuracy_score(y_fold_val, y_fold_pred)
    f1 = f1_score(y_fold_val, y_fold_pred, average='weighted')
    
    accuracies.append(acc)
    f1_scores.append(f1)
    
    print(f"Fold {fold}: Acuracia = {acc:.4f} | F1-Score = {f1:.4f}")
    fold += 1

# Converter para numpy array
y_real_total = np.array(y_real_total)
y_proba_total = np.array(y_proba_total)

# Limiar Padrão (0.50)
threshold_padrao = 0.50
y_pred_padrao = (y_proba_total >= threshold_padrao).astype(int)

# Limiar Ajustado (0.35)
threshold_ajustado = 0.35
y_pred_ajustado = (y_proba_total >= threshold_ajustado).astype(int)

# --- VISUALIZAÇÃO COMPARATIVA ---

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Matriz 1: Padrão (0.50)
cm_padrao = confusion_matrix(y_real_total, y_pred_padrao)
sns.heatmap(cm_padrao, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title(f'Limiar Padrao (0.50)\nFalsos Negativos: {cm_padrao[1,0]}')
axes[0].set_ylabel('Realidade')
axes[0].set_xlabel('Previsao')

# Matriz 2: Ajustado (0.35)
cm_ajustado = confusion_matrix(y_real_total, y_pred_ajustado)
sns.heatmap(cm_ajustado, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
axes[1].set_title(f'Limiar Ajustado (0.35)\nFalsos Negativos: {cm_ajustado[1,0]}')
axes[1].set_ylabel('Realidade')
axes[1].set_xlabel('Previsao')

plt.tight_layout()
plt.show()

print(f"Media de Acuracia: {np.mean(accuracies) * 100:.2f}")
print(f"Desvio Padrao de Acuracia: {np.std(accuracies, ddof=1) * 100:.2f}")

# --- RELATÓRIO DE IMPACTO ---
fn_padrao = cm_padrao[1,0]
fn_ajustado = cm_ajustado[1,0]
diff_fn = fn_padrao - fn_ajustado

fp_padrao = cm_padrao[0,1]
fp_ajustado = cm_ajustado[0,1]
diff_fp = fp_ajustado - fp_padrao

rec_padrao = recall_score(y_real_total, y_pred_padrao)
rec_ajustado = recall_score(y_real_total, y_pred_ajustado)

print("\n=== ANALISE DO IMPACTO REAL ===")
print(f"Ao baixar a regua de 50% para 35%:")
print(f"1. Vidas 'salvas' (Queda nos Falsos Negativos): {diff_fn} pessoas.")
print(f"2. Novo Recall (Sensibilidade): subiu de {rec_padrao:.2%} para {rec_ajustado:.2%}.")
print(f"3. Custo (Novos Falsos Positivos): {diff_fp} exames desnecessarios gerados.")