import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === 1. Charger les données shapefile
data = gpd.read_file('dataset.shp')

# === 2. Supprimer les lignes avec NaN dans les colonnes utilisées
features = ['orientatio', 'route', 'urbaine', 'vegetation', 'climat', 'hydro', 'pente']
data = data.dropna(subset=features + ['classe'])

# === 3. Équilibrer les classes par sous-échantillonnage
min_class_count = data['classe'].value_counts().min()
balanced_data = pd.concat([
    group.sample(min_class_count, random_state=42)
    for _, group in data.groupby('classe')
])

# === 4. Préparer les données
X = balanced_data[features]
y = balanced_data['classe']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 5. Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === 6. Initialiser et entraîner le modèle XGBoost
eval_set = [(X_train, y_train), (X_test, y_test)]

model = XGBClassifier(
    n_estimators=1000,
    max_depth=25,
    colsample_bytree=0.9,
    gamma=3.10508,
    min_child_weight=4,
    reg_alpha=37,
    reg_lambda=1.80918,
    scale_pos_weight=2,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# === 7. Prédictions
y_pred = model.predict(X_test)
y_pred_label = label_encoder.inverse_transform(y_pred)
y_test_label = label_encoder.inverse_transform(y_test)

# === 8. Évaluation
acc = accuracy_score(y_test_label, y_pred_label)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"✅ Accuracy : {acc:.4f}")
print(f"✅ Précision : {precision:.4f}")
print(f"✅ Rappel : {recall:.4f}")
print("Classification report :\n", classification_report(y_test_label, y_pred_label))

# === 9. Matrice de confusion
cm = confusion_matrix(y_test_label, y_pred_label)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion")
plt.xlabel("Prédits")
plt.ylabel("Réels")
plt.tight_layout()
plt.show()

# === 10. Courbe de Log Loss
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
plt.xlabel('Itérations')
plt.ylabel('Log Loss')
plt.title('Courbe de Log Loss (Overfitting vs Underfitting)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 11. Prédictions sur tout le dataset (pour export)
X_full = data[features]
y_full_pred = model.predict(X_full)
data['prediction'] = label_encoder.inverse_transform(y_full_pred.astype(int))

# === 12. Ajouter coordonnées
data['X'] = data.geometry.x
data['Y'] = data.geometry.y

# === 13. Exporter résultats
data[['X', 'Y', 'classe', 'prediction']].to_csv("predictions_xgb.csv", index=False)
data.to_file("predictions_xgb.shp")

# === 14. Sauvegarder le modèle et l'encodeur
joblib.dump(model, "xgboost_model.joblib")
joblib.dump(label_encoder, "xgb_label_encoder.joblib")

print("✅ Modèle XGBoost entraîné et sauvegardé avec succès.")
