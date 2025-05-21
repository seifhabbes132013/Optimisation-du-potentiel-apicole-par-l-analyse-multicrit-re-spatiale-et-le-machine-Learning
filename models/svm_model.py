import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# === 1. Charger les données
data = gpd.read_file('dataset.shp')
print("Colonnes disponibles :", list(data.columns))

# === 2. Colonnes sélectionnées
features = ['orientatio', 'route', 'urbaine', 'vegetation', 'climat', 'hydro', 'pente']
data = data.dropna(subset=features + ['classe'])

# === 3. Préparation des données
X = data[features]
y = data['classe']

# Encodage des classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Séparation en train/test (stratifiée)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === 5. Application de SMOTE uniquement sur le jeu d'entraînement
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# === 6. Modèle SVM avec les hyperparamètres optimisés
svm_model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
svm_model.fit(X_smote, y_smote)

# === 7. Prédictions
y_pred = svm_model.predict(X_test)
y_pred_label = label_encoder.inverse_transform(y_pred)
y_test_label = label_encoder.inverse_transform(y_test)

# === 8. Évaluation du modèle
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"✅ Accuracy : {acc:.4f}")
print(f"✅ Précision : {precision:.4f}")
print(f"✅ Rappel : {recall:.4f}")
print("Classification report :\n", classification_report(y_test_label, y_pred_label))

# === 9. Matrice de confusion
cm = confusion_matrix(y_test_label, y_pred_label, labels=label_encoder.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion - SVM avec SMOTE")
plt.xlabel("Prédits")
plt.ylabel("Réels")
plt.tight_layout()
plt.show()

# === 10. Prédiction sur tous les points du jeu complet
X_full_scaled = scaler.transform(data[features])
data['prediction'] = svm_model.predict(X_full_scaled)
data['prediction'] = label_encoder.inverse_transform(data['prediction'])

# === 11. Extraction des coordonnées géographiques
data['X'] = data.geometry.x
data['Y'] = data.geometry.y

# === 12. Exportation des résultats
data[['X', 'Y', 'classe', 'prediction']].to_csv("predictions_svm.csv", index=False)
data.to_file("predictions_svm.shp")

# Sauvegarde du modèle et outils associés
joblib.dump(svm_model, "svm_model.joblib")
joblib.dump(label_encoder, "svm_label_encoder.joblib")
joblib.dump(scaler, "svm_scaler.joblib")

print("\n✅ Modèle SVM avec SMOTE et fichiers exportés avec succès.")
