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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === 1. Charger les données shapefile
data = gpd.read_file('dataset.shp')
print("Colonnes :", data.columns)

# === 2. Vérifier colonnes utiles
features = ['orientatio', 'route', 'urbaine', 'vegetation', 'climat', 'hydro', 'pente']
data = data.dropna(subset=features + ['classe'])  # Supprimer lignes incomplètes

# === 3. Équilibrer les classes via undersampling
min_count = data['classe'].value_counts().min()
balanced_data = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in data.groupby('classe')
])

# === 4. Préparer les données
X = balanced_data[features]
y = balanced_data['classe']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 5. Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 6. Modèle Random Forest
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=45,
    max_features='auto',
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# === 7. Prédiction
y_pred = model.predict(X_test)
y_pred_label = label_encoder.inverse_transform(y_pred)
y_test_label = label_encoder.inverse_transform(y_test)

# === 8. Évaluation
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
plt.title("Matrice de confusion")
plt.xlabel("Prédits")
plt.ylabel("Réels")
plt.tight_layout()
plt.show()

# === 10. Courbe d'évolution de l'accuracy selon n_estimators
scores = []
n_estimators_range = range(10, 310, 20)
for n in n_estimators_range:
    temp_model = RandomForestClassifier(n_estimators=n, max_depth=20, random_state=42, class_weight='balanced')
    temp_model.fit(X_train, y_train)
    y_val_pred = temp_model.predict(X_test)
    acc_temp = accuracy_score(y_test, y_val_pred)
    scores.append(acc_temp)

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_range, scores, marker='o')
plt.title("Évolution de l'Accuracy selon le nombre d’arbres")
plt.xlabel("Nombre d'estimateurs")
plt.ylabel("Accuracy sur le test")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 11. Export des prédictions
data['prediction'] = model.predict(data[features])
data['prediction'] = label_encoder.inverse_transform(data['prediction'].astype(int))
data['classe'] = data['classe'].astype(str)

# === 12. Importance des critères
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Critère': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Critère', data=importance_df, palette='viridis')
plt.title("Importance des critères (Random Forest)")
plt.xlabel("Pondération apprise")
plt.ylabel("Critère")
plt.tight_layout()
plt.show()

# === 13. Ajouter coordonnées et exporter
data['X'] = data.geometry.x
data['Y'] = data.geometry.y

data[['X', 'Y', 'classe', 'prediction']].to_csv("predictions_rf.csv", index=False)
data.to_file("predictions_rf.shp")

joblib.dump(model, "random_forest_model.joblib")
joblib.dump(label_encoder, "label_encoder_rf.joblib")
