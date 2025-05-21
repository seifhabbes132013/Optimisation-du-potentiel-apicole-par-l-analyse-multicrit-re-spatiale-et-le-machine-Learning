
import pandas as pd
import numpy as np

# === 1. Charger le fichier CSV ===
file_path = 'promethee.csv'  # chemin vers ton fichier
df = pd.read_csv(file_path, sep=';')  # important : séparateur point-virgule
#df = df.sample(n=1750000, random_state=42)  # sous-échantillon aléatoire (si fichier est plus grand)

# === 2. Filtrer les colonnes ===
# Supprimer colonnes non critère, puis ne garder que les colonnes numériques
colonnes_a_ignorer = ['OBJECTID', 'aptitude',  'X', 'Y']
criteres = df.drop(columns=colonnes_a_ignorer, errors='ignore')
criteres = criteres.select_dtypes(include=[np.number])  # ne garder que les critères numériques

# === 3. Définir les pondérations (adapter si besoin) ===
# Le nombre doit correspondre au nombre de colonnes dans "criteres"
ponderations = np.array([0.1668, 0.0799, 0.0746, 0.3103, 0.2798, 0.0556, 0.0331])
ponderations = ponderations[:criteres.shape[1]]  # s'ajuste automatiquement si besoin
ponderations /= ponderations.sum()

# === 4. Normalisation Min-Max ===
criteres_norm = (criteres - criteres.min()) / (criteres.max() - criteres.min())
X = criteres_norm.values
n, m = X.shape

# === 5. Initialiser les flux
phi_plus = np.zeros(n)
phi_minus = np.zeros(n)
batch_size = 100  # Taille de traitement par lot

# === 6. Calcul des flux (PROMETHEE II optimisé) ===
for i in range(0, n, batch_size):
    i_end = min(i + batch_size, n)
    Xi = X[i:i_end]  # sous-matrice du lot

    # Différences positives uniquement
    diff = Xi[:, None, :] - X[None, :, :]  # (batch, n, m)
    positive_diff = np.where(diff > 0, diff, 0)
    preference = np.sum(positive_diff * ponderations, axis=2)  # (batch, n)

    # Calcul des flux pour le lot
    phi_plus[i:i_end] = np.sum(preference, axis=1) / (n - 1)
    phi_minus[i:i_end] = np.sum(preference, axis=0)[i:i_end] / (n - 1)

# === 7. Flux net
phi_net = phi_plus - phi_minus
df['phi_net'] = phi_net

# === 8. Export des résultats triés
df_sorted = df.sort_values(by='phi_net', ascending=False)
df_sorted.to_csv('promethemateur.csv', index=False)

print("✅ Résultats exportés dans : resultats_promethee22.csv")
