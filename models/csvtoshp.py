import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# === 1. Charger le fichier CSV avec le bon séparateur
csv_path = 'test.csv'
df = pd.read_csv(csv_path, sep=';')  # ← assure-toi que le fichier utilise bien ;

# === 2. Nettoyer les noms de colonnes (sans changer la casse)
df.columns = df.columns.str.strip()

# === 3. Vérifier les colonnes X et Y (majuscules)
if 'X' not in df.columns or 'Y' not in df.columns:
    print("Colonnes disponibles :", df.columns.tolist())
    raise ValueError(" Les colonnes 'X' et 'Y' sont requises dans le fichier CSV.")

# === 4. Créer géométrie Point
geometry = [Point(xy) for xy in zip(df['X'], df['Y'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# === 5. Définir le système de coordonnées (par défaut EPSG:4326)
gdf.set_crs(epsg=32632, inplace=True)

# === 6. Exporter en shapefile
shapefile_path = csv_path.replace(".csv", ".shp")
gdf.to_file(shapefile_path)

print(f" Shapefile exporté avec succès : {shapefile_path}")
