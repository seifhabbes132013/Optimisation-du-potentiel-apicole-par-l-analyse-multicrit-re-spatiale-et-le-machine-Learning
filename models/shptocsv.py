
import geopandas as gpd

# === 1. Charger le shapefile ===
shp_path = "test.shp"
gdf = gpd.read_file(shp_path)

# === 2. Extraire les coordonnées X et Y de chaque géométrie (Point uniquement) ===
gdf['X'] = gdf.geometry.x
gdf['Y'] = gdf.geometry.y

# === 3. Exporter vers CSV ===
csv_output = shp_path.replace(".shp", ".csv")
gdf.drop(columns='geometry').to_csv(csv_output, index=False)

print(f"✅ CSV exporté : {csv_output}")
