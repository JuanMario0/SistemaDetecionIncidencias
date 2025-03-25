from flask import Flask, render_template_string
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import joblib
import os

# Crear la aplicación Flask con el directorio raíz explícito
app = Flask(__name__, static_folder='../static', static_url_path='/static')
# Asegurarse de que la carpeta 'static' existe en la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio de app.py
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')    # Subir un nivel desde api/
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


# Verificar si heatmap.html existe, si no, generarlo
heatmap_path = os.path.join(STATIC_DIR, 'heatmap.html')
print(f"Buscando heatmap.html en: {heatmap_path}")  # Depuración
if not os.path.exists(heatmap_path):
    print("Generando heatmap.html...")
# Cargar y procesar datos
full_data = pd.read_csv("full_data.csv")
full_data['arrival_time'] = pd.to_datetime(full_data['arrival_time'], format='%H:%M:%S', errors='coerce')
full_data = full_data.sort_values(by=['stop_id', 'arrival_time'])
full_data['wait_time'] = full_data.groupby('stop_id')['arrival_time'].diff().dt.total_seconds() / 60
full_data['headway_mins'] = full_data['headway_secs'] / 60
full_data['delay'] = full_data['wait_time'] - full_data['headway_mins']
full_data['delay'] = full_data['delay'].fillna(0)
np.random.seed(42)
simulated_delays = np.random.poisson(lam=5, size=len(full_data))
full_data['simulated_delay'] = np.where(full_data['delay'] <= 0, simulated_delays, full_data['delay'])

# Clustering
X = full_data[['stop_lat', 'stop_lon', 'simulated_delay']].dropna()
kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
full_data['cluster'] = kmeans.predict(X)

# Métricas
silhouette_avg = silhouette_score(X, kmeans.labels_)
cluster_delays = full_data.groupby('cluster')['simulated_delay'].mean()
delays_by_stop = full_data.groupby('stop_id').agg({
    'stop_lat': 'first',
    'stop_lon': 'first',
    'stop_name': 'first',
    'simulated_delay': 'mean'
}).reset_index()
top_delays = delays_by_stop.sort_values('simulated_delay', ascending=False).head(10)

# Guardar modelo
joblib.dump(kmeans, 'kmeans_model.pkl')

# Generar mapa de calor
map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
m = folium.Map(location=map_center, zoom_start=12)
heat_data = [[row['stop_lat'], row['stop_lon'], row['simulated_delay']] 
             for _, row in delays_by_stop.iterrows()]
HeatMap(heat_data, radius=15).add_to(m)
for _, row in top_delays.iterrows():
    folium.Marker(
        location=[row['stop_lat'], row['stop_lon']],
        popup=f"{row['stop_name']}: {row['simulated_delay']:.1f} min",
        icon=folium.Icon(color='red', icon='bus', prefix='fa')
    ).add_to(m)
m.save("static/heatmap.html")
print(f"heatmap.html generado en: {heatmap_path}")

@app.route('/')
def home():
    # Plantilla HTML simple para mostrar métricas y tabla
    html = """
    <html>
    <head><title>Incidencias RTP</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
    </head>
    <body>
    <h1>Incidencias en Transporte RTP</h1>
    <h3>Métricas</h3>
    <p>Silhouette Score: {{ silhouette }}</p>
    <h3>Retraso Promedio por Clúster (minutos)</h3>
    {{ cluster_table | safe }}
    <h3>Top 10 Paradas con Mayores Retrasos</h3>
    {{ top_table | safe }}
    <h3>Mapa de Calor</h3>
    <iframe src="/static/heatmap.html" width="100%" height="500px" frameborder="0"></iframe>
    </body>
    </html>
    """
    cluster_table = cluster_delays.to_frame().to_html()
    top_table = top_delays[['stop_name', 'simulated_delay']].to_html(index=False)
    return render_template_string(html, silhouette=silhouette_avg, cluster_table=cluster_table, top_table=top_table)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)