from flask import Flask, render_template_string, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import joblib
import os

# Crear la aplicación Flask con el directorio raíz explícito
app = Flask(__name__, template_folder='../templates', static_folder='../static')
# Asegurarse de que la carpeta 'static' existe
if not os.path.exists('static'):
    os.makedirs('static')

# Asegurarse de que la carpeta 'templates' existe
if not os.path.exists('templates'):
    os.makedirs('templates')

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

# Nueva ruta para obtener las paradas por clúster
@app.route('/get_stops/<int:cluster_id>')
def get_stops(cluster_id):
    print(f"Cluster ID recibido: {cluster_id}")
    stops_in_cluster = full_data[full_data['cluster'] == cluster_id][['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    return stops_in_cluster.to_json(orient='records')

# Nueva ruta para obtener detalles de una parada específica
@app.route('/get_stop_details/<string:stop_id>')
def get_stop_details(stop_id):
    # Filtrar los detalles de la parada por su identificador
    stop_details = full_data[full_data['stop_id'] == stop_id][['stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    
    # Verificar si se encontró la parada
    if stop_details.empty:
        return jsonify({'error': 'Stop not found'}), 404
    
    # Seleccionar la primera fila (ya que stop_id debería ser único)
    stop_details = stop_details.iloc[0]
    return jsonify({
        'stop_name': stop_details['stop_name'],
        'stop_lat': stop_details['stop_lat'],
        'stop_lon': stop_details['stop_lon'],
        'simulated_delay': stop_details['simulated_delay']
    })

@app.route('/')
def home():
    # En lugar de usar render_template_string con HTML en línea
    # ahora usamos render_template con el archivo HTML
    cluster_table = cluster_delays.to_frame().to_html(classes="table table-striped")
    top_table = top_delays[['stop_name', 'simulated_delay']].to_html(index=False, classes="table table-striped")
    
    # Filtrar solo los valores numéricos válidos y convertirlos a enteros
    cluster_ids = []
    for x in full_data['cluster'].unique():
        # Convertir a string para verificar si es un dígito
        x_str = str(x)
        # Filtrar solo valores que son dígitos y no son NaN
        if x_str.isdigit() and pd.notna(x):
            cluster_ids.append(int(x_str))
    
    # Ordenar los clústeres numéricamente
    cluster_ids.sort()

    # Imprimir para depuración
    print("Cluster IDs filtrados:", cluster_ids)
    
    num_clusters = len(set(cluster_ids))  # Usamos `set` para asegurarnos de que sean únicos
    
    map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
    
    print("map_center:", map_center)
    
    import json
    return render_template(
        'improved-template.html', 
        silhouette=silhouette_avg, 
        cluster_table=cluster_table, 
        top_table=top_table, 
        cluster_ids=json.dumps(cluster_ids),
        map_center=json.dumps(map_center),
        cluster_delays=cluster_delays,
        num_clusters=num_clusters # Se pasa el número de clústeres a la plantilla
    )

@app.route('/heatmap')
def heatmap():
    return send_from_directory('static', 'heatmap.html')

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(host='0.0.0.0', port=3000)