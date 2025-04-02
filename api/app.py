from flask import Flask, render_template_string, request, jsonify
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

# Nueva ruta para obtener las paradas por clúster
@app.route('/get_stops/<int:cluster_id>')
def get_stops(cluster_id):
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
    # Plantilla HTML con Bootstrap
    html = """
    <html>
    <head>
        <title>Incidencias RTP en la CDMX</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <!-- Leaflet CSS y JS -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 10px; }
            #map { height: 500px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center my-4">Incidencias en Transporte RTP</h1>
            
            <div class="row">
                <div class="col-md-6">
                    <h3>Métricas</h3>
                    <p><strong>Silhouette Score:</strong> {{ silhouette }}</p>
                </div>
                <div class="col-md-6">
                    <h3>Retraso Promedio por Clúster (minutos)</h3>
                    {{ cluster_table | safe }}
                </div>
            </div>

            <div class="row my-4">
                <div class="col-md-12">
                    <h3>Top 10 Paradas con Mayores Retrasos</h3>
                    {{ top_table | safe }}
                </div>
            </div>

            <div class="row my-4">
                <div class="col-md-6">
                    <h3>Seleccionar Clúster y Parada</h3>
                    <div class="mb-3">
                        <label for="cluster" class="form-label">Clúster:</label>
                        <select id="cluster" class="form-select" onchange="updateStops()">
                            {% for cluster_id in cluster_ids %}
                            <option value="{{ cluster_id }}">Clúster {{ cluster_id }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="stop" class="form-label">Parada:</label>
                        <select id="stop" class="form-select" onchange="updateMap()"></select>
                    </div>
                </div>
                <div class="col-md-6">
                    <h3>Mapa de Parada Seleccionada</h3>
                    <div id="map" class="border"></div>
                </div>
            </div>
        </div>

        <script>
            let map = L.map('map').setView([{{ map_center[0] }}, {{ map_center[1] }}], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19
            }).addTo(map);
            let marker;

            function updateStops() {
                const clusterId = $('#cluster').val();
                $.getJSON(`/get_stops/${clusterId}`, function(data) {
                    $('#stop').empty();
                    data.forEach(function(stop) {
                        $('#stop').append(`<option value="${stop.stop_id}">${stop.stop_name}</option>`);
                    });
                    updateMap();  // Actualizar el mapa con la primera parada
                });
            }

            function updateMap() {
                const stopId = $('#stop').val();
                $.getJSON(`/get_stop_details/${stopId}`, function(data) {
                    if (marker) {
                        map.removeLayer(marker);
                    }
                    marker = L.marker([data.stop_lat, data.stop_lon]).addTo(map)
                        .bindPopup(`<b>${data.stop_name}</b><br>Retraso: ${data.simulated_delay.toFixed(1)} min`).openPopup();
                    map.setView([data.stop_lat, data.stop_lon], 14);
                });
            }

            // Inicializar con el primer clúster
            $(document).ready(function() {
                updateStops();
            });
        </script>
    </body>
    </html>
    """
    cluster_table = cluster_delays.to_frame().to_html(classes="table table-striped")
    top_table = top_delays[['stop_name', 'simulated_delay']].to_html(index=False, classes="table table-striped")
    cluster_ids = full_data['cluster'].unique()
    map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
    return render_template_string(html, silhouette=silhouette_avg, cluster_table=cluster_table, top_table=top_table, cluster_ids=cluster_ids, map_center=map_center)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)