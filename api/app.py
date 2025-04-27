from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import joblib
import os
import re
from datetime import datetime

# Crear la aplicación Flask con el directorio raíz explícito
app = Flask(__name__, template_folder='../templates', static_folder='../static')
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('templates'):
    os.makedirs('templates')

# Cargar y procesar datos de clustering
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

# Guardar modelo de clustering
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

# Cargar el modelo y vectorizador de PLN
try:
    comment_classifier = joblib.load("comment_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: No se encontraron los archivos comment_classifier.pkl o vectorizer.pkl.")
    comment_classifier = None
    vectorizer = None

try:
    comments_df = pd.read_csv("comments_processed.csv", encoding='utf-8')
except UnicodeDecodeError:
    print("Error de codificación UTF-8, intentando con 'latin1'...")
    comments_df = pd.read_csv("comments_processed.csv", encoding='latin1')
except FileNotFoundError:
    comments_df = pd.DataFrame(columns=[
        "comentarios", "source", "fecha", "tiene_groseria", 
        "comentarios_censurados", "etiqueta", "etiqueta_predicha", "relevancia"
    ])
# Lista de groserías y funciones para manejarlas
groserias = [
    "puto", "puta", "chinga", "cabrón", "cabron", "idiota", "estúpido", "estupido", 
    "mierda", "joder", "pendejo", "culero", "verga"
]
patron_groserias = "|".join([re.escape(groseria) for groseria in groserias])
patron_groserias = patron_groserias.replace("0", "[0o]").replace("e", "[e3]").replace("i", "[i1]")

def contiene_groseria(comentario):
    comentario = comentario.lower()
    return bool(re.search(patron_groserias, comentario))

def censurar_groseria(comentario):
    comentario_lower = comentario.lower()
    for groseria in groserias:
        censura = "*" * len(groseria)
        patron = groseria.replace("e", "[e3]").replace("i", "[i1]").replace("o", "[0o]")
        comentario = re.sub(patron, censura, comentario, flags=re.IGNORECASE)
    return comentario

# Ruta para clasificar nuevos comentarios
@app.route('/classify_comment', methods=['POST'])
def classify_comment():
    global comments_df

    data = request.get_json()
    nuevo_comentario = data.get('comment', '')

    if not nuevo_comentario:
        return jsonify({'error': 'No se proporcionó un comentario'}), 400

    tiene_groseria = contiene_groseria(nuevo_comentario)
    comentario_censurado = censurar_groseria(nuevo_comentario)

    vector = vectorizer.transform([comentario_censurado])
    prediccion = comment_classifier.predict(vector)[0]
    relevancia = vector.sum()

    fecha_actual = datetime(2025, 4, 18).strftime("%Y-%m-%d")
    nuevo_registro = {
        "comentarios": nuevo_comentario,
        "source": "Formulario Web",
        "fecha": fecha_actual,
        "tiene_groseria": tiene_groseria,
        "comentarios_censurados": comentario_censurado,
        "etiqueta": "negativo" if tiene_groseria else prediccion,
        "etiqueta_predicha": prediccion,
        "relevancia": relevancia
    }

    comments_df = pd.concat([comments_df, pd.DataFrame([nuevo_registro])], ignore_index=True)
    comments_df.to_csv("comments_processed.csv", index=False, encoding='utf-8')

    return jsonify({
        'comment': nuevo_comentario,
        'censored_comment': comentario_censurado,
        'category': prediccion,
        'relevance': relevancia
    })

# Ruta para descargar el CSV
@app.route('/download_comments_csv')
def download_comments_csv():
    csv_path = "comments_processed.csv"
    comments_df.to_csv(csv_path, index=False, encoding='utf-8')
    return send_file(
        csv_path,
        as_attachment=True,
        download_name="comments_processed.csv",
        mimetype="text/csv"
    )

# Rutas existentes
@app.route('/get_stops/<int:cluster_id>')
def get_stops(cluster_id):
    print(f"Cluster ID recibido: {cluster_id}")
    stops_in_cluster = full_data[full_data['cluster'] == cluster_id][['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    return stops_in_cluster.to_json(orient='records')

@app.route('/get_stop_details/<string:stop_id>')
def get_stop_details(stop_id):
    stop_details = full_data[full_data['stop_id'] == stop_id][['stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    if stop_details.empty:
        return jsonify({'error': 'Stop not found'}), 404
    stop_details = stop_details.iloc[0]
    return jsonify({
        'stop_name': stop_details['stop_name'],
        'stop_lat': stop_details['stop_lat'],
        'stop_lon': stop_details['stop_lon'],
        'simulated_delay': stop_details['simulated_delay']
    })

@app.route('/')
def home():
    # Preparar datos para las cards
    positivos = comments_df[comments_df["etiqueta_predicha"] == "positivo_sugerencia"]
    top_positivos = positivos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_positivos = [item["comentarios_censurados"] for item in top_positivos]

    negativos = comments_df[comments_df["etiqueta_predicha"] == "negativo"]
    top_negativos = negativos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_negativos = [item["comentarios_censurados"] for item in top_negativos]

    # Añadir comentarios neutros
    neutros = comments_df[comments_df["etiqueta_predicha"] == "neutral"]
    top_neutros = neutros.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_neutros = [item["comentarios_censurados"] for item in top_neutros]

    # Resto del código existente
    cluster_table = cluster_delays.to_frame().to_html(classes="table table-striped")
    top_table = top_delays[['stop_name', 'simulated_delay']].to_html(index=False, classes="table table-striped")
    
    cluster_ids = []
    for x in full_data['cluster'].unique():
        try:
            # Convertir a entero y asegurarse de que no sea NaN
            x_int = int(float(x))  # Convertir primero a float para manejar valores como "1.0"
            if not pd.isna(x_int):
                cluster_ids.append(x_int)
        except (ValueError, TypeError):
            continue  # Ignorar valores que no se puedan convertir a entero
    cluster_ids.sort()
    
    num_clusters = len(set(cluster_ids))
    map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
    
    import json
    return render_template(
        'improved-template.html', 
        silhouette=silhouette_avg, 
        cluster_table=cluster_table, 
        top_table=top_table, 
        cluster_ids=json.dumps(cluster_ids),
        map_center=json.dumps(map_center),
        cluster_delays=cluster_delays,
        num_clusters=num_clusters,
        top_positivos=top_positivos,
        top_negativos=top_negativos,
        top_neutros=top_neutros
    )

@app.route('/heatmap')
def heatmap():
    return send_from_directory('static', 'heatmap.html')

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(host='0.0.0.0', port=3000)