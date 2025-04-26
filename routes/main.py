import requests
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import os

main_bp = Blueprint('main', __name__)

API_BASE_URL = "http://127.0.0.1:8000"

@main_bp.route('/')
def home():
    if 'api_token' not in session:
        return redirect(url_for('auth.login'))

    # Obtener métricas y comentarios clasificados desde la API
    url = f"{API_BASE_URL}/data/metrics"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {session['api_token']}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 401:
            session.pop('api_token', None)
            session.pop('email', None)
            return redirect(url_for('auth.login'))
        response.raise_for_status()
        metrics = response.json()

        # Extraer datos del response
        silhouette_avg = metrics['silhouette_avg']
        cluster_delays = metrics['cluster_delays']  # Esto es una lista de diccionarios
        top_delays = pd.DataFrame(metrics['top_delays'])
        map_center = metrics['map_center']
        cluster_ids = metrics['cluster_ids']
        num_clusters = metrics['num_clusters']
        top_positivos = metrics['top_positivos']
        top_negativos = metrics['top_negativos']
        top_neutros = metrics['top_neutros']

        # Calcular el promedio de average_delay
        if cluster_delays:
            average_delay = sum(item['average_delay'] for item in cluster_delays) / len(cluster_delays)
        else:
            average_delay = 0

        # Generar la tabla de retrasos por clúster (cluster_table)
        cluster_table = '<table class="table table-striped"><thead><tr><th>Cluster</th><th>Retraso Promedio (minutos)</th></tr></thead><tbody>'
        for item in cluster_delays:
            cluster = item['cluster']
            delay = item['average_delay']
            cluster_table += f'<tr><td>{cluster}</td><td>{delay:.2f}</td></tr>'
        cluster_table += '</tbody></table>'

        # Generar la tabla de top 10 retrasos
        top_table = top_delays[['stop_name', 'simulated_delay']].to_html(index=False, classes="table table-striped")
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener métricas desde la API: {str(e)}")
        # Valores por defecto en caso de error
        silhouette_avg = 0
        cluster_table = "<p>Error al cargar los datos.</p>"
        top_table = "<p>Error al cargar los datos.</p>"
        cluster_ids = []
        map_center = [0, 0]
        average_delay = 0
        num_clusters = 0
        top_positivos = []
        top_negativos = []
        top_neutros = []

    return render_template(
        'improved-template.html', 
        silhouette=silhouette_avg, 
        cluster_table=cluster_table, 
        top_table=top_table, 
        cluster_ids=cluster_ids,
        map_center=map_center,
        cluster_delays=average_delay,  # Pasamos el promedio calculado
        num_clusters=num_clusters,
        email=session.get('email', 'Usuario'),
        top_positivos=top_positivos,
        top_negativos=top_negativos,
        top_neutros=top_neutros,
        session=session
    )

@main_bp.route('/classify_comment', methods=['POST'])
def classify_comment():
    if 'api_token' not in session:
        return jsonify({'error': 'Por favor, inicia sesión'}), 401

    comment = request.form.get('comment', '')
    if not comment:
        return jsonify({'error': 'Por favor, ingresa un comentario'}), 400

    url = f"{API_BASE_URL}/data/classify_comment"
    payload = {"comment": comment}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {session['api_token']}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 401:
            session.pop('api_token', None)
            session.pop('email', None)
            return jsonify({'error': 'Sesión expirada, por favor inicia sesión de nuevo'}), 401
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error al clasificar el comentario: {str(e)}"}), 500

@main_bp.route('/get_stops/<int:cluster_id>')
def get_stops(cluster_id):
    url = f"{API_BASE_URL}/data/stops/{cluster_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {session['api_token']}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 401:
            session.pop('api_token', None)
            session.pop('email', None)
            return redirect(url_for('auth.login'))
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error al obtener paradas: {str(e)}"}), 500

@main_bp.route('/get_stop_details/<string:stop_id>')
def get_stop_details(stop_id):
    url = f"{API_BASE_URL}/data/stop_details/{stop_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {session['api_token']}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 401:
            session.pop('api_token', None)
            session.pop('email', None)
            return redirect(url_for('auth.login'))
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"Error al obtener detalles de la parada: {str(e)}"}), 500

@main_bp.route('/heatmap')
def heatmap():
    if 'api_token' not in session:
        return redirect(url_for('auth.login'))
    return send_from_directory('static', 'heatmap.html')