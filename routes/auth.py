import requests
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify

# Crear un blueprint para las rutas de autenticación
auth_bp = Blueprint('auth', __name__)

# URL base de la API FastAPI
API_BASE_URL = "http://127.0.0.1:8000"

# Función para registrar un usuario en la API FastAPI
def register_user(email, password):
    url = f"{API_BASE_URL}/user/"  # Cambiar /register/ por /user/
    payload = {"email": email, "password": password}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return True
    else:
        raise Exception(f"Error al registrar usuario: {response.status_code} - {response.text}")

# Función para obtener el token JWT desde la API FastAPI
def get_api_token(email, password):
    url = f"{API_BASE_URL}/login/"
    payload = {"email": email, "password": password}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Error al obtener token: {response.status_code} - {response.text}")

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return render_template('login.html', error="Por favor, ingresa email y contraseña")

        try:
            token = get_api_token(email, password)
            session['api_token'] = token
            session['email'] = email
            return redirect(url_for('main.home'))
        except Exception as e:
            return render_template('login.html', error=str(e))

    return render_template('login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return render_template('signup.html', error="Por favor, ingresa email y contraseña")

        try:
            register_user(email, password)
            # Una vez registrado, iniciamos sesión automáticamente
            token = get_api_token(email, password)
            session['api_token'] = token
            session['email'] = email
            return redirect(url_for('main.home'))
        except Exception as e:
            return render_template('signup.html', error=str(e))

    return render_template('signup.html')

@auth_bp.route('/logout')
def logout():
    session.pop('api_token', None)
    session.pop('email', None)
    return redirect(url_for('auth.login'))