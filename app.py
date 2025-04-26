from flask import Flask
from routes.auth import auth_bp
from routes.main import main_bp
import os
# Crear la aplicación Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "tu-clave-secreta"

# Registrar los blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    app.run(host='0.0.0.0', port=3000)