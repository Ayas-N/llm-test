from flask import Flask, render_template, request, jsonify, session
from app_blueprint import app_blueprint
from flask_socketio import SocketIO 
import eventlet


socketio = SocketIO(cors_allowed_origins="*")
def create_app():
    app = Flask(__name__, static_folder = "static")
    socketio.__init__(app)
    app.config['SECRET_KEY'] = 'THEREISNOSECRET'
    app.config['SESSION_TYPE'] = 'filesystemsession'
    app.register_blueprint(app_blueprint)
    socketio.run(app, debug = True, keyfile = 'certs/website.test.key', certfile = 'certs/website.test.crt')

if __name__ == "__main__":
    app = create_app()
    eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 8000), app))