# Gevent needed for sockets
from gevent import monkey
monkey.patch_all()

# Imports
import os
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

# Configure app
socketio = SocketIO()
app = Flask(__name__, static_folder = 'static')
app.config.from_object(os.environ["APP_SETTINGS"])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
db = SQLAlchemy(app)

# Import + Register Blueprints
from app.accounts import accounts as accounts
app.register_blueprint(accounts)
from app.irsystem import irsystem as irsystem
app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

# Home
@app.route('/')
def home():
  return render_template("search.html")

# Random Cereal Generator
@app.route('/random', methods=['GET'])
def random_generator():
  return render_template("RandomCereal.html")


# Compare Cereals
@app.route('/compare', methods=['GET'])
def compare_generator():
  return render_template("comparecereals.html")

# Top 100
@app.route('/top', methods=['GET'])
def top_generator():
  return render_template("Top100.html")

# Search Results
# @app.route('/search', methods=['GET', 'POST'])
# def searchresults_generator():
#   if method == 'POST':
#     query = request.form['input']

#   return render_template("SearchResults.html")

# HTTP error handling
@app.errorhandler(404)
def not_found(error):
  return render_template("404.html"), 404
