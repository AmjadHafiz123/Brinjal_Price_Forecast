from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import routes 
from app.api import *