from flask import Flask
from flask_sqlalchemy import SQLAlchemy 
import os

# fernet key for encrypting and decrypting data
# fernet = Fernet(os.getenv("FERNET_KEY"))

db = SQLAlchemy()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

app.config['SECRET_KEY'] = os.getenv("SQL_ACLCHEMY_KEY")

from core.models import User

from core.views import home

app.register_blueprint(home.home)
