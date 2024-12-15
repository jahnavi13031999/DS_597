from flask import Flask, flash, request, redirect, url_for, render_template, session
from flask_bcrypt import Bcrypt
from flask_session import Session
from datetime import datetime
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
from collections import Counter
from dotenv import load_dotenv, dotenv_values
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import urllib.request
import os
import base64
import numpy as np
import base64
import json
import concurrent.futures
import pandas as pd
import urllib.parse
import sqlalchemy
import numpy as np
import shutil
import glob
import urllib.parse

# import psycopg2
import sqlalchemy
import warnings
import time
import jwt
import uuid
import re

import plotly.express as px
import plotly.io as pio

import logging

logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
app = Flask("store_management_app")
bcrypt = Bcrypt(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

product_detailes_path = os.path.join("Detailes", "products_detailes_img.xlsx")
product_category_path = os.path.join("Detailes", "product_categories.xlsx")
social_sites_path = os.path.join("Detailes", "social_media.xlsx")
image_save_path = os.path.join("store_management_app", "static", "assets", "images")

app.secret_key = "miojcoijewnnodiwe043t546545dsc32r"
# app.config['SECRET_KEY'] = os.urandom(32).hex()
# app.config['PERMANENT_SESSION_LIFETIME'] = datetime.datetime.timedelta(minutes=5)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# image to base64 conversion
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def allowed_image_file(filename):
    """
    Checks if the filename extension is allowed for images.

    Args:
        filename (str): The filename of the uploaded file.

    Returns:
        bool: True if the extension is allowed, False otherwise.
    """
    # (Assuming you have the allowed_image_file function defined elsewhere)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_id():
    # Generate a unique UUID (Universal Unique Identifier)
    return str(uuid.uuid4())


# Define the capitalize_sentences function
def capitalize_sentences(text):
    if pd.isna(text):  # Check if the value is NaN
        return " "

    # Function to capitalize the first letter of each sentence
    def capitalize_match(match):
        return match.group(0).capitalize()

    # Regular expression to match the beginning of a sentence
    sentence_pattern = re.compile(r"(?:^|(?<=[.!?])\s+)([a-z])")

    # Apply the capitalize_match function to each match
    return sentence_pattern.sub(lambda m: m.group(0).capitalize(), text)


from store_management_app.pages.navigations import *
from store_management_app.models.site_manager import *
from store_management_app.models.user_management import *
