from store_management_app import app, login_required

from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    render_template,
    session,
    jsonify,
)
from store_management_app.models.products_categories_data import (
    primary_products_cat,
    product_category_path,
)
from store_management_app.controllers.controller import DataManager

import os
import json
import pandas as pd
from datetime import datetime
import pytz
import base64
from PIL import Image
from io import BytesIO

ist = pytz.timezone("Asia/Kolkata")
cet = pytz.timezone("Europe/Paris")
usa = pytz.timezone("US/Eastern")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Route decorator for the '/login' endpoint. Handles both GET and POST requests.

    Parameters:
        None

    Returns:
        If the request method is POST:
            - If the entered username and password match the expected values, store the username and role in the session and render the 'layout/layout.html' template with the username.
            - If the entered username and password do not match the expected values, render the 'login.html' template.
        If the request method is not POST, render the 'login.html' template.

    """
    if request.method == "POST":
        # Get the entered username and password from the form
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the username and password are not None and match the expected values
        if (
            username
            and password
            and (username == "binnyfashion")
            and (password == "Binny@123")
        ):
            # Passwords match, store username and role in session
            session["username"] = username
            session["role"] = "admin"  # Assuming the role is stored in the database

            return render_template(
                "/sitemanagement/siteManagement.html",
                user=session["username"]
            )  # Redirect to the dashboard upon successful login
        else:
            return render_template("login.html")
    else:
        return render_template("login.html")

@app.route("/sitemanagement")
def sitemanagement():
    return render_template("sitemanagement/siteManagement.html")
@app.route("/logout")
@login_required
def logout():
    """
    Logout the user by clearing the session and redirecting to the login page.

    Returns:
        Rendered 'login.html' template.
    """
    # Clear the session to logout the user
    session.clear()

    # Redirect to the login page
    return render_template("login.html")
