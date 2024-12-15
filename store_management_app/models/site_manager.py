from store_management_app import (
    app,
    login_required,
    allowed_file,
    secure_filename,
    base64,
    product_detailes_path,
    image_to_base64,
    image_save_path,
    generate_unique_id,
    capitalize_sentences,
    social_sites_path,
)
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
from store_management_app import product_category_path

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

# Create an instance of DataManager
data_manager = DataManager(
    product_detailes_path, social_sites_path, product_category_path
)


@app.route("/sitemanager", methods=["POST", "GET"])
@login_required
def site_manager():
    """
    Route for the site manager page.

    Returns:
        Rendered 'site_manager.html' template.
    """
    return render_template(
        "sitemanagement/siteManagement.html",
        user=session.get("username"),
        traffic_data=DataManager.get_traffic_count(),
    )


@app.route("/socialmediamanager", methods=["POST", "GET"])
@login_required
def social_media_manager():
    social_sites = data_manager.get_social_sites()

    # Unpack the social sites into individual variables
    facebook = social_sites[0]["link"]
    twitter = social_sites[1]["link"]
    instagram = social_sites[2]["link"]
    linkedin = social_sites[3]["link"]
    whatsapp = social_sites[4]["link"]
    youtube = social_sites[5]["link"]
    mobile = social_sites[6]["link"]
    try:
        social_site_list = data_manager.get_social_sites()
        return render_template(
            "sitemanagement/social_sites_edit.html",
            social_site_list=social_site_list,
            user=session.get("username"),
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )
    except FileNotFoundError as e:
        return render_template("other/error.html", error_message=str(e))


# update the social media sites
@app.route("/update_site", methods=["POST"])
def update_site():
    data = request.json
    site = data.get("site")
    link = data.get("link")
    df = pd.read_excel(social_sites_path)
    if site.lower() in df["site"].values:
        df.loc[df["site"] == site.lower(), "link"] = link
        df.to_excel(social_sites_path, index=False)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Site not found"}), 404


@app.route("/uploadproduct", methods=["POST"])
@login_required
def upload_product():
    """
    Route for uploading a new product.

    This function reads the form data for the new product, saves the image,
    converts it to base64, appends the product details to the existing data,
    saves the data to an Excel file, and returns a JSON response.

    Returns:
        JSON response with success status.
    """
    social_sites = data_manager.get_social_sites()

    # Unpack the social sites into individual variables
    facebook = social_sites[0]["link"]
    twitter = social_sites[1]["link"]
    instagram = social_sites[2]["link"]
    linkedin = social_sites[3]["link"]
    whatsapp = social_sites[4]["link"]
    youtube = social_sites[5]["link"]
    mobile = social_sites[6]["link"]
    try:
        # Retrieve form data for the new product
        PrimaryCategory = request.form.get("PrimaryCategory")
        Tagline = request.form.get("category")
        name = request.form.get("name")
        price = request.form.get("price")
        discount = request.form.get("discount")
        image = request.files.get("image")
        description = request.form.get("description")
        uploaded_date_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Load existing products
        products_df = pd.read_excel(
            product_detailes_path
        )  # Replace with your actual path

        # Check if an image was uploaded
        if image:
            # Assuming 'image' is the object containing the uploaded file
            image_filename, image_extension = os.path.splitext(image.filename)

            # Generate the unique filename
            unique_filename = f"{PrimaryCategory}_{name}_{image_filename.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}{image_extension}"

            # Construct the full path to save the image
            image_to_save_path = os.path.join(
                "store_management_app", "static", "assets", "images", unique_filename
            )
            image.save(image_to_save_path)

            # Convert image to Base64
            image_base64 = image_to_base64(image)
        else:
            # If no image was uploaded, set image_base64 to None
            image_base64 = None

        # Create a DataFrame for the new product
        new_product_df = pd.DataFrame(
            [
                {
                    "ID": generate_unique_id(),  # Generate a unique ID for the product
                    "Primary_Category_Alt": PrimaryCategory.title(),  # Primary category of the product
                    "Tagline": Tagline,  # Category of the product
                    "Name": name.title(),  # Name of the product
                    "Price": price,  # Price of the product
                    "Discount": discount,  # Discount of the product
                    "Image_Location": unique_filename,  # Location of the image file
                    "Image_Base64": image_base64,  # Base64 representation of the image
                    "Description": description,  # Description of the product
                    "upload_date_time": uploaded_date_time,
                }
            ]
        )

        # Append the new product to the existing DataFrame
        updated_products_df = pd.concat(
            [products_df, new_product_df], ignore_index=True
        )

        # Save the updated DataFrame to the Excel file
        updated_products_df.to_excel(
            product_detailes_path, index=False
        )  # Replace with your actual path

        # Return a JSON response with success status
        return jsonify(success=True)
    except Exception as e:
        # If an error occurs, print the error and return a JSON response with an error code
        print(f"Error: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/updateproduct", methods=["POST"])
@login_required
def update_product():
    """
    Update product details based on the POST request data.

    This function loads existing products, updates the product details, saves the changes to the Excel file,
    and redirects to the products management page.

    Returns:
        Response: JSON response indicating success or failure.
    """
    try:
        # Load existing products
        products_df = pd.read_excel(product_detailes_path)
        image = request.files.get("editImage")
        product_id = request.form.get("editProductId")
        if not product_id:
            print("Product ID is missing")
            return jsonify(success=False, error="Product ID is missing"), 400
        PrimaryCategory = request.form.get("editPrimaryCategory")
        category = request.form.get("editCategory")
        name = request.form.get("editName")
        price = request.form.get("editPrice")
        discount = request.form.get("editDiscount")
        description = request.form.get("editDescription")
        updated_date_time = datetime.now().astimezone(usa).strftime("%d-%m-%Y %H:%M:%S")
        if image and image.filename:
            if allowed_file(image.filename):
                image_to_save_path = os.path.join(
                    "store_management_app",
                    "static",
                    "assets",
                    "images",
                    secure_filename(image.filename),
                )
                image.save(image_to_save_path)
                image_base64 = image_to_base64(image)
                Image_Location = image.filename
                # Update the DataFrame including the image location
                products_df.loc[
                    products_df["ID"] == str(product_id),
                    [
                        "Primary_Category_Alt",
                        "Tagline",
                        "Name",
                        "Price",
                        "Discount",
                        "Image_Location",
                        "Description",
                        "upload_date_time",
                    ],
                ] = [
                    PrimaryCategory,
                    category,
                    name,
                    price,
                    discount,
                    Image_Location,
                    description,
                    updated_date_time,
                ]
                products_df[["Name", "Primary_Category_Alt"]] = products_df[
                    ["Name", "Primary_Category_Alt"]
                ].applymap(lambda x: x.title())
                products_df[["Description", "Tagline"]] = products_df[
                    ["Description", "Tagline"]
                ].applymap(capitalize_sentences)

                # Write back to the file
                products_df.to_excel(product_detailes_path, index=False)
            else:
                return jsonify(success=False, error="File not allowed"), 400
        else:
            # Update the DataFrame without the image location
            products_df.loc[
                products_df["ID"] == str(product_id),
                [
                    "Primary_Category_Alt",
                    "Tagline",
                    "Name",
                    "Price",
                    "Discount",
                    "Description",
                    "upload_date_time",
                ],
            ] = [
                PrimaryCategory,
                category,
                name,
                price,
                discount,
                description,
                updated_date_time,
            ]
            products_df[["Name", "Primary_Category_Alt"]] = products_df[
                ["Name", "Primary_Category_Alt"]
            ].applymap(lambda x: x.title())
            products_df[["Description", "Tagline"]] = products_df[
                ["Description", "Tagline"]
            ].applymap(capitalize_sentences)

            # Write back to the file
            products_df.to_excel(product_detailes_path, index=False)

        return jsonify(success=True)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/deleteproduct", methods=["POST"])
@login_required
def delete_product():
    """
    Delete a product from the database and remove its image file.

    This function receives a POST request with a JSON payload containing the
    product ID and its image location. It then reads the existing products
    from the Excel file, filters out the product to be deleted, saves the
    updated DataFrame back to the Excel file, and deletes the corresponding
    image file.

    Returns:
        A JSON response indicating the success of the operation.
    """
    try:
        # Get the product ID and image location from the request payload
        data = request.json
        product_id_to_delete = data.get("row_ID")
        image_location = data.get("imageLocation")

        # Load existing products
        products_df = pd.read_excel(product_detailes_path)

        # Filter out the product to be deleted
        products_df = products_df[products_df["ID"] != product_id_to_delete]

        # Save the updated DataFrame to the Excel file
        products_df.to_excel(product_detailes_path, index=False)

        # Optionally, delete the image file
        image_path = os.path.join(image_save_path, image_location)
        print(image_path)
        if os.path.exists(image_path.replace("\\", "/")):
            os.remove(image_path)

        # Send back a JSON response
        return jsonify(success=True)

    except Exception as e:
        # Handle any exceptions and return an error response
        print(f"Error: {e}")
        return jsonify(success=False), 500


@app.route("/stockupdateproduct", methods=["POST"])
@login_required
def stock_update_product():
    """
    Update product details based on the POST request data.
    """
    try:
        # Load existing products
        products_df = pd.read_excel(product_detailes_path)
        data = request.get_json()  # Get JSON data from request

        product_id = data.get("id")
        status = data.get("status")  # Get the status from JSON data

        if not product_id or status is None:
            return jsonify(success=False, error="Product ID or status is missing"), 400
        # Update the product status in the DataFrame
        products_df.loc[products_df["ID"] == str(product_id), "Status"] = status

        # Write back to the file
        products_df.to_excel(product_detailes_path, index=False)
        return jsonify(success=True)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/productsmanagement", methods=["POST", "GET"])
@login_required
def products_management():
    social_sites = data_manager.get_social_sites()

    # Unpack the social sites into individual variables
    facebook = social_sites[0]["link"]
    twitter = social_sites[1]["link"]
    instagram = social_sites[2]["link"]
    linkedin = social_sites[3]["link"]
    whatsapp = social_sites[4]["link"]
    youtube = social_sites[5]["link"]
    mobile = social_sites[6]["link"]
    try:
        products_list = data_manager.get_product_details()
        primary_category = list(
            set([product["Primary_Category_Alt"] for product in products_list])
        )
        all_primary_category = data_manager.get_product_categories()
        return render_template(
            "/products/new_product_upload.html",
            products_list=products_list,
            PrimaryCategory=primary_category,
            AllPrimaryCategory=all_primary_category,
            user=session.get("username"),
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )
    except FileNotFoundError as e:
        return render_template("other/error.html", error_message=str(e))


@app.route("/Productcategorymanager", methods=["POST", "GET"])
@login_required
def ProductCategoryManager():
    social_sites = data_manager.get_social_sites()

    # Unpack the social sites into individual variables
    facebook = social_sites[0]["link"]
    twitter = social_sites[1]["link"]
    instagram = social_sites[2]["link"]
    linkedin = social_sites[3]["link"]
    whatsapp = social_sites[4]["link"]
    youtube = social_sites[5]["link"]
    mobile = social_sites[6]["link"]
    try:
        productCat_list = pd.read_excel(product_category_path)
        return render_template(
            "sitemanagement/product_categories_manager.html",
            productCat_list=productCat_list.to_dict(orient="records"),
            user=session.get("username"),
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )
    except FileNotFoundError as e:
        return render_template("other/error.html", error_message=str(e))


@app.route("/newprimarycategoryaddition", methods=["POST"])
@login_required
def newprimarycatadd():
    """
    Route for adding a new primary category to the product details.

    This function reads the form data for the new primary category,
    adds it to the list of primary categories, and redirects to the
    products management page.

    Returns:
        Redirect to the products management page.
    """
    # Read the form data for the new primary category
    new_category = request.form.get("primaryCategory")

    product_category_df = pd.read_excel(product_category_path)
    # Check if the new category is not None or empty
    if not new_category:
        return redirect(url_for("products_managememnt"))

    # Determine the new id
    if not product_category_df.empty:
        new_id = product_category_df["id"].max() + 1
    else:
        new_id = 1
    # Create a DataFrame for the new product
    new_product_category_df = pd.DataFrame(
        [
            {
                "id": new_id,
                "product_category": new_category,  # Primary category of the product
            }
        ]
    )

    # Append the new product to the existing DataFrame
    updated_products_category_df = pd.concat(
        [product_category_df, new_product_category_df], ignore_index=True
    )
    updated_products_category_df["product_category"] = updated_products_category_df[
        "product_category"
    ].apply(lambda x: x.title())
    # Save the updated DataFrame to the Excel file
    updated_products_category_df.to_excel(product_category_path, index=False)

    # Redirect to the products management page
    return jsonify(success=True)


@app.route("/editproductcat", methods=["POST"])
@login_required
def edit_product_cat():
    """
    Route for editing a product category.

    This function updates the product category details in the existing data
    and saves the updated data to an Excel file.

    Returns:
        JSON response with success status.
    """
    print(request.form)
    try:
        # Retrieve form data for the edited product category
        product_cat = request.form.get("editproductCat")
        product_cat_id = request.form.get("editproductCatId")
        # Load existing product categories
        product_cats_df = pd.read_excel(
            product_category_path
        )  # Replace with your actual path
        # Update the product category in the DataFrame
        product_cats_df["product_category"][
            product_cats_df["id"] == int(product_cat_id)
        ] = product_cat
        product_cats_df["product_category"] = product_cats_df["product_category"].apply(
            lambda x: x.title()
        )
        # Save the updated DataFrame to the Excel file
        product_cats_df.to_excel(
            product_category_path, index=False
        )  # Replace with your actual path

        # Return a JSON response with success status
        return jsonify(success=True)
    except Exception as e:
        # If an error occurs, print the error and return a JSON response with an error code
        print(f"Error: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route("/deleteproductcat", methods=["DELETE"])
@login_required
def delete_product_cat():
    """
    Route for deleting a product category.

    This function deletes the product category from the existing data
    and saves the updated data to an Excel file.

    Returns:
        JSON response with success status.
    """
    data = request.json
    product_id_to_delete = data.get("row_ID")
    try:
        # Load existing product categories
        product_cats_df = pd.read_excel(product_category_path)
        product_cats_df = product_cats_df[
            product_cats_df["id"] != int(product_id_to_delete)
        ]
        # Save the updated DataFrame to the Excel file
        product_cats_df.to_excel(product_category_path, index=False)
        # Return a JSON response with success status
        return jsonify(success=True)
    except Exception as e:
        # If an error occurs, print the error and return a JSON response with an error code
        print(f"Error: {e}")
        return jsonify(success=False, error=str(e)), 500
