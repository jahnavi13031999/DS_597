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
    logger,
    social_sites_path,
    px,
    pio,
    timedelta,
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
    Response,
)
from store_management_app.models.products_categories_data import (
    primary_products_cat,
    product_category_path,
)
from store_management_app.models.site_manager import data_manager

import os
import json
import pandas as pd
from datetime import datetime
import pytz
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

ist = pytz.timezone("Asia/Kolkata")
cet = pytz.timezone("Europe/Paris")
usa = pytz.timezone("US/Eastern")


social_sites = data_manager.get_social_sites()

# Unpack the social sites into individual variables
facebook = social_sites[0]["link"]
twitter = social_sites[1]["link"]
instagram = social_sites[2]["link"]
linkedin = social_sites[3]["link"]
whatsapp = social_sites[4]["link"]
youtube = social_sites[5]["link"]
mobile = social_sites[6]["link"]


@app.route("/aboutus")
def aboutus():
    """
    Route for rendering the About Us page.

    This function is responsible for handling the '/aboutus' route and
    rendering the 'About_Us.html' template. It is used to display the
    information about the company on the 'About Us' page.

    Returns:
        A rendered template of the 'About_Us.html' page.
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
    # Render the 'About_Us.html' template and return it as the response
    return render_template(
        # "/other/About_Us.html",
        "sitemanagement/aboutus_edit.html",
        user=session.get("username"),
        facebook=facebook,
        twitter=twitter,
        instagram=instagram,
        linkedin=linkedin,
        whatsapp=whatsapp,
        youtube=youtube,
        mobile=mobile,
    )


@app.route("/sitemap.xml")
def sitemap():
    pages = [
        {"loc": "/", "lastmod": "2024-08-12", "changefreq": "daily", "priority": "1.0"},
        {
            "loc": "/products/All",
            "lastmod": "2024-08-12",
            "changefreq": "weekly",
            "priority": "0.8",
        },
        {
            "loc": "/products/Clearance%20Sale,%20Limited%20Time%20Only",
            "lastmod": "2024-08-12",
            "changefreq": "weekly",
            "priority": "0.8",
        },
        {
            "loc": "/products/Popular%20Items",
            "lastmod": "2024-08-12",
            "changefreq": "weekly",
            "priority": "0.8",
        },
        {
            "loc": "/products/Lehenga%20Choli%20Set%20For%20Women",
            "lastmod": "2024-08-12",
            "changefreq": "weekly",
            "priority": "0.8",
        },
        {
            "loc": "/products/Sarees%20For%20Women",
            "lastmod": "2024-08-12",
            "changefreq": "weekly",
            "priority": "0.8",
        },
        {
            "loc": "/about-us/",
            "lastmod": "2024-08-12",
            "changefreq": "monthly",
            "priority": "0.5",
        },
        {
            "loc": "/contact-us/",
            "lastmod": "2024-08-12",
            "changefreq": "monthly",
            "priority": "0.5",
        },
    ]

    sitemap_xml = render_sitemap(pages)
    return Response(sitemap_xml, mimetype="application/xml")


def render_sitemap(pages):
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for page in pages:
        xml += "  <url>\n"
        xml += f'    <loc>{request.host_url[:-1]}{page["loc"]}</loc>\n'
        xml += f'    <lastmod>{page["lastmod"]}</lastmod>\n'
        xml += f'    <changefreq>{page["changefreq"]}</changefreq>\n'
        xml += f'    <priority>{page["priority"]}</priority>\n'
        xml += "  </url>\n"
    xml += "</urlset>"
    return xml


@app.route("/site_map")
def site_map():
    return render_template(
        "other/sitemap.html",
        user=session.get("username"),
        facebook=facebook,
        twitter=twitter,
        instagram=instagram,
        linkedin=linkedin,
        whatsapp=whatsapp,
        youtube=youtube,
        mobile=mobile,
    )


@app.route("/contactus")
def contactus():
    """
    Route for rendering the Contact Us page.

    This function is responsible for handling the '/contactus' route and
    rendering the 'Contact_Us.html' template. It is used to display the
    contact information of the company on the 'Contact Us' page.

    Returns:
        A rendered template of the 'Contact_Us.html' page.
    """
    # Render the 'Contact_Us.html' template and return it as the response
    social_sites = data_manager.get_social_sites()

    # Unpack the social sites into individual variables
    facebook = social_sites[0]["link"]
    twitter = social_sites[1]["link"]
    instagram = social_sites[2]["link"]
    linkedin = social_sites[3]["link"]
    whatsapp = social_sites[4]["link"]
    youtube = social_sites[5]["link"]
    mobile = social_sites[6]["link"]
    return render_template(
        "/other/Contact_Us.html",
        user=session.get("username"),
        facebook=facebook,
        twitter=twitter,
        instagram=instagram,
        linkedin=linkedin,
        whatsapp=whatsapp,
        youtube=youtube,
        mobile=mobile,
    )


@app.route("/")
def products_homepage():
    """
    Route for rendering the product homepage.

    This function is responsible for handling the root route ('/') and
    rendering the 'products/product.html' template. It reads the Excel file
    containing product details and sends it as a JSON response to the
    client. If the file is not found or an unexpected error occurs,
    it renders an error template with an appropriate message.

    Returns:
        A rendered template of the 'products/product.html' page, or an
        error template in case of an error.
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
        # Read the Excel file containing product details
        products_df = pd.read_excel(product_detailes_path)
        products_df = products_df[products_df["Status"] == "Active"]
        # Fill missing values with a space
        products_df.fillna(" ", inplace=True)
        products_df[["Name", "Primary_Category_Alt"]] = products_df[
            ["Name", "Primary_Category_Alt"]
        ].applymap(lambda x: x.title() if isinstance(x, str) else x)
        products_df[["Description", "Tagline"]] = products_df[
            ["Description", "Tagline"]
        ].applymap(capitalize_sentences)
        # Define custom category order
        category_order = ["ALL", "Popular Items", "New Arrival"]

        # Create a new column 'CategoryOrder' for sorting
        products_df["CategoryOrder"] = products_df["Category"].apply(
            lambda x: (
                category_order.index(x) if x in category_order else len(category_order)
            )
        )
        # Sort DataFrame based on 'CategoryOrder' and then drop the helper column
        products_df.sort_values("CategoryOrder", ascending=True, inplace=True)
        # Get a list of unique primary categories
        PrimaryCategory = ["All"] + list(set(products_df["Primary_Category_Alt"]))
        # Render the 'products/product.html' template with the product data
        if session.get("username") is not None:
            # and primary categories as parameters
            return render_template(
                "/products/product.html",
                products_df=products_df.to_dict(orient="records"),
                AllPrimaryCategory=PrimaryCategory,
                FilteredPrimaryCategory=PrimaryCategory,
                user=session["username"],
                facebook=facebook,
                twitter=twitter,
                instagram=instagram,
                linkedin=linkedin,
                whatsapp=whatsapp,
                youtube=youtube,
                mobile=mobile,
            )
        else:
            return render_template(
                "/products/product.html",
                products_df=products_df.to_dict(orient="records"),
                AllPrimaryCategory=PrimaryCategory,
                FilteredPrimaryCategory=PrimaryCategory,
                facebook=facebook,
                twitter=twitter,
                instagram=instagram,
                linkedin=linkedin,
                whatsapp=whatsapp,
                youtube=youtube,
                mobile=mobile,
            )
    except FileNotFoundError:
        # Handle the case when the product details file is not found
        return render_template(
            "other/error.html", error_message="Product details file not found."
        )
    except Exception as e:
        # Handle any unexpected errors and render an error template
        print(f"Error: {e}")
        return render_template(
            "other/error.html", error_message="An unexpected error occurred."
        )


@app.route("/product/<product_id>")
def product_details(product_id):
    try:
        # Read the Excel file containing product details
        products_df = pd.read_excel(product_detailes_path)
        products_df.fillna(" ", inplace=True)
        products_df[["Name", "Primary_Category_Alt"]] = products_df[
            ["Name", "Primary_Category_Alt"]
        ].applymap(lambda x: x.title() if isinstance(x, str) else x)
        products_df[["Description", "Tagline"]] = products_df[
            ["Description", "Tagline"]
        ].applymap(capitalize_sentences)
        # Filter the dataframe for the specific product
        product = products_df[products_df["ID"] == str(product_id)].iloc[0].to_dict()
        # Render the 'products/details.html' template with the product data'
        PrimaryCategory = ["All"] + list(set(products_df["Primary_Category_Alt"]))
        return render_template(
            "products/details.html",
            product=product,
            products_df=products_df.to_dict(orient="records"),
            PrimaryCategory=PrimaryCategory,
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )
    except FileNotFoundError:
        # Handle the case when the product details file is not found
        return render_template(
            "other/error.html",
            error_message="Product details file not found.",
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )
    except Exception as e:
        # Handle any unexpected errors and render an error template
        print(f"Error: {e}")
        return render_template(
            "other/error.html",
            error_message="An unexpected error occurred.",
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )


@app.route("/products/<category>")
def product_category_details(category):
    try:
        products_df = pd.read_excel(product_detailes_path)
        products_df = products_df[products_df["Status"] == "Active"]

        # Check if DataFrame is empty after filtering
        if products_df.empty:
            return render_template(
                "other/error.html", error_message="No active products found."
            )

        # Fill missing values with a space
        products_df.fillna(" ", inplace=True)

        products_df[["Name", "Primary_Category_Alt"]] = products_df[
            ["Name", "Primary_Category_Alt"]
        ].applymap(lambda x: x.title() if isinstance(x, str) else x)

        products_df[["Description", "Tagline"]] = products_df[
            ["Description", "Tagline"]
        ].applymap(capitalize_sentences)

        # Ensure 'Category' column exists and contains data
        if (
            "Category" not in products_df.columns
            or products_df["Category"].isnull().all()
        ):
            return render_template(
                "other/error.html", error_message="Category data is missing."
            )

        # Define custom category order
        category_order = ["ALL", "Popular Items", "New Arrival"]

        # Create a new column 'CategoryOrder' for sorting
        products_df["CategoryOrder"] = products_df["Category"].apply(
            lambda x: (
                category_order.index(x) if x in category_order else len(category_order)
            )
        )

        # Sort DataFrame based on 'CategoryOrder' and then drop the helper column
        products_df = products_df.sort_values("CategoryOrder").drop(
            columns="CategoryOrder"
        )
        AllPrimaryCategory = ["All"] + list(set(products_df["Primary_Category_Alt"]))
        if category != "All":
            products_df = products_df[products_df["Primary_Category_Alt"] == category]
        FilteredPrimaryCategory = list(set(products_df["Primary_Category_Alt"]))
        # Render the 'products/details.html' template with the filtered products
        return render_template(
            "/products/product.html",
            products_df=products_df.to_dict(orient="records"),
            AllPrimaryCategory=AllPrimaryCategory,
            FilteredPrimaryCategory=FilteredPrimaryCategory,
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )

    except FileNotFoundError:
        # Handle the case when the product details file is not found
        return render_template(
            "other/error.html",
            error_message="Product details file not found.",
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )

    except Exception as e:
        # Handle any unexpected errors and render an error template
        print(f"Error: {e}")
        return render_template(
            "other/error.html",
            error_message="An unexpected error occurred.",
            facebook=facebook,
            twitter=twitter,
            instagram=instagram,
            linkedin=linkedin,
            whatsapp=whatsapp,
            youtube=youtube,
            mobile=mobile,
        )

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def calculate_inventory_metrics(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    future_discount: float = 10,
    future_weeks: int = 2
):
    # Ensure the sales_data DataFrame contains the necessary columns
    required_columns = [
        'week', 'ID', 'Demand', 'Discount_Percentage', 'lag_1', 
        'rolling_mean', 'upload_date_time', 'Primary_Category_Alt', 
        'Name', 'price_range_description', 'Price', 'Description', 
        'Status', 'Inventory_Level', 'Reorder_Point', 
        'Lead_Time_Days', 'Supplier', 'Supplier_Name', 
        'Last_Restock_Date', 'Total_Revenue'
    ]
    if not all(col in sales_data.columns for col in required_columns):
        raise ValueError("Input DataFrame must contain the required columns: " + ", ".join(required_columns))

    # Convert 'upload_date_time' to datetime for filtering
    sales_data['upload_date_time'] = pd.to_datetime(sales_data['upload_date_time'])

    # Extract year from the upload date
    sales_data['year'] = sales_data['upload_date_time'].dt.year

    # Calculate previous year demand for each product
    previous_year_demand = sales_data[sales_data['year'] == sales_data['year'].max() - 1].groupby(['ID'])['Demand'].sum().reset_index()
    previous_year_demand.rename(columns={'Demand': 'Previous_Year_Demand'}, inplace=True)

    # Prepare features and target
    features = ['week', 'ID', 'lag_1', 'rolling_mean', 'Discount_Percentage']
    X = sales_data[features]
    y = sales_data['Demand']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the XGBoost model
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Make predictions and calculate RMSE
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Cross-validation to evaluate model performance
    n_train_samples = X_train_scaled.shape[0]
    if n_train_samples > 1:
        n_splits = min(5, n_train_samples)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=n_splits, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
    else:
        cv_rmse = np.array([rmse])

    # Forecast for future weeks
    selected_id = sales_data['ID'].iloc[0]
    last_record = sales_data[sales_data['ID'] == selected_id].sort_values('upload_date_time').tail(1)

    if last_record.empty:
        raise ValueError("No records found for the selected ID.")

    last_demand = last_record['Demand'].values[0]
    last_rolling_mean = last_record['rolling_mean'].values[0]

    predicted_demands = []
    future_predictions = []
    current_lag = last_demand
    current_rolling = last_rolling_mean

    # Get today's date
    today = pd.Timestamp.now()

    # Prepare future predictions
    for week in range(1, future_weeks + 1):
        week_number = today.isocalendar()[1] + week
        future_week = pd.DataFrame({
            'week': [week_number],
            'ID': [selected_id],
            'lag_1': [current_lag],
            'rolling_mean': [current_rolling],
            'Discount_Percentage': [future_discount]
        })

        future_week_scaled = scaler.transform(future_week)
        predicted_demand = model.predict(future_week_scaled)[0]
        predicted_demands.append(predicted_demand)

        future_predictions.append({
            'week': week_number,
            'ID': selected_id,
            'Demand': predicted_demand,
            'Discount_Percentage': future_discount,
            'lag_1': current_lag,
            'rolling_mean': current_rolling,
            'upload_date_time': today + pd.Timedelta(weeks=week),
            'Primary_Category_Alt': last_record['Primary_Category_Alt'].values[0],
            'Name': last_record['Name'].values[0],
            'price_range_description': last_record['price_range_description'].values[0],
            'Price': last_record['Price'].values[0],
            'Description': last_record['Description'].values[0],
            'Status': last_record['Status'].values[0],
            'Inventory_Level': last_record['Inventory_Level'].values[0],
            'Reorder_Point': last_record['Reorder_Point'].values[0],
            'Lead_Time_Days': last_record['Lead_Time_Days'].values[0],
            'Supplier': last_record['Supplier'].values[0],
            'Supplier_Name': last_record['Supplier_Name'].values[0],
            'Last_Restock_Date': last_record['Last_Restock_Date'].values[0],
            'Total_Revenue': last_record['Total_Revenue'].values[0],
        })

        # Update lag and rolling mean for the next iteration
        current_lag = predicted_demand
        current_rolling = (current_rolling * 3 + predicted_demand) / 4

    # Convert future predictions to a DataFrame
    future_df = pd.DataFrame(future_predictions)

    # Combine original sales_data with future predictions
    sales_data = pd.concat([sales_data, future_df], ignore_index=True)

    # Merge with previous year demand data
    sales_data = sales_data.merge(previous_year_demand, on='ID', how='left')

    # Output metrics
    print(f"RMSE: {rmse:.2f}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f}")
    print(f"Predicted Demands: {predicted_demands}")
    return sales_data, predicted_demands

@app.route('/sales_data')
def sales_data():
    try:
        # Load the sales data
        df = pd.read_csv(os.path.join("Detailes", "sales_4yrs.csv"))

        # Convert 'upload_date_time' to datetime
        df['upload_date_time'] = pd.to_datetime(df['upload_date_time'])
        today = datetime.now()
        
        # Define the date range (e.g., last 30 days)
        start_date = today - pd.Timedelta(days=30)  # Change the number of days as needed
        end_date = today

        # Filter the DataFrame for the specified date range
        filtered_df = df[(df['upload_date_time'] >= start_date) & 
                         (df['upload_date_time'] <= end_date) & 
                         (df['Primary_Category_Alt'].isin(['Accessories', 'Shoes', 'Women', 'Kids', 'Men']))]

        if filtered_df.empty:
            return render_template('products/sales_data.html', 
                                   sales_records=[],
                                   error="No data available for the current week")
        # Calculate inventory metrics
        processed_df, predicted_demands = calculate_inventory_metrics(sales_data=filtered_df)
        
        # Get current year and previous year
        current_year = today.year
        previous_year = current_year - 1
        
        # Prepare previous and current year demands
        previous_year_demand = filtered_df[filtered_df['upload_date_time'].dt.year == previous_year].groupby('Primary_Category_Alt')['Demand'].sum().reset_index()
        current_year_demand = filtered_df[filtered_df['upload_date_time'].dt.year == current_year].groupby('Primary_Category_Alt')['Demand'].sum().reset_index()

        # Merging the two dataframes
        previous_year_demand.rename(columns={'Demand': 'Previous_Year_Demand'}, inplace=True)
        current_year_demand.rename(columns={'Demand': 'Current_Year_Demand'}, inplace=True)
        demand_df = pd.merge(previous_year_demand, current_year_demand, on='Primary_Category_Alt', how='outer')

        # Logic for placing orders
        order_recommendations = []
        for index, row in processed_df.iterrows():
            category = row.get('Primary_Category_Alt', 'Unknown')
            previous_demand_row = demand_df[demand_df['Primary_Category_Alt'] == category]
            previous_year_demand_value = previous_demand_row['Previous_Year_Demand'].values[0] if not previous_demand_row.empty else 0
            current_year_demand_value = previous_demand_row['Current_Year_Demand'].values[0] if not previous_demand_row.empty else 0
            
            recommendation = {
                'Category': category,
                'Price_Range': row.get('price_range_description', 'Unknown'),
                'Product': row.get('Name', 'Unknown'),
                'Previous_Year_Demand': previous_year_demand_value,
                'Current_Year_Demand': current_year_demand_value,
                'Demand': row.get('Demand', 0),
                'Predicted_Demand': predicted_demands,  # Adjust this to reflect the predicted value correctly
                'Inventory_Level': row.get('Inventory_Level', 0),
                'Reorder_Point': row.get('Reorder_Point', 0),
                'Lead_Time': row.get('Lead_Time_Days', 0),  # Use .get() to avoid KeyError
                'Supplier_Name':row.get('Supplier_Name', 'Unknown'),
            }
            
            # Define order recommendations based on inventory status
            if row['Demand'] > row['Reorder_Point'] and row['Inventory_Level'] < row['Reorder_Point']:
                recommendation['Action'] = "Action Required: Place an order immediately due to high demand and insufficient inventory."
            elif row['Demand'] > row['Reorder_Point'] and row['Inventory_Level'] >= row['Reorder_Point']:
                recommendation['Action'] = "Action Recommended: Monitor closely due to ongoing promotions."
            else:
                recommendation['Action'] = "No action required."

            order_recommendations.append(recommendation)

        return render_template('products/sales_data.html', 
                               sales_records=order_recommendations,
                               categories = ['Men', 'Women', 'Kids', 'Accessories', 'Shoes'],
                               error=None,
                               user=session["username"],
                                facebook=facebook,
                                twitter=twitter,
                                instagram=instagram,
                                linkedin=linkedin,
                                whatsapp=whatsapp,
                                youtube=youtube,
                                mobile=mobile,
                                )

    except Exception as e:
        print(f"Error: {e}")
        return render_template('products/sales_data.html', 
                               sales_records=[],
                               error="An error occurred while processing sales data.")

# Define the file path for the Excel file
FILE_PATH = os.path.join("Detailes", 'orders.xlsx')

def save_order_to_excel(order_data):
    # Create a DataFrame from the order data
    df = pd.DataFrame([order_data])

    # Append to existing file or create new with headers
    if os.path.exists(FILE_PATH):
        with pd.ExcelWriter(FILE_PATH, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            # Identify the last row in the existing sheet
            sheet = writer.sheets.get('Sheet1', writer.book.create_sheet('Sheet1'))
            startrow = sheet.max_row if sheet.max_row is not None else 0
            df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df.to_excel(FILE_PATH, index=False)

@app.route('/submit_order', methods=['POST'])
def submit_order():
    # Receive the product and quantity data from the request
    data = request.get_json()
    product_name = data.get('product_name')
    quantity = data.get('quantity')
    supplier = data.get('supplier')
    lead_time = data.get('lead_time')  # Optional field
    # Validate required fields
    if not product_name or not quantity:
        return jsonify({'error': 'Product name and quantity are required.'}), 400
    
    # Ensure quantity is a positive integer
    try:
        quantity = int(quantity)
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be a positive integer.'}), 400
    except ValueError:
        return jsonify({'error': 'Quantity must be a valid integer.'}), 400

    # Save the order data to Excel
    try:
        save_order_to_excel({
            'product_name': product_name,
            'quantity': quantity,
            'supplier': supplier,
            'lead_time': lead_time
        })
    except Exception as e:
        return jsonify({'error': f'Failed to save order: {str(e)}'}), 500

    # Respond with success
    return jsonify({'success': 'Order has been successfully placed.'}), 200

@app.route('/suppliers_orders')
def suppliers_orders():
    suppliers_orders = pd.read_excel(os.path.join("Detailes", "orders.xlsx"))
    suppliers_orders = suppliers_orders.to_dict(orient='records')
    return render_template('supplier_orders.html', suppliers_orders=suppliers_orders,                user=session["username"],
                facebook=facebook,
                twitter=twitter,
                instagram=instagram,
                linkedin=linkedin,
                whatsapp=whatsapp,
                youtube=youtube,
                mobile=mobile,
                )

@app.route('/sale')
def sale():
    # Load the CSV file
    file_path = os.path.join("Detailes", "sales_4yrs.csv")
    sales_data = pd.read_csv(file_path)
    
    # Filter for relevant columns
    filtered_columns = [
        'week', 'ID', 'Name', 'Demand', 'Discount_Percentage', 'Price', 
        'Total_Revenue', 'Inventory_Level', 'Reorder_Point', 'Lead_Time_Days', 
        'Supplier_Name', 'Last_Restock_Date'
    ]
    # Filter and convert to dictionary
    sales_data = sales_data[filtered_columns].to_dict(orient='records')
    
    # Render the HTML template with sales data
    return render_template('sale.html', sales_data=sales_data,                user=session["username"],
                facebook=facebook,
                twitter=twitter,
                instagram=instagram,
                linkedin=linkedin,
                whatsapp=whatsapp,
                youtube=youtube,
                mobile=mobile,
                )

@app.route('/products')
def get_products():
    try:
        df = pd.read_csv(os.path.join("Detailes", "sales_4yrs.csv"))
        products = df['Name'].unique().tolist()  # Get unique product names
        return jsonify(products)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/record_sale', methods=['POST'])
def record_sale():
    data = request.json
    product_name = data.get('productName')
    quantity_sold = int(data.get('quantity'))
    sale_price = float(data.get('salePrice'))
    
    # Load the full inventory data
    df = pd.read_csv(os.path.join("Detailes", "sales_4yrs.csv"))
    
    # Find the product in the DataFrame
    product_index = df[df['Name'] == product_name].index
    
    if not product_index.empty:
        index = product_index[0]
        
        # Check current inventory level
        current_inventory = df.at[index, 'Inventory_Level']
        if current_inventory >= quantity_sold:
            # Update inventory level and total revenue
            df.at[index, 'Inventory_Level'] -= quantity_sold
            df.at[index, 'Total_Revenue'] += sale_price * quantity_sold
            df.at[index, 'upload_date_time'] = datetime.now()  # Update sale date
            
            # Save updated DataFrame back to CSV
            df.to_csv(os.path.join("Detailes", "sales_4yrs.csv"), index=False)
            
            return jsonify({'message': 'Sale recorded and inventory updated successfully.'})
        else:
            return jsonify({'message': 'Not enough inventory available.'}), 400
    else:
        return jsonify({'message': 'Product not found.'}), 404

