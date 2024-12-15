from flask import Flask, render_template
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Create a Flask server
server = Flask(__name__)

# Load your CSV data (replace with the path to your CSV file)
data_path = r'F:\Assignements\Binny_inventorymanagement\Retail_store_manager\ecommers\Detailes\sales_4yrs.xlsx'  # Update this path to your data file
data = pd.read_excel(data_path)

# Ensure the date column is in datetime format
data['upload_date_time'] = pd.to_datetime(data['upload_date_time'])



# Create a Dash app
app = Dash(__name__, server=server, url_base_pathname='/dash/')

# Layout for the Dash app
app.layout = html.Div([
    html.H1('Sales Data Dashboard'),

    # Date Range Picker
    dcc.DatePickerRange(
        id='date-picker',
        start_date=data['upload_date_time'].min().date(),
        end_date=data['upload_date_time'].max().date(),
        display_format='YYYY-MM-DD'
    ),

    # Dropdown for Product Categories (example filter)
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': cat, 'value': cat} for cat in data['Primary_Category_Alt'].unique()],
        multi=True,
        placeholder="Select Product Categories"
    ),

    # Graph Placeholder
    dcc.Graph(id='sales-graph')
])

# Callback to update the graph based on filters
@app.callback(
    Output('sales-graph', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    Input('product-dropdown', 'value')
)
def update_graph(start_date, end_date, selected_categories):
    # Filter data based on selected date range
    filtered_data = data[
        (data['upload_date_time'] >= start_date) & 
        (data['upload_date_time'] <= end_date)
    ]

    # Filter data based on selected product categories
    if selected_categories:
        filtered_data = filtered_data[filtered_data['Primary_Category_Alt'].isin(selected_categories)]

    # Create the plot
    fig = px.line(
        filtered_data,
        x='upload_date_time',
        y='Total_Revenue',
        title='Total Revenue Over the Selected Period',
        labels={'Total_Revenue': 'Total Revenue', 'upload_date_time': 'Date'}
    )
    
    return fig

# Route to render the main Flask page
@server.route('/')
def index():
    return render_template('layout.html')

# Run the Flask server
if __name__ == '__main__':
    server.run(debug=True)