from store_management_app import *
from store_management_app.pages import navigations
import pandas as pd

primary_products_cat = [
    "New Arrivals",
    "Popular Items",
    "Silk Sarees",
    "Seasonal",
    "Clearance",
]

# products_table
# products_table = pd.read_excel(navigations.product_detailes_path)
# products_table['Primary_Category_Alt'].unique()
# #product catigories
# product_categories = pd.read_excel(navigations.product_category_path)
# product_categories['product_category']


# [i for i in products_table['Primary_Category_Alt'].unique() if i in product_categories['product_category']]
