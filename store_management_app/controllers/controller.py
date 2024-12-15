import pandas as pd
import datetime


class DataManager:
    def __init__(self, product_details_path, social_sites_path, product_category_path):
        self.product_details_path = product_details_path
        self.social_sites_path = social_sites_path
        self.product_category_path = product_category_path

    def get_social_sites(self):
        try:
            social_site_df = pd.read_excel(self.social_sites_path)
            social_site_df.sort_values(
                by="upload_date_time", inplace=True, ascending=False
            )
            social_site_df.fillna(" ", inplace=True)
            social_site_df["site"] = (
                social_site_df["site"]
                .apply(lambda x: x.capitalize() if isinstance(x, str) else x)
                .apply(self.capitalize_sentences)
            )
            return [i._asdict() for i in social_site_df.itertuples()]
        except FileNotFoundError:
            raise FileNotFoundError("Social sites file not found.")

    def get_product_details(self):
        try:
            products_df = pd.read_excel(self.product_details_path)
            products_df.sort_values(
                by="upload_date_time", inplace=True, ascending=False
            )
            products_df.fillna(" ", inplace=True)
            products_df[["Name", "Primary_Category_Alt"]] = products_df[
                ["Name", "Primary_Category_Alt"]
            ].applymap(lambda x: x.title() if isinstance(x, str) else x)
            products_df[["Description", "Tagline"]] = products_df[
                ["Description", "Tagline"]
            ].applymap(self.capitalize_sentences)
            return [i._asdict() for i in products_df.itertuples()]
        except FileNotFoundError:
            raise FileNotFoundError("Product details file not found.")

    def get_product_categories(self):
        try:
            product_category_df = pd.read_excel(self.product_category_path)
            product_category_df.sort_values(
                by="product_category", ascending=True, inplace=True
            )
            return product_category_df["product_category"].tolist()
        except FileNotFoundError:
            raise FileNotFoundError("Product category file not found.")

    @staticmethod
    def capitalize_sentences(text):
        if isinstance(text, str):
            return ". ".join([sentence.capitalize() for sentence in text.split(". ")])
        return text
