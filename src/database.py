import psycopg2
import pandas as pd
from src.config import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        try:
          self.conn = psycopg2.connect(**DB_CONFIG)
          print("Database connected")
        except Exception as e:
          print("Error occured while connection to the db")
          raise
         
    def disconnect(self):
       if self.conn:
          self.conn.close()
          print("Disconnected from the db")

    def get_order_data(self):
        query = """
            SELECT
            p.product_name,
            s.company_name AS shipper_company_name,
            od.unit_price,
            od.quantity,
            (od.unit_price * od.quantity) AS total,
            CONCAT(p.product_name, '_', s.company_name) AS product_shipper_cross
            FROM
            order_details od
            INNER JOIN
            products p ON od.product_id = p.product_id
            INNER JOIN
            orders o ON od.order_id = o.order_id
            INNER JOIN
            shippers s ON o.ship_via = s.shipper_id;
        """
        try:
          df = pd.read_sql_query(query, self.conn)
          return df
        except Exception as e:
          print(f"Error fetching data: {e}")








