import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df):
        # Toplam fiyat
        df["total"] = df["unit_price"] * df["quantity"]

        # Ürün + Kargo kombinasyonu
        df["product_shipper_cross"] = df["product_name"] + "_" + df["shipper_company_name"]

        # One-hot encoding
        df_encoded = pd.get_dummies(df[["product_shipper_cross"]], prefix="cross")

        # Orijinal veriyle birleştir
        df_final = pd.concat([df, df_encoded], axis=1)

        return df_final

    def prepare_model_data(self, df):
        # Özellik sütunları (sayısal olanlar + encoded olanlar)
        feature_columns = ["unit_price", "quantity", "total"] + \
            [col for col in df.columns if col.startswith("cross_")]

        X = df[feature_columns]
        y = np.zeros(len(df))  # Dummy hedef değişken (etiket), çünkü yok

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
