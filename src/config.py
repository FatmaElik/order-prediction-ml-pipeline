
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Veri tabanı ayarları
DB_CONFIG = {
    "dbname":os.getenv("DB_NAME","GYK1Northwind"),
    "user":os.getenv("DB_USER","postgres"),
    "password":os.getenv("DB_PASSWORD","fatma"),
    "host":os.getenv("DB_HOST","localhost"),
    "port":os.getenv("DB_PORT","5432")  
}

# Ortak model eğitim ayarları
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "epochs": int(os.getenv("EPOCHS", 50))  # çevre değişkeni üzerinden ayarlanabilir
}

# Özellik mühendisliği ayarları
"""FEATURE_CONFIG = {
    "high_discount_threshold": 0.75,
    "low_amount_threshold": 0.25
}"""

# Linear Regression ayarları
LINEAR_REGRESSION_CONFIG = {
    "fit_intercept": True,
    "copy_X": True,
    "n_jobs": -1
}

# XGBoost ayarları
XGBOOST_CONFIG = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "random_state": MODEL_CONFIG["random_state"],
    "n_jobs": -1
}


