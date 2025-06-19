# src/train.py

from model import ReturnRiskModel
from src.database import DatabaseManager
from src.feature_engineering import FeatureEngineer


def train_pipeline(model_type="tensorflow"):
    # 1. Veriyi çek
    db = DatabaseManager()
    df = db.get_order_data()

    # 2. Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    X, y = fe.prepare_model_data(df)

    # 3. Model inşası ve eğitimi
    model = ReturnRiskModel(model_type=model_type)
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    model.build_model(input_dim=X_train.shape[1])
    model.train(X_train, X_test, y_train, y_test)

    # 4. Modeli değerlendir
    results = model.evaluate(X_test, y_test)
    print(f"\n✅ {model_type.upper()} Evaluation Result:", results)

    return model
