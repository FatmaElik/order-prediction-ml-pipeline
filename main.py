from src.database import DatabaseManager
from src.feature_engineering import FeatureEngineer
from src.model import ReturnRiskModel


def main(model_type="tensorflow"):
    print("📦 Connecting to database...")
    db_manager = DatabaseManager()

    print("📊 Fetching order data...")
    df = db_manager.get_order_data()

    print("🛠️  Creating features...")
    feature_engineer = FeatureEngineer()
    df_processed = feature_engineer.create_features(df)

    X, y = feature_engineer.prepare_model_data(df_processed)

    print(f"🧠 Initializing '{model_type}' model...")
    model = ReturnRiskModel(model_type=model_type)
    X_train, X_test, y_train, y_test = model.split_data(X, y)

    print("🏗️  Building model...")
    model.build_model(input_dim=X_train.shape[1])

    print("🚀 Training started...")
    model.train(X_train, X_test, y_train, y_test)

    print("✅ Evaluating model...")
    results = model.evaluate(X_test, y_test)

    print(f"\n🎯 Accuracy Result ({model_type.upper()}): {results}")


if __name__ == "__main__":
    # TensorFlow kullanmak için:
    #main(model_type="tensorflow")

    # XGBoost kullanmak için:
    main(model_type="xgboost")
