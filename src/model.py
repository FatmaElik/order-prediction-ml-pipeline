# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.config import MODEL_CONFIG, XGBOOST_CONFIG


class ReturnRiskModel:
    def __init__(self, model_type="tensorflow"):
        self.model_type = model_type
        self.model = None
        self.history = None

    def build_model(self, input_dim):
        if self.model_type == "tensorflow":
            model = Sequential([
                Dense(64, activation="relu", input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
            self.model = model

        elif self.model_type == "xgboost":
            self.model = XGBClassifier(**XGBOOST_CONFIG)

        else:
            raise ValueError("Unsupported model type")

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=MODEL_CONFIG["test_size"], random_state=MODEL_CONFIG["random_state"])

    def train(self, X_train, X_test, y_train, y_test):
        if self.model_type == "tensorflow":
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
            ]
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=MODEL_CONFIG["epochs"],
                callbacks=callbacks,
                verbose=1
            )
        elif self.model_type == "xgboost":
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.model_type == "tensorflow":
            return self.model.evaluate(X_test, y_test)
        elif self.model_type == "xgboost":
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return {"accuracy": acc}

    def predict(self, X):
        if self.model_type == "tensorflow":
            return self.model.predict(X)
        elif self.model_type == "xgboost":
            return self.model.predict_proba(X)[:, 1]
