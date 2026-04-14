import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn

from preprocessing import preprocess

df = pd.read_csv("data/creditcard.csv")
df = preprocess(df)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=500,
    random_state=42
)


mlflow.set_experiment("fraud-detection")

with mlflow.start_run():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]

    # Metric
    roc_auc = roc_auc_score(y_test, y_pred)


    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": 500
    })

    mlflow.sklearn.log_model(model, "model")

    print(f"ROC-AUC: {roc_auc:.4f}")


joblib.dump(model, "models/model.pkl")

print("Model saved successfully ✅")