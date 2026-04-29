import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

# ===== LOAD DATA =====
train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path  = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["charges"])
y_train = train_df["charges"]
X_test = test_df.drop(columns=["charges"])
y_test = test_df["charges"]

# ===== SET MLFLOW =====
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Insurance_Cost_Prediction_Tuning")

# ===== DAGSHUB =====
dagshub.init(repo_owner='yudhakr', repo_name='MLMS', mlflow=True)

# ===== HYPERPARAMETER =====
n_estimators_range = np.linspace(50, 500, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_score = -np.inf
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"rf_{n_estimators}_{max_depth}"):

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)  # ✅ aman semua versi
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # ===== LOGGING =====
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth
            })

            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MAPE", mape)

            mlflow.sklearn.log_model(model, "model")

            # ===== TRACK BEST =====
            if r2 > best_score:
                best_score = r2
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

print("Best R2:", best_score)
print("Best Params:", best_params)