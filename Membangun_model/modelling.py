import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path  = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df =  pd.read_csv(test_path)

X_train = train_df.drop(columns=["charges"])
y_train = train_df["charges"]
X_test = test_df.drop(columns=["charges"])
y_test = test_df["charges"]

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Insurance_Cost_Prediction")

with mlflow.start_run(run_name="manual_run"):
    mlflow.sklearn.autolog()

    model = RandomForestRegressor()
    model.fit(X_train,y_train)