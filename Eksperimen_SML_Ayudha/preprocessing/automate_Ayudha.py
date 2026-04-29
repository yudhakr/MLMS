import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_data(
    data: pd.DataFrame,
    target_column: str = "charges",
    save_pipeline_path: str = None,
    save_columns_path: str = None
):
    """
    Melakukan preprocessing:
    - split fitur & target
    - encoding categorical
    - scaling numerical
    - split train-test
    - (optional) simpan pipeline & metadata
    """

    # ===== SPLIT X & y =====
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # ===== DEFINE COLUMN =====
    cat_cols = ["sex", "smoker", "region"]
    num_cols = ["age", "bmi", "children"]

    # ===== ENCODING =====
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[cat_cols])

    # ===== SCALING =====
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num_cols])

    # ===== CONCAT =====
    X_final = np.concatenate([X_num, X_cat], axis=1)

    # ===== SPLIT DATA =====
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    # ===== SAVE PIPELINE =====
    if save_pipeline_path:
        os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
        joblib.dump(
            {
                "scaler": scaler,
                "encoder": encoder,
                "num_cols": num_cols,
                "cat_cols": cat_cols
            },
            save_pipeline_path
        )

    # ===== SAVE COLUMN HEADER =====
    if save_columns_path:
        os.makedirs(os.path.dirname(save_columns_path), exist_ok=True)

        encoded_cols = encoder.get_feature_names_out(cat_cols)
        all_cols = list(num_cols) + list(encoded_cols)

        pd.DataFrame(all_cols, columns=["feature_names"]) \
            .to_csv(save_columns_path, index=False)

    return X_train, X_test, y_train, y_test