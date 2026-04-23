import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# =====================================================
# PREPARE DATASET FOR REPLY PREDICTION
# =====================================================

def prepare_reply_dataset(df):

    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date")

    # Calculate reply time in minutes
    df["reply_time"] = df["date"].diff().dt.total_seconds() / 60
    df = df.dropna(subset=["reply_time"])

    # Only when user changes (real reply)
    df = df[df["user"] != df["user"].shift(1)]

    # Remove unrealistic reply times
    df = df[(df["reply_time"] > 0) & (df["reply_time"] < 360)]

    # Feature engineering
    df["word_count"] = df["message"].apply(lambda x: len(str(x).split()))
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day_name()

    return df


# =====================================================
# TRAIN MODEL
# =====================================================

def train_reply_model(df):

    try:
        df = prepare_reply_dataset(df)

        # Basic validations
        if len(df) < 20:
            return {"Error": "Not enough reply data to train model."}

        if df["user"].nunique() < 2:
            return {"Error": "Need at least 2 users for reply prediction."}

        # Encode categorical features
        le_user = LabelEncoder()
        le_day = LabelEncoder()

        df["user_encoded"] = le_user.fit_transform(df["user"])
        df["day_encoded"] = le_day.fit_transform(df["day"])

        X = df[["word_count", "hour", "user_encoded", "day_encoded"]]
        y = df["reply_time"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Evaluation Metrics
        metrics = {
            "Linear Regression": {
                "MAE": mean_absolute_error(y_test, y_pred_lr),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                "R2": r2_score(y_test, y_pred_lr)
            },
            "Random Forest": {
                "MAE": mean_absolute_error(y_test, y_pred_rf),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                "R2": r2_score(y_test, y_pred_rf)
            }
        }

        # Save best model (Random Forest)
        joblib.dump(rf, "reply_model.pkl")
        joblib.dump(le_user, "user_encoder.pkl")
        joblib.dump(le_day, "day_encoder.pkl")

        return metrics

    except Exception as e:
        return {"Error": str(e)}


# =====================================================
# PREDICT REPLY TIME
# =====================================================

def predict_reply_time(word_count, hour, user, day):

    # Check if model exists
    if not os.path.exists("reply_model.pkl"):
        return {"Error": "Model not trained"}

    try:
        model = joblib.load("reply_model.pkl")
        le_user = joblib.load("user_encoder.pkl")
        le_day = joblib.load("day_encoder.pkl")

        user_encoded = le_user.transform([user])[0]
        day_encoded = le_day.transform([day])[0]

        X = pd.DataFrame(
            [[word_count, hour, user_encoded, day_encoded]],
            columns=["word_count", "hour", "user_encoded", "day_encoded"]
        )

        prediction = model.predict(X)[0]

        return {"Prediction": round(prediction, 2)}

    except Exception as e:
        return {"Error": str(e)}