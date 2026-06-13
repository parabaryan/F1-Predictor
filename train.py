import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_merge():
    race = pd.read_csv("data/race_results.csv")
    quali = pd.read_csv("data/qualifying.csv")
    standings = pd.read_csv("data/standings.csv")

    df = race.merge(quali, on=["year", "round", "circuit", "driver", "constructor"], how="left")
    df = df.merge(standings, on=["year", "driver"], how="left")
    df["quali_position"] = df["quali_position"].fillna(20)
    return df

def engineer_features(df):
    df = df.sort_values(["driver", "year", "round"])

    # avg finish position over last 3 races per driver
    df["avg_finish_last3"] = (
        df.groupby("driver")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # avg points over last 3 races
    df["avg_points_last3"] = (
        df.groupby("driver")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # constructor avg finish last 3 races
    df["constructor_avg_last3"] = (
        df.groupby("constructor")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df["is_winner"] = (df["position"] == 1).astype(int)
    df["is_podium"] = (df["position"] <= 3).astype(int)

    df = df.fillna(10)
    return df

def encode(df):
    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()
    le_circuit = LabelEncoder()

    df["driver_enc"] = le_driver.fit_transform(df["driver"])
    df["constructor_enc"] = le_constructor.fit_transform(df["constructor"])
    df["circuit_enc"] = le_circuit.fit_transform(df["circuit"])

    os.makedirs("models", exist_ok=True)
    joblib.dump(le_driver, "models/le_driver.pkl")
    joblib.dump(le_constructor, "models/le_constructor.pkl")
    joblib.dump(le_circuit, "models/le_circuit.pkl")

    return df

def train():
    print("Loading data...")
    df = load_and_merge()
    df = engineer_features(df)
    df = encode(df)

    features = [
        "quali_position", "grid", "driver_enc", "constructor_enc",
        "circuit_enc", "avg_finish_last3", "avg_points_last3",
        "constructor_avg_last3", "championship_points", "championship_position"
    ]

    X = df[features]

    # --- Model 1: Race Winner ---
    y_winner = df["is_winner"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_winner, test_size=0.2, random_state=42)
    winner_model = RandomForestClassifier(n_estimators=100, random_state=42)
    winner_model.fit(X_train, y_train)
    print(f"Winner model accuracy: {winner_model.score(X_test, y_test):.2f}")
    joblib.dump(winner_model, "models/winner_model.pkl")

    # --- Model 2: Podium ---
    y_podium = df["is_podium"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_podium, test_size=0.2, random_state=42)
    podium_model = RandomForestClassifier(n_estimators=100, random_state=42)
    podium_model.fit(X_train, y_train)
    print(f"Podium model accuracy: {podium_model.score(X_test, y_test):.2f}")
    joblib.dump(podium_model, "models/podium_model.pkl")

    # --- Model 3: Championship Points ---
    y_points = df["championship_points"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_points, test_size=0.2, random_state=42)
    points_model = RandomForestRegressor(n_estimators=100, random_state=42)
    points_model.fit(X_train, y_train)
    print(f"Points model R² score: {points_model.score(X_test, y_test):.2f}")
    joblib.dump(points_model, "models/points_model.pkl")

    print("\nAll models saved to /models")

if __name__ == "__main__":
    train()