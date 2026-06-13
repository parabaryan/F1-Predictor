import joblib
import pandas as pd
import numpy as np

def load_models():
    winner_model = joblib.load("models/winner_model.pkl")
    podium_model = joblib.load("models/podium_model.pkl")
    points_model = joblib.load("models/points_model.pkl")
    le_driver = joblib.load("models/le_driver.pkl")
    le_constructor = joblib.load("models/le_constructor.pkl")
    le_circuit = joblib.load("models/le_circuit.pkl")
    return winner_model, podium_model, points_model, le_driver, le_constructor, le_circuit

def predict_race(circuit, drivers_data):
    """
    drivers_data: list of dicts with keys:
        driver, constructor, quali_position, grid,
        avg_finish_last3, avg_points_last3,
        constructor_avg_last3, championship_points, championship_position
    """
    winner_model, podium_model, points_model, le_driver, le_constructor, le_circuit = load_models()

    known_drivers = set(le_driver.classes_)
    known_constructors = set(le_constructor.classes_)
    known_circuits = set(le_circuit.classes_)

    if circuit not in known_circuits:
        raise ValueError(f"Unknown circuit: '{circuit}'. Check le_circuit.pkl for valid names.")

    rows = []
    for d in drivers_data:
        if d["driver"] not in known_drivers:
            raise ValueError(f"Unknown driver: '{d['driver']}'")
        if d["constructor"] not in known_constructors:
            raise ValueError(f"Unknown constructor: '{d['constructor']}'")

        rows.append({
            "quali_position": d["quali_position"],
            "grid": d["grid"],
            "driver_enc": le_driver.transform([d["driver"]])[0],
            "constructor_enc": le_constructor.transform([d["constructor"]])[0],
            "circuit_enc": le_circuit.transform([circuit])[0],
            "avg_finish_last3": d.get("avg_finish_last3", 10),
            "avg_points_last3": d.get("avg_points_last3", 5),
            "constructor_avg_last3": d.get("constructor_avg_last3", 10),
            "championship_points": d.get("championship_points", 0),
            "championship_position": d.get("championship_position", 10),
        })

    X = pd.DataFrame(rows)

    win_probs = winner_model.predict_proba(X)[:, 1]
    podium_probs = podium_model.predict_proba(X)[:, 1]
    pred_points = points_model.predict(X)

    results = []
    for i, d in enumerate(drivers_data):
        results.append({
            "driver": d["driver"],
            "constructor": d["constructor"],
            "win_probability": round(float(win_probs[i]) * 100, 1),
            "podium_probability": round(float(podium_probs[i]) * 100, 1),
            "predicted_championship_points": round(float(pred_points[i]), 1),
        })

    results.sort(key=lambda x: x["win_probability"], reverse=True)
    return results

if __name__ == "__main__":
    # example prediction - Monaco 2024 grid
    sample_drivers = [
        {"driver": "leclerc", "constructor": "ferrari", "quali_position": 1, "grid": 1, "avg_finish_last3": 3, "avg_points_last3": 15, "constructor_avg_last3": 3, "championship_points": 45, "championship_position": 3},
        {"driver": "max_verstappen", "constructor": "red_bull", "quali_position": 4, "grid": 4, "avg_finish_last3": 2, "avg_points_last3": 20, "constructor_avg_last3": 2, "championship_points": 161, "championship_position": 1},
        {"driver": "sainz", "constructor": "ferrari", "quali_position": 2, "grid": 2, "avg_finish_last3": 4, "avg_points_last3": 12, "constructor_avg_last3": 3, "championship_points": 40, "championship_position": 4},
        {"driver": "norris", "constructor": "mclaren", "quali_position": 3, "grid": 3, "avg_finish_last3": 5, "avg_points_last3": 10, "constructor_avg_last3": 5, "championship_points": 83, "championship_position": 2},
    ]

    results = predict_race("monaco", sample_drivers)
    print("\n🏎️  F1 Race Prediction — Monaco\n")
    for i, r in enumerate(results):
        # new
        print(f"{i+1}. {r['driver'].replace('_', ' ').title()} ({r['constructor'].replace('_', ' ').title()})")
        print(f"   Win probability:    {r['win_probability']}%")
        print(f"   Podium probability: {r['podium_probability']}%")
        print(f"   Predicted points:   {r['predicted_championship_points']}\n")