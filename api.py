from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from predict import predict_race

app = FastAPI(title="F1 Race Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DriverInput(BaseModel):
    driver: str
    constructor: str
    quali_position: int
    grid: int
    avg_finish_last3: Optional[float] = 10
    avg_points_last3: Optional[float] = 5
    constructor_avg_last3: Optional[float] = 10
    championship_points: Optional[float] = 0
    championship_position: Optional[int] = 10

class RaceInput(BaseModel):
    circuit: str
    drivers: List[DriverInput]

@app.get("/")
def root():
    return {"message": "F1 Race Predictor API is running!"}

@app.post("/predict")
def predict(race: RaceInput):
    try:
        drivers_data = [d.model_dump() for d in race.drivers]
        results = predict_race(race.circuit, drivers_data)
        return {
            "circuit": race.circuit,
            "predictions": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/circuits")
def get_circuits():
    import joblib
    le = joblib.load("models/le_circuit.pkl")
    return {"circuits": sorted(le.classes_.tolist())}

@app.get("/drivers")
def get_drivers():
    import joblib
    le = joblib.load("models/le_driver.pkl")
    return {"drivers": sorted(le.classes_.tolist())}

@app.get("/constructors")
def get_constructors():
    import joblib
    le = joblib.load("models/le_constructor.pkl")
    return {"constructors": sorted(le.classes_.tolist())}