'''FastAPI definition'''

from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from threading import Lock

from pipeline import load_pipeline, predict_duration
from train import main as train_model



''' Define which values are allowed for predictions'''
class RideInput(BaseModel):
    PULocationID: int = Field(gt=0)                 # gt: > 0
    DOLocationID: int = Field(gt=0)                 # gt: > 0
    trip_distance: float = Field(ge=0.1, le=100)     # ge: >= 0.1 ... le: <= 100

'''Input pattern for parquet files'''
class TrainingInput(BaseModel):
    load_parquet: str = Field(
        pattern=r"^green_tripdata_[0-9]{4}-(0[1-9]|1[0-2])\.parquet$",
        examples=["green_tripdata_2024-01.parquet"],
    )


''' Ist die Webanwendung, an der anschließend /predict, etc. anmeldet'''
#TERMINAL- COMMAND: uv run uvicorn api:app --app-dir 00_project_implementation --host 127.0.0.1 --port 8000 --reload
app = FastAPI()

''' Tuned model saved in mlflow'''
model_uri = "models:/green-taxi-random-forest@production"

''' Load tuned model once as API starts'''
model = load_pipeline(model_uri)



#@app.get("/Check System Health")



'''
    @app.post("/predict") verbindet die Funktion mit dieser Route.
    ride: RideInput sorgt dafür, dass FastAPI die eingehenden Daten anhand deiner Klasse prüft.
    Ungültige Eingaben werden abgewiesen.
'''
@app.post("/predict")           #Aufruf bei Post: '/Predict'- Anfrage, übermitteln an API mittels Decorator
def predict(ride:RideInput):
    ride_data = pd.DataFrame([ride.model_dump()])     # .model_dump() wandelt das 'RideInput'-Objekt in ein Dictionary um

    predictions = predict_duration(model, ride_data)
    predicted_duration_minutes = {"predicted_duration_minutes": float(predictions[0])}
    return predicted_duration_minutes


# Sorgt dafür, dass kein 2. Lauf gestartet werden kann während der erste noch läuft
training_lock = Lock()

@app.post("/train")
def train_endpoint(request: TrainingInput):
    project_dir = Path(__file__).resolve().parent.parent
    parquet_path = project_dir / "data" / request.load_parquet

    if not parquet_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Parquet file not found: {request.load_parquet}",
        )

    if not training_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Training already running.",
        )

    try:
        train_model(request.load_parquet)

        return {
            "status": "completed",
            "dataset": request.load_parquet,
            "message": "Training and MLflow processing completed.",
        }
    finally:
        training_lock.release()