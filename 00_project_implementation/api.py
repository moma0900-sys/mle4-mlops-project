'''FastAPI definition'''

from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from threading import Lock

from time import perf_counter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from pipeline import load_pipeline, predict_duration
from train import main as train_model
from prediction_logging import log_prediction



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

PREDICTION_REQUESTS = Counter(
    "taxi_prediction_requests_total",
    "Number of prediction requests by HTTP status.",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "taxi_prediction_duration_seconds",
    "Server processing time for prediction requests.",
)


@app.middleware("http")
async def monitor_predictions(request, call_next):
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)

    start = perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        PREDICTION_REQUESTS.labels(status=str(status)).inc()
        PREDICTION_LATENCY.observe(perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )



''' Tuned model saved in mlflow'''
model_uri = "models:/green-taxi-random-forest@production"

''' Load tuned model once as API starts'''
model = load_pipeline(model_uri)


'''
    @app.post("/predict") verbindet die Funktion mit dieser Route.
    ride: RideInput sorgt dafür, dass FastAPI die eingehenden Daten anhand deiner Klasse prüft.
    Ungültige Eingaben werden abgewiesen.
'''
@app.post("/predict")           #Aufruf bei Post: '/Predict'- Anfrage, übermitteln an API mittels Decorator
def predict(ride: RideInput):
    inputs = ride.model_dump()
    ride_data = pd.DataFrame([inputs])

    predictions = predict_duration(model, ride_data)
    prediction = float(predictions[0])

    log_prediction(inputs, prediction)

    return {"predicted_duration_minutes": prediction}


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