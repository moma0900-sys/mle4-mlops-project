import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_DIR / "monitoring" / "prediction_log.csv"
LOG_LOCK = Lock()

FIELDS = [
    "timestamp",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "prediction",
]


def log_prediction(inputs: dict, prediction: float) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "PULocationID": inputs["PULocationID"],
        "DOLocationID": inputs["DOLocationID"],
        "trip_distance": inputs["trip_distance"],
        "prediction": prediction,
    }

    with LOG_LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        )

        with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)

            if write_header:
                writer.writeheader()

            writer.writerow(row)