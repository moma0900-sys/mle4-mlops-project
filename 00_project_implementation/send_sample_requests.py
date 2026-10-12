from pathlib import Path
from time import sleep

import requests

from prepare_data import load_data, clean_data


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "green_tripdata_2024-01.parquet"

    data = clean_data(load_data(data_path))
    sample = data.sample(n=2000)

    with requests.Session() as session:
        for number, row in enumerate(sample.itertuples(index=False), start=1):
            payload = {
                "PULocationID": int(row.PULocationID),
                "DOLocationID": int(row.DOLocationID),
                "trip_distance": float(row.trip_distance),
            }

            response = session.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            if number % 100 == 0:
                print(f"{number}/2000 sucessfull requests")

            sleep(0.1)


if __name__ == "__main__":
    main()