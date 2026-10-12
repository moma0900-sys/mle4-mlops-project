'''Download data as parquet and save them locally'''
from pathlib import Path
from urllib.request import urlretrieve

#Define which dataset will be downloaded
parameters = [{
    "year": 2024,
    "month": 1,
    "color": "green"
},
{
    "year": 2024,
    "month": 5,
    "color": "green"
},
{
    "year": 2024,
    "month": 8,
    "color": "green"
},
{
    "year": 2025,
    "month": 1,
    "color": "green"
}]


def download_data():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for parameter in parameters:
        year = parameter["year"]
        month = parameter["month"]
        color = parameter["color"]

        filename = f"{color}_tripdata_{year}-{month:02d}.parquet"
        data_path = data_dir / filename

        if data_path.exists():
            print(f"Bereits vorhanden: {filename}")
            continue

        url = (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"{filename}"
        )

        print(f"Lade herunter: {filename}")
        urlretrieve(url, data_path)
        print(f"Gespeichert: {data_path}")


if __name__ == "__main__":
    download_data()