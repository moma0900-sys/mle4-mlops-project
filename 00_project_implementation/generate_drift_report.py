from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from prepare_data import load_data, clean_data


def main():
    project_dir = Path(__file__).resolve().parent.parent

    reference_path = (
        project_dir / "data" / "green_tripdata_2024-01.parquet"
    )
    log_path = project_dir / "monitoring" / "prediction_log.csv"
    report_path = (
        project_dir / "monitoring" / "reports" / "input_drift.html"
    )

    features = ["PULocationID", "DOLocationID", "trip_distance"]

    # Die letzten 200 protokollierten Anfragen auswählen.
    current_data = pd.read_csv(log_path)

    if len(current_data) < 2000:
        raise ValueError("At least 2000 predictions necessary.")

    current_data = current_data.tail(2000)[features].copy()

    # Dieselbe Bereinigung wie beim Training verwenden.
    reference_data = clean_data(load_data(reference_path))
    reference_data = reference_data[features].copy()

    # Orts-IDs sind Kategorien, keine numerischen Messwerte.
    definition = DataDefinition(
        categorical_columns=["PULocationID", "DOLocationID"],
        numerical_columns=["trip_distance"],
    )

    reference = Dataset.from_pandas(
        reference_data,
        data_definition=definition,
    )
    current = Dataset.from_pandas(
        current_data,
        data_definition=definition,
    )

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(
        current_data=current,
        reference_data=reference,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(report_path))

    print(f"Referenz: {len(reference_data)} drives")
    print(f"Current sample: {len(current_data)} requests")
    print(f"Protocol saved: {report_path}")


if __name__ == "__main__":
    main()