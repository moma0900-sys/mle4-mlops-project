'''Dataframes'''
import pandas as pd
from pipeline import load_pipeline, predict_duration


def main() -> None:
    model_uri = "models:/green-taxi-random-forest@production"

    example_ride =[
        {"PULocationID": 229, "DOLocationID": 237, "trip_distance": 1.6}
    ]
    ride_data = pd.DataFrame(example_ride)
    # ride_data = pd.read_parquet(Path("notebooks/data/cleaned/cleaned_yellow_tripdata_2025_01.parquet"))
    # ride_data = ride_data[
    #     ["PULocationID", "DOLocationID", "trip_distance"]
    #     ].head()

# from pipeline.py
    pipeline = load_pipeline(model_uri)
    print("Model loaded successfully")

# from pipeline.py
    predictions = predict_duration(pipeline, ride_data)
    print("Prediction finished successfully on data:")
    print(ride_data)

    print("Predicted duration:", predictions)


if __name__ == "__main__":
    main()