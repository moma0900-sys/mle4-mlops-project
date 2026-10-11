'''Loading locally existing parquet files and cleaning data from faulty entries'''
'''Dataframes'''
import pandas as pd

#Used in train.py
def load_data(path):
    df = pd.read_parquet(path)
    return df

#Used in train.py
def clean_data(df):
    df = df.copy()
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds()/60

    duration_window = df["duration"].between(1,60)
    df_clean = df[duration_window].copy()

    distance_window = df_clean["trip_distance"].between(0.1,100)
    df_clean = df_clean[distance_window].copy()

    print("--- Validating DURATIONS --- in data")
    print("Kept trips:", duration_window.sum())
    print("Excluded trips:", (~duration_window).sum())
    print(f"Excluded trips [%]: {(~duration_window).mean():.2%}")

    print("--- Validating DISTANCES --- in data")
    print("Kept trips:", distance_window.sum())
    print("Excluded trips:", (~distance_window).sum())
    print(f"Excluded trips [%]: {(~distance_window).mean():.2%}")

    return df_clean