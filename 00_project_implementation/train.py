'''.'''
from pathlib import Path
import os

'''Split Dataset for testing'''
from sklearn.model_selection import train_test_split

'''For transforming "PULocationID", "DOLocationID" without chaning the trip distance (just parsing)'''
from sklearn.compose import ColumnTransformer

'''For tansforming "PULocationID", "DOLocationID" to binary values'''
from sklearn.preprocessing import OneHotEncoder

'''Pipeline'''
from sklearn.pipeline import Pipeline

''' and model class'''
from sklearn.ensemble import RandomForestRegressor

''' For calculation RSME'''
from sklearn.metrics import root_mean_squared_error

''' Administring and documenting results of trained models'''
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import argparse

''' From other script files'''
from prepare_data import load_data, clean_data

'''Load secrets for dagshub'''
from dotenv import load_dotenv



load_dotenv(
    Path(__file__).resolve().parent.parent / ".env",
    override=False,
)


# Calling train.py with argument of parquet file, e.g.:
# uv run python 00_project_implementation/train.py green_tripdata_2024-01.parquet
def main(load_parquet: str) -> None:

    project_dir = Path(__file__).resolve().parent.parent
    parquet_file_raw = project_dir / "data" / load_parquet

    if not parquet_file_raw.is_file():
        raise FileNotFoundError(
            f"File not found: {parquet_file_raw}. "
            "Download files via 'download_data.py' first."
        )

# split filename for mlflow tags later on
    filename = parquet_file_raw.stem

    color, dataset_type, year_month = filename.split("_")
    year, month = year_month.split("-")

    year = int(year)
    month = int(month)

    if color != "green" or dataset_type != "tripdata":
        raise ValueError("Green Taxi data expected in filename.")

    if not 1 <= month <= 12:
        raise ValueError("Invalid month representation in filename")

# from prepare_data.py
    df = load_data(parquet_file_raw)
    df_clean = clean_data(df)

    print(f"Using {parquet_file_raw} for training.")
    print("Rows after cleaning:", len(df_clean))

# Online mflow.db, tracking on dagshub.com
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Register new experiment named "green-taxi-random-forest", description and tags
# Expermient sammelt zusammengehörige Trainingsläufe
    model_name = "green-taxi-random-forest"
    mlflow.set_experiment(model_name)

    client = MlflowClient()
    try:
        client.create_registered_model(
            name=model_name,
            description="Random Forest predicting NY green taxi duration.",
            tags={
                "framework": "scikit-learn",
                "developer": "Markus",
            },
        )
        print("Model registered: ", model_name)

#If experiment is already exising
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        print(f"Model entry '{model_name}' already exists.")

#Start training the model with choosen parquet file
    train(df_clean, model_name, color, year, month)



def train(df, model_name, color, year, month):
    # Einen Modelleintrag in mlflow anlegen
    client = MlflowClient()

    print("--- Generating training and vaildation data set ---")
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    X = df[features].copy()
    y = df["duration"].copy()

#Setting up the test and validation data set
    X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    categorical_features = ["PULocationID", "DOLocationID"]
    preprocessor = ColumnTransformer(
        transformers = [
            ("locations", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    print("--- Creating pipeline ---")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(max_depth=20, min_samples_leaf=5, random_state=42))
    ])

    ''' Start mlflow for registrating model run'''
# Run: Ein einzelner Trainingsdurchlauf mit Einstellungen, Ergebnissen und Modell
    with mlflow.start_run(run_name="random-forest-script") as run:

        tags = {
            "model": "random forest",
            "developer": "Markus",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": "duration",
        }
        mlflow.set_tags(tags)

        print("--- Train the model ---")
        pipeline.fit(X_train,y_train)

        print("--- Model starts predicting on data ---")
        y_pred = pipeline.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"Done, RMSE: {rmse} from {len(X_val)} rows")

# Einstellungen und Ergebnis protokollieren
        mlflow.log_params({"training_rows": len(X_train), "max_depth": 20, "min_samples_leaf": 5, "random_state": 42})
        mlflow.log_metric("rmse_validation", rmse)

# Neu trainierte Pipeline in mlflow speichern
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

# Gespeichertes Modell als neue Version registrieren
        model_version = client.create_model_version(
            name=model_name,
            source=model_info.model_uri,
            run_id=run.info.run_id,
        )

        print(
            f"Model '{model_name}' "
            f"registered as version {model_version.version}."
        )


# Validation, if new registered_model predicts better than the current production_version
        registered_model = client.get_registered_model(model_name)
        production_version = registered_model.aliases.get("production")

# If there is no production model set by now
        if production_version is None:
            promote = True

# Read out current production_model
        else:
            production_model = mlflow.sklearn.load_model(
                f"models:/{model_name}/{production_version}"
            )

# Aktuell verwendetes Production-Model auf denselben Daten bewerten wie das neue Modell
            production_predictions = production_model.predict(X_val)

            production_rmse = root_mean_squared_error(
                y_val,
                production_predictions,
            )

            mlflow.log_metric(
                "rmse_production_on_validation",
                production_rmse,
            )

            print(f"RMSE current model: {production_rmse:.4f}")
            print(f"RMSE new model:      {rmse:.4f}")

            promote = rmse < production_rmse

# Alias nur beim ersten Lauf oder bei einer Verbesserung setzen
        if promote:
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=model_version.version,
            )
            print(f"New production version {model_version.version}.")

        else:
            print("No RMSE improvement. No change to production model")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train model on Random Forest with local parquet-file."
    )

    parser.add_argument(
        "load_parquet",
        help="Name of parquet-file in folder 'data'",
    )

    args = parser.parse_args()
    main(args.load_parquet)