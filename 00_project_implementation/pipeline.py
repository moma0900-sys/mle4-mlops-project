'''Administring and documenting results of trained models in mlflow'''
from pathlib import Path
import mlflow

project_dir = Path(__file__).resolve().parent.parent
tracking_db = project_dir / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")


'''Loading tuned model'''
#Used in main.py
def load_pipeline(model_uri):
    loaded_tuned_pipeline = mlflow.sklearn.load_model(model_uri)
    return loaded_tuned_pipeline


'''Predicting duration on ride_data'''
#Used in main.py
def predict_duration(pipeline, ride_data):
    predictions = pipeline.predict(ride_data)
    return predictions