'''Administring and documenting results of trained models in mlflow'''
import os
import mlflow


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


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