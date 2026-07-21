from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class IrisPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_DIR / "iris_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    def predict_species(
        self,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ):

        sample = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        sample = self.scaler.transform(sample)

        prediction = self.model.predict(sample)

        return self.encoder.inverse_transform(prediction)[0]

    def predict_proba(
        self,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ):

        # Check if the model supports probabilities
        if not hasattr(self.model, "predict_proba"):
            return None

        sample = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        sample = self.scaler.transform(sample)

        probabilities = self.model.predict_proba(sample)[0]

        return {
            species: float(prob)
            for species, prob in zip(self.encoder.classes_, probabilities)
        }