from pathlib import Path
import logging
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Iris.csv"

MODEL_DIR = BASE_DIR / "models"

IMAGE_DIR = BASE_DIR / "images"

class IrisModelEvaluator:

    def __init__(self):

        self.data = None

        self.model = None

        self.scaler = None

        self.encoder = None

        self.X_test = None

        self.y_test = None
        
    def load_dataset(self):

        logging.info("Loading Dataset...")

        self.data = pd.read_csv(DATA_PATH)

        self.data.drop("Id", axis=1, inplace=True)
        
    def load_saved_objects(self):

        logging.info("Loading Saved Model...")

        self.model = joblib.load(
            MODEL_DIR / "iris_model.pkl"
        )

        self.scaler = joblib.load(
            MODEL_DIR / "scaler.pkl"
        )

        self.encoder = joblib.load(
            MODEL_DIR / "label_encoder.pkl"
        )

    def prepare_test_data(self):

        X = self.data.drop("Species", axis=1)

        y = self.encoder.transform(
            self.data["Species"]
        )

        _, self.X_test, _, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.X_test = self.scaler.transform(
            self.X_test
        )
        
    def evaluate_model(self):

        predictions = self.model.predict(
            self.X_test
        )

        accuracy = accuracy_score(
            self.y_test,
            predictions
        )

        precision = precision_score(
            self.y_test,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            self.y_test,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            average="weighted"
        )

        print("\n==========================")
        print("Model Evaluation")
        print("==========================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report\n")

        print(
            classification_report(
                self.y_test,
                predictions,
                target_names=self.encoder.classes_
            )
        )

        return predictions

    def plot_confusion_matrix(self, predictions):

        cm = confusion_matrix(
            self.y_test,
            predictions
        )

        plt.figure(figsize=(7,6))

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            fmt="d",
            xticklabels=self.encoder.classes_,
            yticklabels=self.encoder.classes_
        )

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted Label")

        plt.ylabel("True Label")

        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / "confusion_matrix.png",
            dpi=300
        )

        plt.show()

        logging.info(
            "Confusion Matrix Saved Successfully"
        )

if __name__ == "__main__":

    evaluator = IrisModelEvaluator()

    evaluator.load_dataset()

    evaluator.load_saved_objects()

    evaluator.prepare_test_data()

    predictions = evaluator.evaluate_model()

    evaluator.plot_confusion_matrix(predictions)