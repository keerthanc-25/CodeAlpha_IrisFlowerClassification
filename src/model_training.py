from pathlib import Path
import logging
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Iris.csv"

MODEL_DIR = BASE_DIR / "models"

IMAGE_DIR = BASE_DIR / "images"

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_DIR.mkdir(exist_ok=True)

class IrisModelTrainer:

    def __init__(self):

        self.data = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()

        self.encoder = LabelEncoder()

        self.results = []
        
    def load_dataset(self):

        logging.info("Loading Dataset...")

        self.data = pd.read_csv(DATA_PATH)

        self.data.drop("Id", axis=1, inplace=True)

        logging.info("Dataset Loaded Successfully")

        print(self.data.head())
        
        
    def prepare_data(self):

        X = self.data.drop("Species", axis=1)

        y = self.data["Species"]

        y = self.encoder.fit_transform(y)

        joblib.dump(
            self.encoder,
            MODEL_DIR / "label_encoder.pkl"
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.X_train = self.scaler.fit_transform(self.X_train)

        self.X_test = self.scaler.transform(self.X_test)

        joblib.dump(
            self.scaler,
            MODEL_DIR / "scaler.pkl"
        )

        logging.info("Data Prepared Successfully")
        
    def initialize_models(self):

        return {

            "Logistic Regression": LogisticRegression(max_iter=200),

            "Decision Tree": DecisionTreeClassifier(random_state=42),

            "Random Forest": RandomForestClassifier(random_state=42),

            "KNN": KNeighborsClassifier(),

            "SVM": SVC(probability=True, random_state=42),

            "Naive Bayes": GaussianNB()
        }
        
            
    def train_models(self):

        logging.info("Training Machine Learning Models...")

        models = self.initialize_models()

        best_accuracy = 0
        best_model = None
        best_model_name = ""

        for model_name, model in models.items():

            # Train model
            model.fit(self.X_train, self.y_train)

            # Predict
            predictions = model.predict(self.X_test)

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, predictions)

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

            # Store results
            self.results.append({

                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1

            })

            logging.info(f"{model_name} Accuracy : {accuracy:.4f}")

            # Save best model
            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        # Save best model
        joblib.dump(
            best_model,
            MODEL_DIR / "iris_model.pkl"
        )

        logging.info(f"Best Model : {best_model_name}")
        logging.info(f"Best Accuracy : {best_accuracy:.4f}")

    def display_results(self):

        results_df = pd.DataFrame(self.results)

        results_df = results_df.sort_values(
            by="Accuracy",
            ascending=False
        )

        print("\n==============================")
        print("Model Comparison")
        print("==============================")

        print(results_df)

        return results_df
    
    def plot_accuracy(self, results_df):

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=results_df,
            x="Model",
            y="Accuracy",
            hue="Model",
            palette="viridis",
            legend=False
        )


        plt.title("Model Accuracy Comparison")

        plt.xlabel("Machine Learning Models")

        plt.ylabel("Accuracy")

        plt.xticks(rotation=20)

        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / "accuracy_comparison.png",
            dpi=300
        )

        plt.show()

        logging.info("Accuracy Comparison Graph Saved")
        
        
        
if __name__ == "__main__":

    trainer = IrisModelTrainer()

    trainer.load_dataset()

    trainer.prepare_data()

    trainer.train_models()

    results = trainer.display_results()

    trainer.plot_accuracy(results)