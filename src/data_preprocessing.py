from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class DataPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

        # Dataset Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.dataset_path = BASE_DIR / "data" / "Iris.csv"

    # ---------------------------------------------------

    def load_dataset(self):

        """
        Load dataset from CSV file.
        """

        logging.info("Loading Dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        logging.info("Dataset Loaded Successfully")

        return df

    # ---------------------------------------------------

    def dataset_info(self, df):

        print("\n========== Dataset Shape ==========")
        print(df.shape)

        print("\n========== First Five Rows ==========")
        print(df.head())

        print("\n========== Dataset Information ==========")
        print(df.info())

        print("\n========== Missing Values ==========")
        print(df.isnull().sum())

        print("\n========== Duplicate Rows ==========")
        print(df.duplicated().sum())

        print("\n========== Statistical Summary ==========")
        print(df.describe())

    # ---------------------------------------------------

    def preprocess_data(self, df):
   
    # Remove unnecessary Id column
        df = df.drop(columns=["Id"])

    # Features
        X = df.drop(columns=["Species"])

    # Target
        y = df["Species"]

        logging.info("Feature and Target Separation Completed")

        return X, y

    # ---------------------------------------------------

    def split_dataset(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        logging.info("Train-Test Split Completed")

        return X_train, X_test, y_train, y_test

    # ---------------------------------------------------

    def scale_features(self, X_train, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train)

        X_test_scaled = self.scaler.transform(X_test)

        logging.info("Feature Scaling Completed")

        return X_train_scaled, X_test_scaled


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    processor = DataPreprocessor()

    df = processor.load_dataset()

    processor.dataset_info(df)

    X, y = processor.preprocess_data(df)

    X_train, X_test, y_train, y_test = processor.split_dataset(X, y)

    X_train_scaled, X_test_scaled = processor.scale_features(
        X_train,
        X_test
    )

    print("\n========== Summary ==========")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print("\nPreprocessing Completed Successfully.")


if __name__ == "__main__":
    main()