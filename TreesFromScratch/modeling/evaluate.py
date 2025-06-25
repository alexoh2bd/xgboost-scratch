import typer
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from loguru import logger
import pickle
from TreesFromScratch.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_model(model_path: Path):
    """Load the trained XGBoost model."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        logger.error(f"Model not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(
    model_path: Path = MODELS_DIR / "model.pkl",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    features: List[str] = ["Age", "CryoSleep", "VIP"],
):
    """
    Evaluate the trained model on test data.

    Parameters:
    -----------
    model_path: Path
        Path to the saved model
    test_path: Path
        Path to the test data
    features: List[str]
        List of feature columns to use

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    try:
        # Load model
        model = load_model(model_path)

        # Load test data
        df = pd.read_csv(test_path)

        # Extract features and target
        X_test = df[features]
        y_test = df["Transported"]

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        metrics = {"accuracy": accuracy, "precision": precision, "num_samples": len(y_test)}

        logger.success(f"Evaluation metrics:")
        logger.success(f"Accuracy: {accuracy:.4f}")
        logger.success(f"Precision: {precision:.4f}")
        logger.success(f"Number of samples: {len(y_test)}")

        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    features: List[str] = typer.Option(
        ["Age", "CryoSleep", "VIP"], help="Features to use for prediction"
    ),
):
    """
    Evaluate the trained XGBoost model on test data.

    Args:
        test_path: Path to the test data CSV file
        model_path: Path to the saved model file
        features: List of feature columns to use
    """
    logger.info("Starting model evaluation...")

    try:
        evaluate_model(model_path=model_path, test_path=test_path, features=features)
        logger.success("Model evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    app()
