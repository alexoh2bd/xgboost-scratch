from pathlib import Path
import requests
import typer
import pandas as pd
from loguru import logger

from sklearn.model_selection import train_test_split

from TreesFromScratch.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
import base64


# normalization and binary encode in place
def normalize(df, feature: str) -> None:
    maxVal = df[feature].max()
    minVal = df[feature].min()
    df[feature] = df[feature].apply(lambda x: (x - minVal) / (maxVal - minVal))


def binaryEncode(df, feature: str) -> None:
    df[feature] = df[feature].apply(lambda x: 1 if x == True else 0)


def processSpaceshipData(input_path: str, train_path: str, test_path: str):
    normFeatures = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    binaryFeats = ["CryoSleep", "VIP"]
    transp = "Transported"
    df = pd.read_csv(input_path)

    for f in normFeatures:
        normalize(df, f)
    for f in binaryFeats:
        binaryEncode(df, f)
    if transp in df:
        binaryEncode(df, transp)

    categorical_features = ["HomePlanet", "Destination"]
    df = pd.get_dummies(df, columns=categorical_features)

    df.drop(columns=["Name", "PassengerId", "Cabin"], inplace=True)

    traindf = df.sample(frac=0.8, random_state=42)
    testdf = df.drop(traindf.index)

    traindf.to_csv(train_path, index=False)
    testdf.to_csv(test_path, index=False)


@app.command()
def main(
    input_train_path: Path = RAW_DATA_DIR / "train.csv",
    input_test_path: Path = RAW_DATA_DIR / "test.csv",
    output_train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_test_path: Path = PROCESSED_DATA_DIR / "test.csv",
):
    logger.info("Processing train.csv dataset...")
    processSpaceshipData(input_train_path, output_train_path, output_test_path)
    # processSpaceshipData(input_test_path, output_test_path)
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
