from pathlib import Path
import requests
import typer
import pandas as pd
from loguru import logger
from tqdm import tqdm
import os
from dotenv import load_dotenv

from TreesFromScratch.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()
import base64


# normalization and binary encode in place
def normalize(df, feature: str) -> None:
    maxVal = df[feature].max()
    df[feature] = df[feature].apply(lambda x: x / maxVal)


def binaryEncode(df, feature: str) -> None:
    df[feature] = df[feature].apply(lambda x: 1 if x == True else 0)


def processSpaceshipData(input_path: str, output_path: str):
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

    df.drop(columns=["Name", "PassengerId", "Cabin"], inplace=True)
    df.to_csv(output_path, index=False)


@app.command()
def main(
    input_train_path: Path = RAW_DATA_DIR / "train.csv",
    input_test_path: Path = RAW_DATA_DIR / "test.csv",
    output_train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_test_path: Path = PROCESSED_DATA_DIR / "test.csv",
):
    logger.info("Processing train.csv dataset...")
    # processSpaceshipData(input_train_path, output_train_path)
    processSpaceshipData(input_test_path, output_test_path)
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
