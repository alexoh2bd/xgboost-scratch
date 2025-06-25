from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
from xgb import XGBoostModel
import pickle
from TreesFromScratch.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    df = pd.read_csv(train_path)
    y = [0 if T else 1 for T in df["Transported"]]
    features = ["Age", "CryoSleep", "VIP"]
    model = XGBoostModel(
        features=features,
        num_estimators=10,
        max_depth=4,
        min_child_weight=200,
        reg_alpha=0.3,
        reg_lambda=10,
        reg_gamma=10,
    )
    model.fit(df[features], y, features)
    logger.success("Modeling training complete.")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    # -----------------------------------------


if __name__ == "__main__":
    app()
