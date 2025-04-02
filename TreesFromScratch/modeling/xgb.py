from pathlib import Path

import numpy as np
import pandas as pd
from math import e, log

import typer
from loguru import logger
from tqdm import tqdm

from TreesFromScratch.config import MODELS_DIR, PROCESSED_DATA_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tree import Tree


class XGBoostModel:
    """
    Specify Classification or Regression
    Training: Use "train" function with train set features and lables
    Predicting: Use "predict"
    class variables:
        trees: array of trees and their root nodes
        num_estimators
        max_depth
        reg_alpha: L1
        reg_lambda: L2 regularization constant
        min_cover: min_residuals_per_leaf
        learning_rate: default = 0.3
    """

    def __init__(
        self,
        features: str,
        num_estimators: int,
        max_depth: int,
        learning_rate=0.3,
        min_child_weight=1,
        reg_alpha=0.3,
        reg_lambda=0,
        reg_gamma=0,
    ):
        self.features = features
        self.num_estimators = num_estimators
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.score = 0.5
        self.trees = []
        self.gradients = []
        self.hessians = []

    def fit(self, X, y, features: str):
        """
        Fit multiple
        """
        predictions = [0.5 for i in range(X.shape[0])]

        for i in range(1):
            self.trees.append(
                Tree(
                    features=features,
                    probs=predictions,
                    X=X,
                    y=y,
                    maxDepth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    gamma=self.reg_gamma,
                    lam=self.reg_lambda,
                )
            )
            self.trees[i].buildTree()
            # predictions = self.trees[i].probs


app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    df = pd.read_csv(train_path)

    y = [0 if T else 1 for T in df["Transported"]]

    features = ["Age", "CryoSleep", "VIP"]

    model = XGBoostModel(
        features=features,
        num_estimators=1,
        max_depth=4,
        min_child_weight=200,
        reg_alpha=0.3,
        reg_lambda=10,
        reg_gamma=10,
    )
    model.fit(df[features], y, features)


if __name__ == "__main__":
    app()
