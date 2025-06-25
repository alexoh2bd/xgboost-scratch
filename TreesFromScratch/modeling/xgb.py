from pathlib import Path

import numpy as np
import pandas as pd

import typer
from tqdm import tqdm

from TreesFromScratch.config import PROCESSED_DATA_DIR


import matplotlib.pyplot as plt
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
        Fit multiple Trees to the
        """
        predictions = [0.5 for i in range(X.shape[0])]

        for i in tqdm(range(self.num_estimators)):
            tree = Tree(
                features=features,
                probs=predictions,
                X=X,
                y=y,
                maxDepth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.reg_gamma,
                lam=self.reg_lambda,
            )

            tree.buildTree()

            self.trees.append(tree)
            for j in range(len(predictions)):
                predictions[j] += tree.probs[j] * self.learning_rate

    def predict(self, x):
        def predictTree(t, x):
            currnode = t.rootNode
            while not currnode.isLeaf:
                feature = currnode.feature
                if pd.isna(x[feature]):
                    if currnode.missing_direction == "left":
                        currnode = currnode.leftNode
                    else:
                        currnode = currnode.rightNode
                else:
                    if x[feature] <= currnode.thresh:
                        currnode = currnode.leftNode
                    else:
                        currnode = currnode.rightNode
            return currnode.output

        predictions = 0
        for tree in self.trees:
            tree_output = predictTree(tree, x)
            predictions += self.learning_rate * tree_output

        return 1 if predictions >= 0.5 else 0


def accuracy(y_true, y_pred):
    return np.count_nonzero(y_pred == y_true) / len(y_true)


def precision(y_true, y_pred):
    length = len(y_pred)
    return sum([1 if (y_true[i] == y_pred[i] == 1) else 0 for i in range(length)]) / sum(
        [1 if y_pred[i] == 1 else 0 for i in range(length)]
    )


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
):
    pd.set_option("mode.use_inf_as_na", True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = pd.DataFrame([0 if T else 1 for T in train["Transported"]])

    features = [
        "Age",
        "CryoSleep",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "HomePlanet_Earth",
        "HomePlanet_Europa",
        "HomePlanet_Mars",
        "Destination_55 Cancri e",
        "Destination_PSO J318.5-22",
        "Destination_TRAPPIST-1e",
    ]

    model = XGBoostModel(
        features=features,
        num_estimators=20,
        max_depth=10,
        min_child_weight=10,
        reg_alpha=0.3,
        reg_lambda=10,
        reg_gamma=10,
    )
    model.fit(train[features], y, features)
    # print([row for i, row in test[features].iterrows()])
    predictions = [model.predict(row) for _, row in tqdm(test[features].iterrows())]
    pred = pd.Series(predictions)
    counts = pred.value_counts()

    print(counts)
    print(f"Accuracy: {accuracy(test["Transported"], predictions)}")
    print(f"Precision: {precision(test["Transported"], predictions)}")


if __name__ == "__main__":
    app()
