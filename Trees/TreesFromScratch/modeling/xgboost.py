from pathlib import Path

import numpy as np
import pandas as pd
from math import e 

import typer
from loguru import logger
from tqdm import tqdm

from TreesFromScratch.config import MODELS_DIR, PROCESSED_DATA_DIR

class Node():
    '''
    class variables:
        feature: string
        isLeaf: boolean
        leftNode:
        rightNode:
        value: if leaf: Output, if not leaf: threshold
    '''
    def __init__(self, feature: str,  thresh: float, values: list, gaus, hess):
        self.feature = feature
        self.values = values # If leaf, list of values 
        self.isLeaf= False # by default is a node
        self.thresh = thresh # If not leaf, threshold
        self.gaussians = gaus # residuals calculated from values
        self.hessians = hess # p_i x (1-p_i) calculated from probabilities
        self.probability= None
        self.leftNode = None
        self.rightNode = None
    
class Tree():
    '''
    class Variables:
        root Node
        maxDepth
        min_samples_split
    '''
    def __init__(self, rootNode: Node, maxDepth: int, min_cover : int):
        self.rootNode = rootNode # comes with data
        self.maxDepth = maxDepth
        self.min_cover = min_cover # min samples per leaf
    
    def simScore(self, gausSum: float, hessSum: float, lam: float) -> float:
        return ((gausSum)**2)/ ((hessSum)) + lam
    
    def _gaussians(self, predictions, y) -> None:
        assert len(predictions) == len(y)
        return [-(y[i] - predictions[i]) for i in range(len(y))]
    
    def _hessians(self, probs) -> None: 
        return [prob * (1-prob) for prob in probs]
    
    # find Split by maximizing gaussian
    def findSplit(self, gaussians: list, hessians: list, min_cover=1, lam=1) -> tuple[int, float]:
        # sum the gausssians and hessians once at the beginning, sliding window
        rGSum = sum(gaussians)
        rHSum = sum(hessians)
        lGSum = 0
        lGSum = 0

        maxGain = 0
        tempGain = 0
        rootSim = self.simScore(rGSum, rHSum, lam)
        splitIdx = 0

        # find max Gain to return Split
        for i in range(min_cover, len(gaussians)-min_cover):
            rGSum -= gaussians[i]
            rHSum -= gaussians[i]
            lGSum += gaussians[i]
            lHSum += gaussians[i]
            tempGain = self.simScore(rGSum,rHSum, lam) + self.simScore(lGSum,lHSum, lam) - rootSim
            if tempGain > maxGain:
                maxGain = tempGain
                splitIdx = i
        return (splitIdx, maxGain)

    # values are 0 or 1, predictions are 0-1 probabilities
    def buildTree(self, root: Node, predictions: list, y: list, probs: list, min_cover: int, lam: float):
        # If we reach the end, turn into a leaf and return
        if len(y) <= min_cover:
            root.isLeaf = True
            return
        probs = []
        gaussians = self._gaussians(predictions, y)
        hessians = self._hessians(predictions)
        idx, gain = self.findSplit(gaussians, hessians, min_cover, lam)

        # leftNode = Node(root.feature, thresh=idx, y[:idx], gaussians[:idx], hessians[:idx])
        # rightNode = Node(root.feature, )


    # def fit(self, df: pd.DataFrame, ):

        


     
class XGBoostModel():
    '''
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
    '''
    def __init__(self, num_estimators: int, max_depth: int, learning_rate = 0.3, min_cover = 1, reg_alpha = 0.3, reg_lambda = 0):
        self.num_estimators = num_estimators
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_cover = min_cover
        self.learning_rate = learning_rate
        self.trees = []
        self.gradients = []
        self.hessians = []

'''
    
'''




app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
