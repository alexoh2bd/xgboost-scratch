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
    def __init__(self, feature: str, idx0, idx1):
        self.feature = feature 
        # self.residuals = res # residuals calculated from values
        # self.probs = probs # predictions/probs
        # self.logOdds = sum(res)/sum([(prob * (1-prob)) for prob in probs])
        self.idx0 = idx0
        self.idx1 = idx1
        self.split = None # split index in data
        self.isLeaf= False # by default is a node
        self.thresh = None # If not leaf, threshold
        self.leftNode = None
        self.rightNode = None
    def setSplit(self,split):
        self.split=split
    
class Tree():
    '''
    class Variables:
        root Node
        maxDepth
        min_samples_split
    '''
    def __init__(self, probs, y, x, maxDepth=5, min_cover =1, lam=1, gamma=1,lr=0.3):
        self.y = y                  # predicting class
        self.x = x                  # feature class
        self.rootNode = None    # comes with data
        self.maxDepth = maxDepth
        self.min_cover = min_cover  # min samples per leaf
        self.lam = lam              # Lambda: L2 (Lasso)
        self.gamma = gamma          # min split loss
        self.lr = lr #learning rate
        self.len = len(self.y)

        # calculate residuals(gaussians) and hessians from probs upon init
        self.probs = probs
        self.residuals = [(y[i] - probs[i]) for i in range(self.len)]
        self.hessians = [(prob * (1-prob)) for prob in probs]
    
    def simScore(self, resSum: float, hessSum: float) -> float:
        return ((resSum)**2)/ ((hessSum) + self.lam)
    
    # find Split by maximizing Gain
    def findSplit(self, idx0: int, idx1: int, min_len=1):
        # sum the gausssians and hessians once at the beginning, sliding window
        assert (idx0 < idx1 and isinstance(idx0, int) and isinstance(idx1, int))
        lRSum = 0
        lHSum = 0
        rRSum = sum(self.residuals[idx0:idx1])
        rHSum = sum(self.hessians[idx0:idx1])
        
        maxGain = 0
        tempGain = 0
        rootSim = self.simScore(rRSum, rHSum)
        splitIdx = 0
        # maxLSim=0
        # maxRSim=0

        # find max Gain to return Split
        for i in range(idx0, idx1):
            lRSum += self.residuals[i]
            lHSum += self.hessians[i]
            rRSum -= self.residuals[i]
            rHSum -= self.hessians[i]
            # print(f"{lRSum} {lHSum} {rRSum} {rHSum}")
            lSim = self.simScore(lRSum,lHSum) 
            rSim = self.simScore(rRSum,rHSum) 

            tempGain= rSim+lSim- rootSim
            if tempGain > maxGain:
                maxGain = tempGain
                # maxLSim=lSim
                # maxRSim=rSim
                splitIdx = i
        return splitIdx

    # values are 0 or 1, predictions are 0-1 probabilities
    def buildBranch(self, node: Node, idx0, idx1, currdepth:int):
        # If we reach the end, turn into a leaf and return
        assert isinstance(node, Node) 
        cover = sum(self.hessians[idx0:idx1])
        if  cover<= self.min_cover:
            node.isLeaf = True
            return
        if currdepth>=self.maxDepth:
            node.isLeaf = True
            return
        
        split = self.findSplit(idx0, idx1)

        thresh = (self.x[split] + self.x[(split+1)])/2 
        node.thresh = thresh

        node.leftNode = Node(node.feature, idx0, split)
        node.rightNode = Node(node.feature, split, idx1)

        print(f"{'  ' * currdepth}Splitting at indices ({idx0}, {split}, {idx1}), threshold {thresh}, cover {cover}, and probability {self.probs[split]}")
        
        self.buildBranch(node.leftNode, idx0, split, currdepth+1)
        self.buildBranch(node.rightNode, split, idx1, currdepth+1)

    
    def buildTree(self, features: list):
        self.rootNode = Node(features[0], 0, self.len)
        self.buildBranch(self.rootNode, 0, self.len, 0) 
        # then prune 
        # then update probabilities, return new predictions
        self.visualizeTree(self.rootNode, 0)


    def visualizeTree(self, currnode, depth):
        print(f"{'  '*depth}Node Depth: {depth}. Split Index: {currnode.split}. Bounds: {currnode.idx0} - {currnode.idx1}. ")
        if currnode.isLeaf and currnode.leftNode is None and currnode.rightNode is None:
            print(f"{'  '*depth}Leaf!")
            return
        if currnode.leftNode:
            self.visualizeTree(currnode.leftNode, depth+1)
        if currnode.rightNode:
            self.visualizeTree(currnode.rightNode, depth+1)




        
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
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    df = pd.read_csv(train_path)
    df.sort_values(by='Age', inplace=True)
    # print(df[:5])
    transported = df['Transported']
    transported = [0 if T else 1 for T in transported]
    # print(transported[:10])
    age = df['Age'].to_list()
    # print((age[:10]))
    probs = [0.5 for i in range(len(age))]
    residuals = [(transported[i] - probs[i]) for i in range(len(transported))]
    root = Node('Age', residuals,probs)
    tree = Tree(probs, transported, age, min_cover=300)
    features= ['Age']
    tree.buildTree(features)
    # print(transported)
    # logger.info(transported[:5])
    # logger.info(age[:5])
    # logger.info(len(probs))

if __name__ == "__main__":
    app()
