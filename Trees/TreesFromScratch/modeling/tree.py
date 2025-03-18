import numpy as np
import pandas as pd
from math import e, log

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Node:

    def __init__(self, feature: str, idx0, idx1):
        """
        Creates new Node instance

        Parameters
        ----------

        feature: string
            node feature name
        idx0: int
            lower bound for item index range
        idx1: int
            upper bound for item index range
        """
        self.feature = feature
        self.idx0 = idx0
        self.idx1 = idx1
        self.split = None  # split index in data
        self.isLeaf = False  # by default is a node
        self.thresh = None  # If not leaf, threshold
        self.leftNode = None
        self.rightNode = None
        self.prob = None
        self.output = None


class Tree:
    """
    class Variables:
        root Node
        maxDepth
        min_samples_split
    """

    def __init__(
        self, probs: list, x: list, y: list, maxDepth=5, min_child_weight=1, lam=1, gamma=1, lr=0.3
    ):
        """
        Parameters
        ----------
        probs: Probabilities (predictions) for y
        x: feature class
        y: actual y predicted class
        rootNode: Tree's root node
        maxDepth: maximum depth of tree
        min_child_weight: minimum weight per leaf
        lam: Lambda L2 Regularization Constant (Ridge)
        gamma: Gamma Regularization Constant for minimizing split loss
        lr: learning rate

        Local Variables
        ---------------
        residuals: negative gaussians, difference between actual y and predicted y (probability of y)
        hessians: calculated from probs
        len: dataset length
        """

        self.x = x
        self.y = y
        self.rootNode = None
        self.maxDepth = maxDepth
        self.min_child_weight = min_child_weight
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.len = len(self.y)

        # calculate residuals(gaussians) and hessians from probs upon init
        self.probs = probs
        self.residuals = list(map(lambda x, z: x - z, self.y, self.probs))
        self.hessians = list(map(lambda x: x * (1 - x), self.probs))

    # Calculate the similarity score for calculating gain
    def simScore(self, resSum: float, hessSum: float) -> float:
        assert ((hessSum) + self.lam) != 0
        return ((resSum) ** 2) / ((hessSum) + self.lam)

    # Find Split Index which maximizing Gain
    def findSplit(self, idx0: int, idx1: int):
        """
        Parameters
        ---------
        idx0: left bound
        idx1: right bound

        Sum the gausssians and hessians once at the beginning, sliding window
        """
        assert idx0 < idx1 and isinstance(idx0, int) and isinstance(idx1, int)
        lRSum = sum(self.residuals[idx0 : idx0 + self.min_child_weight])
        lHSum = sum(self.residuals[idx0 : idx0 + self.min_child_weight])
        rRSum = sum(self.residuals[idx0 + self.min_child_weight : idx1])
        rHSum = sum(self.hessians[idx0 + self.min_child_weight : idx1])

        maxGain = 0
        tempGain = 0
        rootSim = self.simScore(rRSum, rHSum)
        splitIdx = idx0 + 1

        # Iterate through Residuals and Hessians to calc Similarity Score and find max Gain
        for i in range(idx0 + self.min_child_weight, idx1 - self.min_child_weight):
            lRSum += self.residuals[i]
            lHSum += self.hessians[i]
            rRSum -= self.residuals[i]
            rHSum -= self.hessians[i]
            if lHSum + self.lam == 0 or rHSum + self.lam == 0:
                continue

            lSim = self.simScore(lRSum, lHSum)
            rSim = self.simScore(rRSum, rHSum)

            tempGain = rSim + lSim - rootSim

            # check maxgain and prune if gain - gamma < 0
            if tempGain > maxGain and tempGain - self.gamma > 0:
                maxGain = tempGain
                splitIdx = i
        return splitIdx

    # Values are 0 or 1, predictions are 0-1 probabilities
    def buildBranch(self, node: Node, idx0: int, idx1: int, currdepth: int):
        """ """

        # If we reach max depth or cover < minimum allowed (prune), turn into a leaf
        assert idx0 < idx1

        cover = sum(self.hessians[idx0:idx1]) - self.lam
        denom = sum(self.hessians[idx0:idx1]) + self.lam
        assert denom != 0
        if cover <= self.min_child_weight or currdepth >= self.maxDepth:
            node.isLeaf = True
            node.output = sum(self.residuals[idx0:idx1]) / denom
            return

        split = self.findSplit(idx0, idx1)

        thresh = (self.x[split] + self.x[(split + 1)]) / 2
        node.thresh = thresh

        node.leftNode = Node(node.feature, idx0, split)
        node.rightNode = Node(node.feature, split, idx1)

        print(
            f"{'  ' * currdepth}Splitting at indices ({idx0}, {split}, {idx1}), threshold {thresh}, cover {cover}, and probability {self.probs[split]}"
        )

        self.buildBranch(node.leftNode, idx0, split, currdepth + 1)
        self.buildBranch(node.rightNode, split, idx1, currdepth + 1)

    # Recurse down to each leaf, calculate new output value, then update all probs
    def updateProbs(self, currnode: Node, depth: int):
        print(f"{'  '*depth}Node Depth: {depth}. Threshold: {currnode.thresh} ")
        if currnode.isLeaf and currnode.leftNode is None and currnode.rightNode is None:
            i, j = currnode.idx0, currnode.idx1

            # lambda L2(Lasso) Regulation in denominator
            denom = sum(self.hessians[i:j]) + self.lam
            currnode.output = sum(self.residuals[i:j]) / denom
            logodd = log(self.probs[i] / (1 - self.probs[i]))
            score = logodd + self.lr * currnode.output  # Log(odds) prediction

            # recalculate probs, residuals, hessians
            self.probs[i:j] = [e**score / (1 + e**score) for _ in range(i, j)]
            self.residuals[i:j] = [(self.y[k] - self.probs[k]) for k in range(i, j)]
            self.hessians[i:j] = [(self.probs[k] * (1 - self.probs[k])) for k in range(i, j)]

            currnode.prob = self.probs[i]
            print(f"{'  '*depth}LEAF! Bounds: {i} - {j}.")
            print(f"{'  '*depth}  Output: {currnode.output}. Log(Odds): {logodd}.")
            print(f"{'  '*depth}  Raw Score: {score}. Prediction: {self.probs[i]}.\n")
            return
        else:
            print("\n")
        if currnode.leftNode:
            self.updateProbs(currnode.leftNode, depth + 1)
        if currnode.rightNode:
            self.updateProbs(currnode.rightNode, depth + 1)

    def buildTree(self, feature: str):
        """
        Build new tree for each iteration, update probabilities(predictions) after each tree

        Declare Root
        Build Branch from root
        Then update probabilities
        """
        self.rootNode = Node(feature, 0, self.len)
        self.buildBranch(self.rootNode, idx0=0, idx1=self.len, currdepth=0)
        self.updateProbs(currnode=self.rootNode, depth=0)

    def extractThresh(self):
        """
        returns the tree data: thresholds and leaf probabilities
        """
        # Initialize lists for each feature
        tree_thresholds = [min(self.x), max(self.x)]
        tree_probs = []

        # Queue for traversing the tree (BFS) starting with root
        queue = [self.rootNode]
        while queue:
            node = queue.pop(0)
            # Note leaf node probs
            if node.isLeaf:
                tree_probs.append(node.prob)
                continue
            tree_thresholds.append(node.thresh)
            if node.leftNode:
                queue.append(node.leftNode)
            if node.rightNode:
                queue.append(node.rightNode)
        return (
            tree_thresholds,
            tree_probs,
        )

    def visualizeTree(self):
        """
        Visualize Predicted Probabilities through Matplotlib
        Inputs: None
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            self.x, self.y, c=self.y, cmap="viridis", alpha=0.6, edgecolors="w", s=30
        )
        thresholds, probs = self.extractThresh()

        thresholds = np.ravel(thresholds)
        thresholds.sort()
        print(f"thresholds: {thresholds}.  probs:{probs}.")
        for b in thresholds:
            ax.axvline(x=b, color="r", linestyle="-", linewidth=1)
        for i in range(len(probs)):
            ax.hlines(
                y=probs[i],
                xmin=thresholds[i],
                xmax=thresholds[i + 1],
                color="b",
                linestyle="-",
                linewidth=1,
            )

        plt.tight_layout()
        plt.show()
