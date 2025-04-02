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

    def __init__(self):
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
        self.feature = "feature"
        self.split = None  # split index in data
        self.isLeaf = False  # by default is a node
        self.thresh = None  # If not leaf, threshold
        self.leftNode = None
        self.rightNode = None
        self.prob = None
        self.hessians = None
        self.residuals = None
        self.X_indices = []
        self.output = None


class Tree:
    """
    class Variables:
        root Node
        maxDepth
        min_samples_split
    """

    def __init__(
        self,
        features: list,
        probs: list,
        X: list,
        y: list,
        maxDepth=3,
        min_child_weight=200,
        lam=1,
        gamma=1,
        lr=0.3,
    ):
        """
        Parameters
        ----------
        features: list of features
        probs: Probabilities (predictions) for y
        X: predictors matrix (pandas dataframe )
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
        self.features = features
        self.X = X
        self.y = y
        self.rootNode = None
        self.maxDepth = maxDepth
        self.min_child_weight = min_child_weight
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.num_feats = len(self.features)
        self.len = len(self.y)

        # Sorted X and X indices
        self.X_sorted = {}
        self.X_indices = {}

        # calculate residuals(gaussians) and hessians from probs upon init
        self.probs = probs
        self.residuals = list(map(lambda x, z: x - z, self.y, self.probs))
        self.hessians = list(map(lambda x: x * (1 - x), self.probs))

    # Calculate the similarity score for calculating gain
    def simScore(self, resSum: float, hessSum: float) -> float:
        assert ((hessSum) + self.lam) != 0
        return ((resSum) ** 2) / ((hessSum) + self.lam)

    # Sort each feature in X and place initial indices
    def sort_X(self, indices):
        X = self.X.loc[indices]
        if "index" not in X.columns:
            pd.options.mode.copy_on_write = True
            X["index"] = X.index

        X_sorted, X_indices = {}, {}

        for feature in self.X.columns:
            copy = X.sort_values(by=feature)
            X_sorted[feature] = copy[feature]
            X_indices[feature] = copy["index"]
        X.reset_index()
        return X_sorted, X_indices

    # Set Residuals and Hessians for each feature according to the new order
    def calcResHess(self, indices):
        """
        indices: sorted indices used to mask self.y and self.probs
        """
        y = np.array(self.y)[indices]
        probs = np.array(self.probs)[indices]
        residuals = list(map(lambda x, z: x - z, y, probs))
        hessians = list(map(lambda x: x * (1 - x), probs))
        return residuals, hessians

    # Find Split Index which maximizing Gain
    def findSplit(self, res: list, hess: list):
        """
        Parameters
        ---------
        feature: str
        idx0: list of left bounds num_features
        idx1: list of right bounds

        Sum the gausssians and hessians once at the beginning, sliding window
        """

        total_residuals = sum(res)
        total_hessians = sum(hess)
        lRSum = sum(res[: self.min_child_weight])
        lHSum = sum(res[: self.min_child_weight])
        rRSum = total_residuals - lRSum
        rHSum = total_hessians - lHSum

        maxGain = 0
        tempGain = 0
        rootSim = self.simScore(rRSum, rHSum)
        splitIdx = self.min_child_weight
        idxrange = len(res)

        # Iterate through Residuals and Hessians to calc Similarity Score and find max Gain
        for i in range(self.min_child_weight, idxrange - self.min_child_weight):
            lRSum += res[i]
            lHSum += hess[i]
            rRSum -= res[i]
            rHSum -= hess[i]
            if lHSum + self.lam == 0 or rHSum + self.lam == 0:
                continue

            lSim = self.simScore(lRSum, lHSum)
            rSim = self.simScore(rRSum, rHSum)

            tempGain = rSim + lSim - rootSim

            # check maxgain and prune/leave out if gain - gamma < 0
            if tempGain > maxGain and tempGain - self.gamma > 0:
                maxGain = tempGain
                splitIdx = i
        return splitIdx, maxGain

    def makeLeaf(self, node, residuals, hessians, idx):
        node.isLeaf = True
        node.output = sum(residuals) / (sum(hessians) + self.lam)
        node.residuals, node.hessians, node.X_indices = residuals, hessians, idx
        node.leftNode, node.rightNode = None, None

    # Values are 0 or 1, predictions are 0-1 probabilities
    def buildBranch(self, node: Node, idx: list, currdepth: int, prevNode=None):
        """
        Inputs
        -------
        node: Base Node to build branch off of
        idx: List of remaining row indices. With each iteration, divide/filter out indices depending on gain and split
        currdepth: Current depth of the tree
        """

        splitFeat, bestSplit, maxGains = 0, 0, 0
        foundSplit = False
        residuals, hessians = self.calcResHess(idx)
        if currdepth >= self.maxDepth:
            self.makeLeaf(node, residuals, hessians, idx)
            return

        X_sorted, X_indices = self.sort_X(idx)

        for feature in self.features:
            # Sort indices in terms of sorted X_indices
            res = list(map(lambda i: self.residuals[i], X_indices[feature]))
            hess = list(map(lambda i: self.hessians[i], X_indices[feature]))

            cover = sum(hess) - self.lam

            # If we reach max depth or cover < minimum allowed (prune)continue
            if cover <= self.min_child_weight:
                continue

            # select the feature and split with the maximum gain
            split, gain = self.findSplit(res, hess)
            if gain > maxGains:
                splitFeat = feature
                bestSplit = X_indices[feature].index[split]
                maxGains = gain
                node.residuals = res
                node.hessians = hess
                foundSplit = True

        # base case, turn into a leaf, store output
        if not foundSplit:
            self.makeLeaf(node, residuals, hessians, idx)
            return

        node.thresh = X_sorted[splitFeat][bestSplit]
        node.feature = splitFeat
        node.prob = self.probs[bestSplit]
        node.X_indices = X_indices[splitFeat]

        if prevNode and node.thresh == prevNode.thresh and node.feature == prevNode.feature:
            self.makeLeaf(node, residuals, hessians, idx)
            return

        print(
            f"{'   ' * currdepth}{currdepth}: Splitting on Feature {splitFeat}\n",
            f"{'   ' * currdepth} Best Split ({bestSplit}), span {len(X_indices[splitFeat])}\n",
            f"{'   ' * currdepth}Threshold {node.thresh} and probability {self.probs[bestSplit]}",
        )

        # updates index arrays, and initiate new nodes
        node.leftNode = Node()
        node.rightNode = Node()

        idx0 = sorted(X_indices[splitFeat][:bestSplit])
        idx1 = sorted(X_indices[splitFeat][bestSplit:])

        self.buildBranch(node.leftNode, idx0, currdepth + 1, node)
        self.buildBranch(node.rightNode, idx1, currdepth + 1, node)

    # Recurse down to each leaf, calculate new output value, then update all probs
    def updateProbs(self, currnode: Node, depth: int):
        print(
            f"{'  '*depth}Node Depth: {depth}. Feature: {currnode.feature}. Threshold: {currnode.thresh}\n "
        )

        if currnode.isLeaf and currnode.leftNode is None and currnode.rightNode is None:
            # lambda L2(Lasso) Regulation in denominator
            indices = currnode.X_indices
            assert len(indices) > 0

            denom = sum(np.array(self.hessians)[indices]) + self.lam
            currnode.output = sum(np.array(self.residuals)[indices]) / denom

            probsum = 0

            # recalculate probs, residuals, hessians
            for idx in indices:
                logodd = log(self.probs[idx] / (1 - self.probs[idx]))
                score = logodd + self.lr * currnode.output
                self.probs[idx] = e**score / (1 + e**score)
                self.residuals[idx] = self.y[idx] - self.probs[idx]
                self.hessians[idx] = self.probs[idx] * (1 - self.probs[idx])

                probsum += self.probs[idx]
            currnode.prob = probsum / (len(indices))

            print(
                f"{'  '*depth}LEAF!.\n",
                f"{'  '*depth}  Output: {currnode.output}. \n",
                f"{'  '*depth}  Log(Odds): {logodd}.\n",
                f"{'  '*depth}  Raw Score: {score}.\n",
                f"{'  '*depth}  Prediction: {currnode.prob}.\n-------------------------------------------",
            )
            return

        if currnode.leftNode:
            self.updateProbs(currnode.leftNode, depth + 1)
        if currnode.rightNode:
            self.updateProbs(currnode.rightNode, depth + 1)

    # Recurse to leaf lvl
    # If one of branches has no points, prune
    # If one of branches gain - gamma < 0, prune
    def prune(self, node: Node):
        if not node.leftNode and not node.rightNode:
            return
        if (node.leftNode and not node.rightNode) or (len(node.rightNode.X_indices) == 0):
            self.makeLeaf(
                node, node.leftNode.residuals, node.leftNode.hessians, node.leftNode.X_indices
            )
            return
        if (node.rightNode and not node.leftNode) or (len(node.leftNode.X_indices) == 0):
            self.makeLeaf(
                node, node.rightNode.residuals, node.rightNode.hessians, node.rightNode.X_indices
            )
            return
        self.prune(node.leftNode)
        self.prune(node.rightNode)

    # Recurse to leaf lvl with your features
    # Deterministic at Leaf level
    def predict(self, node: Node, x: pd.Series) -> int:
        if node.isLeaf:
            return 1 if node.prob >= 0.5 else 0
        if node.thresh > x[node.feature]:
            return self.predict(node.rightNode, x)
        return self.predict(node.leftNode, x)

    def buildTree(self):
        """
        Build new tree for each iteration, update probabilities(predictions) after each tree

        Declare Root
        Calculate Residuals and Hessians for each feature
        Build Branch from root
        Then update probabilities

        Create new tree, retain probabilities, residuals, and hessians for each index
        """
        for _ in range(3):
            indices = list(self.X.index)
            _, _ = self.sort_X(indices)
            self.rootNode = Node()
            # self.setResidualsHessians()
            # set index dictionaries
            self.buildBranch(
                self.rootNode,
                idx=indices,
                currdepth=0,
            )
            self.prune(self.rootNode)
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
        print(f"Thresholds: {thresholds}.  Predictions:{probs}.")
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
