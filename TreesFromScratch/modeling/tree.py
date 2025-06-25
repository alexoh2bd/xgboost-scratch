import numpy as np
import pandas as pd
from math import e, log


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
        self.missing_direction = None


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
        X: pd.DataFrame,
        y: pd.DataFrame,
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

        # Calculate residuals(gaussians) and hessians from probs upon init
        self.probs = probs
        self.residuals = np.array(self.y).ravel() - np.array(self.probs)
        self.hessians = np.array(self.probs) * (1 - np.array(self.probs))

    # Calculate the similarity score for calculating gain
    def simScore(self, resSum: float, hessSum: float) -> float:
        assert ((hessSum) + self.lam) != 0
        return ((resSum) ** 2) / ((hessSum) + self.lam)

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
    def findSplit(self, sorted_indices: list[int], na_indices: list[int], feature: str):
        """
        Parameters
        ---------
        feature: str
        res: Residuals
        hess: Hessians

        Sum the gausssians and hessians once at the beginning, sliding window
        """
        res = self.residuals[sorted_indices]
        hess = self.hessians[sorted_indices]
        total_residuals = np.sum(res)
        total_hessians = np.sum(hess)

        # Initialize sums with minimum child weight
        lRSum = np.sum(res[: self.min_child_weight])
        lHSum = np.sum(hess[: self.min_child_weight])
        rRSum = total_residuals - lRSum
        rHSum = total_hessians - lHSum
        missing_direction = None

        maxGain = 0
        tempGain = 0
        rootSim = self.simScore(rRSum, rHSum)
        splitIdx = self.min_child_weight
        idxrange = len(res)

        # Iterate through Residuals and Hessians to calc Similarity Score and find max Gain
        for i in range(self.min_child_weight, idxrange - self.min_child_weight):
            if i >= len(res) or i >= len(hess):
                break

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
        na_res = self.residuals[na_indices]
        na_hess = self.hessians[na_indices]
        # Calculate gain for missing values in both directions
        if len(na_indices) > 0:
            # Calculate gain if missing values go left
            lRSum_missing = lRSum + np.sum(na_res)
            lHSum_missing = lHSum + np.sum(na_hess)

            gain_missing_left = (
                self.simScore(lRSum_missing, lHSum_missing) + self.simScore(rRSum, rHSum) - rootSim
            )

            # Calculate gain if missing values go right
            rRSum_missing = rRSum + np.sum(na_res)
            rHSum_missing = rHSum + np.sum(na_hess)

            gain_missing_right = (
                self.simScore(lRSum, lHSum) + self.simScore(rRSum_missing, rHSum_missing) - rootSim
            )

            # Store the direction that maximizes gain
            missing_direction = "left" if gain_missing_left > gain_missing_right else "right"

        return splitIdx, maxGain, missing_direction

    # Make Leaf
    def makeLeaf(self, node, residuals, hessians, idx):
        node.isLeaf = True
        node.output = np.sum(residuals) / (np.sum(hessians) + self.lam)
        node.residuals, node.hessians, node.X_indices = residuals, hessians, idx
        node.leftNode, node.rightNode = None, None
        # self.update_probs(node)

    # Values are 0 or 1, predictions are 0-1 probabilities
    def buildBranch(self, node: Node, residuals, hessians, depth):
        """
        Build a branch of the tree recursively

        Parameters:
        -----------
        node: Node
            Current node being processed
        residuals: np.ndarray
            Residuals for current node
        hessians: np.ndarray
            Hessians for current node
        """
        # Get indices for current node
        indices = node.X_indices
        if len(indices) == 0:
            return

        # Calculate cover
        cover = np.sum(hessians[indices]) - self.lam
        if cover <= self.min_child_weight:
            self.makeLeaf(node, residuals[indices], hessians[indices], indices)
            self.updateProbs(node, depth)
            return

        if depth >= self.maxDepth:
            self.makeLeaf(node, residuals[indices], hessians[indices], indices)
            self.updateProbs(node, depth)
            return

        # Initialize best split variables
        best_gain = -np.inf
        best_feature = None
        best_split_idx = None
        best_missing_direction = None
        best_indices = indices

        # Try splitting on each feature
        for feature in self.features:
            # Sort indices for this feature
            feature_values = self.X[feature].iloc[indices]
            na_mask = np.isnan(feature_values)
            na_indices = indices[na_mask]

            sorted_indices = indices[np.argsort(feature_values[~na_mask])]

            # Find best split for this feature
            split_idx, gain, missing_dir = self.findSplit(sorted_indices, na_indices, feature)

            # Update if this is the best split found so far
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split_idx = split_idx
                best_missing_direction = missing_dir
                if best_missing_direction == "right":
                    best_indices = np.concatenate((sorted_indices, na_indices))
                else:
                    best_indices = np.concatenate((na_indices, sorted_indices))

        # If no good split was found, make this a leaf
        if best_gain <= 0:
            self.makeLeaf(node, residuals[indices], hessians[indices], indices)
            self.updateProbs(node, depth)
            return

        # Create split using the best feature and index
        node.feature = best_feature
        node.thresh = self.X[best_feature].iloc[indices[best_split_idx]]
        node.missing_direction = best_missing_direction

        # Create child nodes
        node.leftNode = Node()
        node.rightNode = Node()

        # Split indices
        left_indices = best_indices[self.X[best_feature].iloc[indices] <= node.thresh]
        right_indices = best_indices[self.X[best_feature].iloc[indices] > node.thresh]

        # Build left and right branches
        node.leftNode.X_indices = left_indices
        node.rightNode.X_indices = right_indices
        self.buildBranch(node.leftNode, residuals, hessians, depth + 1)
        self.buildBranch(node.rightNode, residuals, hessians, depth + 1)

    def buildTree(self):
        """
        Build the tree recursively
        """
        # Initialize root node
        self.rootNode = Node()
        self.rootNode.X_indices = np.arange(len(self.X))

        # Build the tree recursively
        self.buildBranch(self.rootNode, self.residuals, self.hessians, depth=0)

    # Recurse down to each leaf, calculate new output value, then update all probs
    def updateProbs(self, currnode: Node, depth: int):
        # lambda L2(Lasso) Regulation in denominator
        indices = currnode.X_indices
        assert len(indices) > 0

        # Calculate output with regularization
        denom = np.sum(np.array(self.hessians)[indices]) + self.lam
        currnode.output = np.sum(np.array(self.residuals)[indices]) / denom

        # Add small epsilon to prevent log(0) and divide by zero
        epsilon = 1e-15
        probsum = 0

        # Update probabilities for each instance
        for idx in indices:
            # Clip probabilities to avoid extreme values
            prob = min(max(self.probs[idx], epsilon), 1 - epsilon)
            logodd = log(prob / (1 - prob))
            # Scale the output to prevent extreme scores
            score = logodd + self.lr * currnode.output
            # Apply sigmoid to get new probability
            new_prob = 1 / (1 + e ** (-score))

            # Update values
            self.probs[idx] = new_prob
            self.residuals[idx] = np.array(self.y).ravel()[idx] - self.probs[idx]
            self.hessians[idx] = self.probs[idx] * (1 - self.probs[idx])

            probsum += self.probs[idx]
        currnode.prob = probsum / (len(indices))
