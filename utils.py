import torch
import numpy as np
from sklearn.tree import _tree


def kmeans(x, k, centroids=None, max_iter=None, epsilon=0.01):
    """
    x: data set of size (n, d) where n is the sample size.
    k: number of clusters
    centroids (optional): initial centroids
    max_iter (optional): maximum number of iterations
    epsilon (optional): error tolerance
    returns
    centroids: centroids found by k-means algorithm
    next_assigns: assignment vector
    mse: mean squared error
    """
    if centroids is None:
        centroids = torch.zeros(k, x.shape[1], device=x.device)
        prev_assigns = torch.randint(0, k, (x.shape[0],), device=x.device)
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)

    distances = torch.cdist(centroids, x) ** 2
    prev_assigns = torch.argmin(distances, dim=0)

    it = 0
    prev_mse = distances[prev_assigns, torch.arange(x.shape[0])].mean()
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)
        distances = torch.cdist(centroids, x) ** 2
        next_assigns = torch.argmin(distances, dim=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
        it += 1
        mse = distances[next_assigns, torch.arange(x.shape[0])].mean()
        error = abs(prev_mse-mse)/prev_mse
        prev_mse = mse
        print("iteration: %d, mse: %.3f" % (it, prev_mse.item()))

        if it == max_iter:
            break
        if error < epsilon:
            break

    return centroids, next_assigns, prev_mse, it


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            left = rules.copy()
            right = rules.copy()
            left.append("(not (%s))" % name)
            right.append("(%s)" % name)
            rules_from_left = recurse(tree_.children_left[node], left)
            rules_from_right = recurse(tree_.children_right[node], right)
            rules = np.concatenate([rules_from_left, rules_from_right])
            return rules
        else:
            precond = " ".join(rules)
            precond = ":precondition (and (pickloc ?above) (stackloc ?below) %s)" % precond
            eff = tree_.value[node][0]
            idx = eff.argmax()
            prob = eff[idx] / eff.sum()
            effect = ":effect (and \n\t\t\t(probabilistic %.3f (eff%d))" % (prob, idx)
            effect += "\n\t\t\t(not (pickloc ?above))"
            effect += "\n\t\t\t(instack ?above)"
            effect += "\n\t\t\t(stackloc ?above)"
            effect += "\n\t\t\t(not (stackloc ?below)))"
            return np.array([[precond, effect]])
    return recurse(0, [])
