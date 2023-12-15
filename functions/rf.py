import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import check_accuracy

SPLITS_TO_TEST = np.arange(0, 255) + 0.5


def one_hot(data, uniques=None):
    if uniques is None:
        uniques = np.unique(data)
    _data = np.squeeze(data)[:, np.newaxis] # Enforce as column vector
    _labels = np.squeeze(uniques)[np.newaxis] # Enforce as row vector
    return _data == _labels  


def entropy(freqs, eps=1e-10):
    _sums = np.sum(freqs, axis=-1, keepdims=True)
    p = np.array(freqs) / np.maximum(_sums, eps)
    return -1. * np.sum(
        p * np.log2(np.maximum(p, eps)),
        axis = -1
    )


def conditional_entropy(freqs):
    shares = np.sum(freqs, axis=-1, keepdims=True)
    _sums = np.sum(
        shares, 
        axis = -min(2, np.ndim(shares)),
        keepdims=True
    )
    shares = shares / _sums
    entropies = entropy(freqs)
    if np.ndim(shares) > 2:
        cond_entr = [
            entr.dot(s)
            for entr, s in zip(entropies, shares)
        ]
    else:
        cond_entr = np.dot(entropies, shares)
    return np.squeeze(cond_entr)


class TreeClassifier():

    def __init__(self, 
                 classes: np.ndarray,
                 max_depth: int = 1,
                 min_samples_per_leaf: int = 1):
        self.classes = classes
        self.max_depth = max_depth
        self.min_samples_per_leaf = min_samples_per_leaf

    def get_split_result(self, x_vec, boundary, y_matrix):
        test = x_vec <= boundary
        return np.array([
            y_matrix[test].sum(0),
            y_matrix[~test].sum(0)
        ])
        
    def single_feature_best_split(self, 
                                  x_vec, 
                                  y_matrix,
                                  splits = SPLITS_TO_TEST,
                                  return_entropies: bool = False,
                                  **kwargs):
        _splits = np.array([
            self.get_split_result(x_vec, s, y_matrix)
            for s in splits
        ])
        conditional_entropies = conditional_entropy(_splits)
        argmin = np.argmin(conditional_entropies)
        best = splits[argmin]
        if return_entropies:
            return argmin, best, conditional_entropies
        else:
            return argmin, best
        
    def all_features_best_split(self, 
                                X, 
                                y_matrix,
                                splits = SPLITS_TO_TEST,
                                return_entropies: bool = False,
                                **kwargs):
        d = X.shape[1]
        argmins = np.inf * np.ones(d)
        entropies = -1. * np.ones((d, len(splits)))
        for i in range(d):
            argmins[i], _, entropies[i] = self.single_feature_best_split(
                x_vec = X[:, i],
                y_matrix = y_matrix,
                splits = splits,
                return_entropies = True
            )
        argmins = argmins.astype(int)
        min_entropies = entropies[np.arange(d), argmins]
        feature_to_use = np.argmin(min_entropies)
        boundary_to_use = splits[argmins[feature_to_use]]
        if return_entropies:
            return feature_to_use, boundary_to_use, entropies
        else:
            return feature_to_use, boundary_to_use

    def fit(self, X, y_matrix, **kwargs):
        self.branches = None
        self.frequencies = y_matrix.sum(0)
        if self.max_depth > 0:
            n = X.shape[0]
            if n > (threshold := self.min_samples_per_leaf):
                self.feature, self.boundary = self.all_features_best_split(
                    X = X,
                    y_matrix = y_matrix,
                    **kwargs
                )
                test = X[:, self.feature] <= self.boundary
                sufficient = lambda arr: arr.sum() >= threshold
                if sufficient(test) and sufficient(~test):
                    self.branches = []
                    for subset in [~test, test]:
                        branch = TreeClassifier(
                            classes = self.classes,
                            max_depth = self.max_depth - 1,
                            min_samples_per_leaf = self.min_samples_per_leaf
                        )
                        branch.fit(X[subset], y_matrix[subset], **kwargs)
                        self.branches.append(branch)

    def predict_proba(self, X, **kwargs):
        if np.ndim(X) == 1:
            X = X[np.newaxis] # Enforce row-vector format
        n = X.shape[0]
        if self.branches is None:
            freqs = self.frequencies / self.frequencies.sum()
            probs = np.array([freqs for _ in range(n)])
        else:
            probs = -1. * np.ones((n, len(self.classes)))
            above_threshold = X[:, self.feature] <= self.boundary
            for case, branch in enumerate(self.branches):
                select = above_threshold == case
                if sum(select) > 0:
                    probs[select] = branch.predict_proba(X[select], **kwargs)
            assert np.all(probs >= 0), f"predict_proba failed!"
        return probs
    
    def predict(self, X, **kwargs):
        probs = self.predict_proba(X, **kwargs)
        return self.classes[np.argmax(probs, axis=-1).astype(int)]


class RandomForestClassifier():

    def __init__(self,
                 classes: np.ndarray,
                 n_trees: int = 10,
                 share_obs: float = .2,
                 share_features: float = .2,
                 *args, **kwargs):
        self.classes = classes
        assert 0 < share_obs <= 1
        self.share_obs = share_obs
        assert 0 < share_features <= 1
        self.share_features = share_features
        self.n_trees = n_trees
        self.tree_args = kwargs
        self.trees = [
            TreeClassifier(classes=classes, **kwargs)
            for _ in range(n_trees)
        ]
        self.features = [None for _ in range(n_trees)]

    def bootstrap_dims(self, n, d):        
        sample_size = int(round(max(1, n * self.share_obs)))
        n_features = int(round(max(1, d * self.share_features)))
        return sample_size, n_features

    def fit_single_tree(self,
                        X, y_matrix, 
                        index,
                        sample_size: int = None,
                        n_features: int = None,
                        replace: bool = True,
                        **kwargs):
        n, d = X.shape
        if sample_size is None or n_features is None:
            sample_size, n_features = self.bootstrap_dims(n, d)
        obs_idx = np.random.choice(
            range(n), 
            sample_size, 
            replace=replace
        )
        feature_idx = np.random.choice(
            range(d), 
            n_features, 
            replace=False
        )
        self.features[index] = feature_idx
        self.trees[index].fit(
            X = X[obs_idx][:, feature_idx],
            y_matrix = y_matrix[obs_idx],
            **kwargs
        )

    def fit(self, 
            X, y_matrix, 
            verbose: bool = False,
            **kwargs):
        sample_size, n_features = self.bootstrap_dims(*X.shape)
        for i in range(self.n_trees):
            if verbose:
                print(f"Fitting tree {i+1} / {self.n_trees}", end="\r")
            self.fit_single_tree(
                X, y_matrix, 
                index = i,
                sample_size = sample_size, 
                n_features = n_features,
                **kwargs
            )
        if verbose: print("")
    
    def add_tree(self, *args, **kwargs):
        self.n_trees += 1
        self.features.append(None)
        self.trees.append(TreeClassifier(
            classes = self.classes, 
            **self.tree_args
        ))
        self.fit_single_tree(*args, index=-1, **kwargs)

    def predict_proba(self, X, **kwargs):
        probs = np.array([
            tree.predict_proba(X[:, features], **kwargs)
            for tree, features in zip(self.trees, self.features)
        ])
        sum_probs = np.sum(probs, axis=0)
        avg_probs = sum_probs / np.sum(sum_probs, axis=-1, keepdims=True)
        return avg_probs
    
    def predict(self, X,
                first_past_the_post: bool = False,
                **kwargs):
        if first_past_the_post:
            tree_preds = np.sum([
                one_hot(
                    tree.predict(X[:, features], **kwargs),
                    self.classes
                ) for tree, features in zip(self.trees, self.features)
            ], axis=0)
        else:
            tree_preds = self.predict_proba(X, **kwargs)
        preds = self.classes[np.argmax(tree_preds, axis=-1)]
        return preds


def evaluation_learning(fitting_args,
                        eval_args,
                        hyperparams,
                        tree_increment,
                        n_increments,
                        verbose: bool = True):
    rf = RandomForestClassifier(
        **hyperparams,
        n_trees = tree_increment
    )
    rf.fit(**fitting_args, verbose=verbose)
    learning = np.zeros((n_increments, 2))
    acc_args = {"model": rf, "verbose": False, **eval_args}
    learning[0] = (acc := check_accuracy(**acc_args))
    if verbose:
        print(f"Initial forest accuracy: {acc}")
    total = n_increments * tree_increment
    for i in range(1, n_increments):
        for j in range(tree_increment):
            if verbose:
                n_trees = i * tree_increment + j + 1
                print(f"{n_trees} / {total}", end="\r")
            rf.add_tree(**fitting_args, verbose=False)
        learning[i] = (acc := check_accuracy(**acc_args))
        if verbose:
            print(f"Accuracy with {n_trees} trees: {acc}")
    learning_df = pd.DataFrame(
        learning, 
        columns=["Voting", "Probabilistic"]
    )
    ti = tree_increment
    learning_df.index = np.arange(ti, total + ti, ti)
    learning_df.index.name = "Trees"
    return rf, learning_df


def plot_learning(learning_df):
    for col in learning_df:
        plt.plot(learning_df[col], label=col)
    plt.legend()
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.show()
