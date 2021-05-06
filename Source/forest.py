"""
Implementation of a Random Forest [Breiman, 2001] and Decision Forest [Ho, 1998].
The base classifier is CART.

Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
"""
from cart import CART

from collections import Counter
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)


class DecisionForest():
    """ Implementation of a Decision Forest ensemble classifier (Ho. 1998)
    
    Each tree is grown using a random subspace (selection) of features, 
    which is the same for all the node splits.
    
    Each training set for each tree is the same original training set
    """
    def __init__(self, NT=20, F=2, verbose=0):
        self.n_trees = NT
        self.n_features = F
        self.verbose = verbose
        
        # TODO: feature importances

    def _get_subspace(self):
        """ Get a random feature subspace.

        Returns: x, y of the subsampled dataset
        """
        # Select f random features
        features = self.x.columns.tolist()
        if self.n_features == 'Runif(1/M)':
            M = len(self.x.columns)
            f =  np.random.randint(1, M+1)
        else:
            f = self.n_features
            
        selected_features = random.sample(features, f)
        
        x = self.x[selected_features]
        y = self.y
        
        # Bootstrap sample
        # x2 = pd.DataFrame(x2.values[indices], columns=x2.columns)
        # y2 = pd.Series(self.y).values[indices]
                
        return x, y  
    
    def fit(self, x, y):
        """ Fit model with training data and create an ensemble of trees.

        Args:
            x (DataFrame): training data features
            y (array-like): training data classification
        """    
        self.x = x
        self.y = y
        
        # Create n_trees number of classifiers
        self.classifiers = []
        for i in range(self.n_trees):
            tree = CART()
            sample_x, sample_y = self._get_subspace()
            
            # Debug information
            if self.verbose > 0:
                print('[DecisionForest] Fitting tree {} with features {}...'.format(i, sample_x.columns.tolist()))
                
            tree.fit(sample_x, sample_y)
            self.classifiers.append(tree)
            
        # Feature importances
        features_count = pd.DataFrame([c.features_count for c in self.classifiers])
        self.feature_importances = features_count.sum()/features_count.sum().sum()
        self.feature_importances.sort_values(ascending=False, inplace=True)

    def predict(self, x):
        """ Use the ensemble of classifier to predict the class of the given examples.

        Args:
            x (DataFrame): data features to predict

        Returns:
            list: predicted classes
        """
        
        assert len(self.classifiers), 'Model not trained, call self.fit(x, y) first!'
        pred_list = np.array([c.predict(x) for c in self.classifiers] )
        
        # Vote the majority class to obtain final prediction 
        pred = [Counter(p).most_common(1)[0][0] for p in pred_list.T]

        return pred
    

class RandomForest():
    """Implementatino of a Random Forest ensemble classifier (Breiman. 2001)
    
    Each tree uses a random subspace (selection) of features to split on at each node,
    and the training set for each tree is sampled (bootstrapping) from the original dataset.
    
    Each training set for each tree is a bootstrapped sampling of the original training set.
    """
    def __init__(self, NT=5, F=3, verbose=0):
        self.n_trees = NT
        self.n_features = F
        self.verbose = verbose
        # TODO: feature importances
        
    def _bootstrap_sample(self):
        """ Bootstrap a dataset sample.
        
        Returns: x, y of the subsampled dataset
        """
        # Bootstrap sample
        indices = np.random.randint(len(self.x), size=len(self.x))
        x = pd.DataFrame(self.x.values[indices], columns=self.x.columns)
        y = pd.Series(self.y).values[indices]
        
        return x, y        
        
    def fit(self, x, y):
        """ Fit model with training data and create an ensemble of trees.

        Args:
            x (DataFrame): training data features
            y (array-like): training data classification
        """    
        self.x = x
        self.y = y
        
        # Create n_trees number of classifiers
        self.classifiers = []
        for i in range(self.n_trees):
            # Debug information
            if self.verbose > 0:
                print('[DecisionForest] Fitting tree {}...'.format(i))
                
            tree = CART(F=self.n_features)
            sample_x, sample_y = self._bootstrap_sample()
            tree.fit(sample_x, sample_y)
            self.classifiers.append(tree)
            
        self.features_count = pd.DataFrame([c.features_count for c in self.classifiers])
        
        # Feature importances
        features_count = pd.DataFrame([c.features_count for c in self.classifiers])
        self.feature_importances = features_count.sum()/features_count.sum().sum()
        self.feature_importances.sort_values(ascending=False, inplace=True)
        
    def predict(self, x):
        """ Use the ensemble of classifier to predict the class of the given examples.

        Args:
            x (DataFrame): data features to predict

        Returns:
            list: predicted classes
        """
        
        assert len(self.classifiers), 'Model not trained, call self.fit(x, y) first!'
        pred_list = np.array([c.predict(x) for c in self.classifiers] )
        
        # Vote the majority class to obtain final prediction 
        pred = [Counter(p).most_common(1)[0][0] for p in pred_list.T]

        return pred
    
    
    
        
# TODO: remove    
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(r'..'))
    from Data import datasets

    x, y = datasets.load_mammographic_mass()
    
    RF = RandomForest(10, 3)
    RF.fit(x, y)
    rf_preds = RF.predict(x)
    
    DF = DecisionForest(10, int(len(x.columns)/2))
    DF.fit(x, y)
    df_preds = DF.predict(x)
    
