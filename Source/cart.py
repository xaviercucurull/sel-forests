"""
Implementation of CART decision tree algorithm.

Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
"""
import tree
import pandas as pd
import numpy as np
import random
from itertools import combinations

random.seed(42)
np.random.seed(42)

class CART():
    """ CART tree classifier.
    
    Args:
        max_depth (int): max depth of the tree
        F (int): number of random features used to split a node
    """
    def __init__(self, max_depth=None, F=None, verbose=0):
        if max_depth:
            self.max_depth = max_depth
        else:
            self.max_depth = np.inf
    
        self.f = F
        self.verbose = verbose
        self.tree = None
    
    def fit(self, x, y):
        """ Fit model with training data and grow a decision tree.

        Args:
            x (DataFrame): training data features
            y (array-like): training data classification
        """
        # If f is bigger than the number of features, set it to None and don' use it
        if self.f is not None:
            if self.f > len(x.columns):
                self.f = None
            
        self.x = x
        self.y = pd.Series(y)
        self.features = self.x.columns.tolist()
        self.features_count = {f: 0 for f in self.features}
        self.tree = self._grow_tree(self.x.index.tolist())
        self.feature_importances = np.array(list(self.features_count.values()))/sum(list(self.features_count.values()))
        
    def predict(self, x):
        """ Use the tree classifier to predict the class of the given examples.

        Args:
            x (DataFrame): data features to predict

        Returns:
            list: predicted classes
        """
        
        assert self.tree is not None, 'Model not trained, call self.fit(x, y) first!'
        pred = [self._predict(row) for i, row in x.iterrows()] 
        
        return pred
    
    def _gini_index(self, data_idx, class_counts):
        """ Calculate the Gini index of a set of instances.

        Args:
            data_idx (list): indices of instances used to calculate the gini index
            class_counts (list): counts of each class
        Returns:
            float: Gini index
        """
        gini = 1.0 - sum((n / sum(class_counts)) ** 2 for n in class_counts)
        return gini
    
    def _get_categorical_splits(self, data, feature):
        """ Calculate the possible subsets of all the possible values of the given feature.
        Args:
            data (array-like): data subset
            feature (string): name of the attribute

        Returns:
            list: containing all the possible values subsets
        """
        values = data[feature].unique()
        if self.verbose > 2:
            print('[CART] \tGet categorical splits for {} different values {}'.format(len(values), values))
            
        splits = []
        for i in range(1, len(values)//2 + 1):
            splits.extend([[set(c), set(values)-set(c)] for c in combinations(values, i)])
                        
        return splits
    
    def _find_best_split(self, data_idx):
        """ Find the best binary split at a node.
        
        Splitting points:
            Continuous attributes: The midpoint between each pair of sorted 
                                   adjacent values is taken as a possible split-point
            Discrete attributes: Examine the partitions resulting from all possible subsets 
                                 of all the possible values of each attribute A.
        Args:
            data_idx (list): indices of all instances to be discriminated at the node

        Returns:
            list: best_gini value (float), best_feature (string), best split point value 
                  and split indices (left, right)
        """
        if self.verbose > 2:
            print('[CART] Find best split...')
            
        # Only split if size of node is higher than 1
        if len(data_idx) <=1:
            return None, None, None, None
        
        X = self.x.loc[data_idx]
        
        # Node class counts
        class_counts = self.y.loc[data_idx].value_counts()

        # Initial best gini is gini of current node
        best_gini = self._gini_index(data_idx, class_counts)
        best_feature = None
        best_sp = None
        split = []
        m = len(data_idx)
        
        # Evaluate features (all features or a random subset of f features)
        if self.f is not None:
            selected_features = random.sample(self.features, self.f)
        else:
            selected_features = self.features
            
        for feature in selected_features:
            # If feature is numerical
            if 'int' in str(type(self.x[feature][0])) or 'float' in str(type(self.x[feature][0])):
                sorted_indices = X.sort_values(feature).index.tolist()
                
                # Initialize number of class counts of each split
                left_class_counts = pd.Series([0] * len(class_counts), index=class_counts.index)
                right_class_counts = class_counts.copy()
                
                # Evaluate each possible split and calculate its gini index
                for i in range(1, len(sorted_indices)):
                    # Skip if two adjacent values are equal
                    # to avoid making an invalid split
                    if X[feature].loc[sorted_indices[i - 1]] == X[feature].loc[sorted_indices[i]]:
                        continue
                
                    # Elements of each split
                    left_idx = sorted_indices[:i]
                    right_idx = sorted_indices[i:]
                    
                    # Update number of class counts of each split
                    c = self.y.loc[sorted_indices[i - 1]]
                    left_class_counts[c] += 1
                    right_class_counts[c] -= 1
                    
                    # Calculate gini of each split
                    gini1 = self._gini_index(left_idx, left_class_counts)
                    gini2 = self._gini_index(right_idx, right_class_counts)
                    
                    # Num elements in current node is m
                    # Num elements in left split is i
                    # Num elements in right split is m-i
                    m1 = i
                    m2 = m - i
                    
                    # Calculate weighted Gini
                    split_gini = (m1 * gini1 + m2 * gini2) / m
                                    
                    # Debug information
                    if self.verbose > 1:
                        split_point = (X[feature].loc[sorted_indices[i - 1]] + X[feature].loc[sorted_indices[i]]) / 2
                        print('[CART] \t{} <= {:.3f} Gini={:.4f}'.format(feature, split_point, split_gini))

                    # Update best split
                    if split_gini < best_gini:
                        best_gini = split_gini
                        best_feature = feature
                        best_sp = (X[feature].loc[sorted_indices[i - 1]] + X[feature].loc[sorted_indices[i]]) / 2
                        split = [left_idx, right_idx]
                        
                    # Stop iterating if best_gini is 0
                    if best_gini == 0:
                        return best_gini, best_feature, best_sp, split
        
            # If feature is categorical
            else:
                splits = self._get_categorical_splits(X, feature)
                
                for s in splits:
                    if self.verbose > 2:
                        print('[CART] Evalute categorical split: {}'.format(s))
                    
                    # Elements of each split
                    left_idx = X.index[X[feature].isin(s[0])]
                    right_idx = X.index[X[feature].isin(s[1])]
                    
                    # Update number of class counts of each split
                    left_class_counts = self.y.loc[left_idx].value_counts()
                    right_class_counts = self.y.loc[right_idx].value_counts()
                    
                    # Calculate gini of each split
                    gini1 = self._gini_index(left_idx, left_class_counts)
                    gini2 = self._gini_index(right_idx, right_class_counts)
                    
                    # Calculate weighted Gini
                    split_gini = (len(left_idx) * gini1 + len(right_idx) * gini2) / m
                                    
                    # Debug information
                    if self.verbose > 1:
                        print('[CART] \t{} ∈ {} Gini={:.4f}'.format(feature, s[0], split_gini))

                    # Update best split
                    if split_gini < best_gini:
                        best_gini = split_gini
                        best_feature = feature
                        best_sp = s[0]
                        split = [left_idx, right_idx]
                        
                    # Stop iterating if best_gini is 0
                    if best_gini == 0:
                        return best_gini, best_feature, best_sp, split
                
        return best_gini, best_feature, best_sp, split
   
    def _grow_tree(self, data_idx, depth=0):
        """ Grow a decision tree by recursively finding the best node split.

        Args:
            data_idx ([type]): [description]
            depth (int, optional): [description]. Defaults to 0.
        """
        if self.verbose > 2:
            print('[CART] Grow tree...')
            
        # Find predicted class of current node (mode)
        predicted_class = self.y.loc[data_idx].mode()[0]

        # Node class counts
        class_counts = self.y.loc[data_idx].value_counts()
                
        # Create node
        node = tree.Node(predicted_class=predicted_class, gini=self._gini_index(data_idx, class_counts))        
        
        # TODO: return if node contains only one class
        if len(class_counts) == 1:
            return node

        # Split recursively until no more examples to cover or max_depth is reached
        if depth < self.max_depth:
            best_gini, best_feature, best_sp, split = self._find_best_split(data_idx)
            
            # If a split has been found keep growing children
            if best_feature is not None: 
                self.features_count[best_feature] += 1
                left_idx = split[0]
                right_idx = split[1]
                
                node.feature = best_feature
                node.split_point = best_sp
                
                # Debug information
                if self.verbose > 0:
                    if type(best_sp) == set:
                        print('[CART] Best split: {} ∈ {}'.format(best_feature, best_sp))
                    else:
                        print('[CART] Best split: {} <= {:.3f}'.format(best_feature, best_sp))  
                    print('[CART] left: {} - right: {}'.format(len(left_idx), len(right_idx))) 
                    print('[CART] class: {}'.format(predicted_class))
                
                # Grow children
                if self.verbose > 2:
                    print('[CART] Grow children...')
                node.left = self._grow_tree(left_idx, depth + 1)
                node.right = self._grow_tree(right_idx, depth + 1)
                    
        return node
  
    def _predict(self, row):
        """Predict class for a single sample."""
        node = self.tree
        
        # Recursively evaluate nodes until a leaf is found (no more children)
        while node.left:
            # If feature is numerical
            if 'int' in str(type(self.x[node.feature][0])) or 'float' in str(type(self.x[node.feature][0])):
                if row[node.feature] <= node.split_point:
                    node = node.left
                else:
                    node = node.right
            # If feature is categorical
            else:
                if row[node.feature] in node.split_point:
                    node = node.left
                else:
                    node = node.right

                
        return node.predicted_class