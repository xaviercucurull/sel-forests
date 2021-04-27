"""
Implementation of CART decision tree algorithm.
Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
"""
import tree
import pandas as pd

class CART():
    def __init__(self, max_depth=None, min_size=1, verbose=0):
        self.max_depth = max_depth
        self.min_size = min_size
        self.verbose = verbose
        self.tree = None
    
    def fit(self, x, y):
        """ Fit model with training data and grow a decision tree.

        Args:
            x (DataFrame): training data features
            y (array-like): training data classification
        """
        self.x = x
        self.y = pd.Series(y)
        self.features = self.x.columns.tolist()
        self.tree, self.depth = self._grow_tree(self.x.index.tolist())
        
    def predict(self, x):
        
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
        # Only split if size of node is higher than min_size
        if len(data_idx) <= self.min_size:
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
        
        # Evaluate all features
        # TODO: will need to change for random forests
        for feature in self.features:
            # TODO: apply for numeric and categorical
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
                split_gini = m1/m * gini1 + m2/m * gini2
                                
                #TODO: remove
                if self.verbose > 1:
                    split_point = (X[feature].loc[sorted_indices[i - 1]] + X[feature].loc[sorted_indices[i]]) / 2
                    print('{} < {:.3f} Gini1={:.4f}'.format(feature, split_point, split_gini))

                # Update best split
                if split_gini < best_gini:
                    best_gini = split_gini
                    best_feature = feature
                    best_sp = (X[feature].loc[sorted_indices[i - 1]] + X[feature].loc[sorted_indices[i]]) / 2
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
        # Find predicted class of current node (mode)
        #TODO: check type and use [0] or [0][0]
        #predicted_class = self.y.loc[data_idx].mode()[0][0]
        predicted_class = self.y.loc[data_idx].mode()[0]

        # Node class counts
        class_counts = self.y.loc[data_idx].value_counts()
        
        # Create node
        node = tree.Node(predicted_class=predicted_class, gini=self._gini_index(data_idx, class_counts))        
        
        # Split recursively until no more examples to cover or max_depth is reached
        # TODO: check max_depth and if None
        if self.max_depth is not None:
            if depth < self.max_depth:
                best_gini, best_feature, best_sp, split = self._find_best_split(data_idx)
                
                # If a split has been found keep growing children
                if best_feature is not None: 
                    left_idx = split[0]
                    right_idx = split[1]
                    
                    node.feature = best_feature
                    node.split_point = best_sp
                    
                    # TODO if categorical
                    if self.verbose > 0:
                        print('Best split: {} < {:.3f}'.format(best_feature, best_sp))  
                        #print('left: {}\nright: {}'.format(left_idx, right_idx))  
                        print('left: {} - right: {}'.format(len(left_idx), len(right_idx))) 
                        print('class: {}'.format(predicted_class))
                    
                    # Grow children
                    node.left, _ = self._grow_tree(left_idx, depth + 1)
                    node.right, _ = self._grow_tree(right_idx, depth + 1)
                    
        return node, depth + 1
  
    def _predict(self, row):
        """Predict class for a single sample."""
        node = self.tree
        
        # Recursively evaluate nodes until a leaf is found (no more children)
        while node.left:
            if row[node.feature] < node.split_point:
                node = node.left
            else:
                node = node.right
                
        return node.predicted_class