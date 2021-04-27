"""
Basic implementation of a binary tree.
Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
"""

class Node:
    def __init__(self, predicted_class, gini, data_idx=None):
        self.predicted_class = predicted_class
        self.gini = gini
        self.data_idx = data_idx
        self.split_point = None
        self.right = None
        self.left = None
            
    def __str__(self):
        return 'Tree - not implemented yet'