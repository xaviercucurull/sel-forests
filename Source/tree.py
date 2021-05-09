"""
Basic implementation of a binary tree.

Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
"""

class Node:
    """ Implementation of a binary tree to use with CART.
    """
    def __init__(self, predicted_class, gini, data_idx=None):
        self.predicted_class = predicted_class
        self.gini = gini
        self.data_idx = data_idx
        self.split_point = None
        self.right = None
        self.left = None
            
    def __str__(self):
        str = []
        self._str_aux(self, s=str)
        return '\n'.join(str)
    
    def _str_aux(self, node, depth=0, s=[]):
        # If not a terminal node
        if node.left:   
            # If feature is categorical
            if type(node.split_point) == set:
                s.append('{}[{} ∈ {}]'.format(depth*' ', node.feature, node.split_point))
            # If feature is numerical
            else:
                s.append('{}[{} < {:.3f}]'.format(depth*' ', node.feature, node.split_point))
            
            # Explore children
            self._str_aux(node.left, depth+1, s)
            self._str_aux(node.right, depth+1, s)
            
        # Terminal node (leaf)
        else:
            s.append('{}[{}]'.format(depth*' ', node.predicted_class))
        