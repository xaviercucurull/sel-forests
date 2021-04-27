"""
Supervised and Experiential Learning (SEL)
Master in Artificial Intelligence (UPC)
PW2 - Implementation of a Decision Forest and Random Forest

Author: Xavier Cucurull Salamero <xavier.cucurull@estudiantat.upc.edu>
Course: 2020/2021
"""

import sys
import os
sys.path.append(os.path.abspath(r'..'))

import pandas
import random
import numpy as np
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Data import datasets

from sklearn import tree as sklearnTree
from cart import CART

def evaluate_model(model, x_train, y_train, x_test, y_test):
    # train classifier
    print('    Training model...')

    time0 = time.time()
    model.fit(x_train, y_train.to_list())

    time_fit = time.time() - time0
    print('    Model trained rained in {:.1f}s'.format(time_fit))
        
    # predict test data
    time0 = time.time()
    y_pred = model.predict(x_test)
    time_predict = time.time() - time0
    print('    CART prediction made in {:.1f}s'.format(time_predict))
    #print('    Tree depth: {}\n'.format(model.get_depth()))
    
    print('    Classification report:')
    try:
        print(classification_report(y_test, y_pred.astype(y_test.dtype)))
    except:
        print(classification_report(y_test, y_pred))

# load rice dataset
x, y = datasets.load_rice()

# split data into 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print('Data attributes: {}'.format(len(X_train.keys())))
print('Training size: {}'.format(len(y_train)))
print('Test size: {}\n'.format(len(y_test)))


# Sklearn CART
print('Evaluate Scikit-Learn CART')
model = sklearnTree.DecisionTreeClassifier()
evaluate_model(model, X_train, y_train, X_test, y_test)

# Custom CART
print('Evaluate custom CART')
t0 = time.time()
cart = CART(max_depth=10, verbose=0)
evaluate_model(cart, X_train, y_train, X_test, y_test)

#print(cart.tree)

# # TODO: here as reference
# # https://stackoverflow.com/a/31534542
# df = pd.DataFrame([1,2])
# bootstrap_size = 50
# randlist = pandas.DataFrame(index=np.random.randint(len(df), size=bootstrap_size))
# df.merge(randlist, left_index=True, right_index=True, how='right')

# # feature subspace smaplingn
# features = [0, 1, 2, 3, 4]
# n_features = 3
# selected_features = random.sample(features, n_features)
