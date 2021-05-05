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
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from Data import datasets
from cart import CART
from forest import DecisionForest, RandomForest

def test_small():
    # Load Heart Disease database from CSV
    # https://www.kaggle.com/ronitf/heart-disease-uci
    print('###########################################################')
    print('##################  Test Small Dataset ####################')
    print('##################     Heart Disease   ####################')
    print('###########################################################\n')
    x, y = datasets.load_csv(os.path.join('..','DATA', 'heart.csv'))
    model = test_dataset(x, y)

def test_medium():
    # Load Mammographic Mass dataset
    print('###########################################################')
    print('################## Test Medium Dataset ####################')
    print('##################  Mammographic Mass  ####################')
    print('###########################################################\n')
    x, y = datasets.load_mammographic_mass()
    model = test_dataset(x, y)

def test_large():
    # Load Rice dataset
    print('###########################################################')
    print('#################### Test Large Dataset ###################')
    print('#########################   Rice  #########################')
    print('###########################################################\n')
    x, y = datasets.load_rice()
    model = test_dataset(x, y)

def test_dataset(x, y):
    # Split data into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    print('Data attributes: {}'.format(len(X_train.keys())))
    print('Training size: {}'.format(len(y_train)))
    print('Test size: {}\n'.format(len(y_test)))
    
    cart = CART(verbose=0)
    evaluate_model(cart, X_train, y_train, X_test, y_test)
    
    RF = RandomForest(10, 3)
    evaluate_model(RF, X_train, y_train, X_test, y_test)
    
    DF = DecisionForest(10, int(len(x.columns)/2))
    evaluate_model(DF, X_train, y_train, X_test, y_test)


def evaluate_model(model, x_train, y_train, x_test, y_test):
    # Train classifier
    print('Training model...')

    time0 = time.time()
    model.fit(x_train, y_train.to_list())

    time_fit = time.time() - time0
    print('Model trained rained in {:.1f}s'.format(time_fit))
        
    # Predict test data
    time0 = time.time()
    y_pred = model.predict(x_test)
    time_predict = time.time() - time0
    print('Prediction made in {:.1f}s'.format(time_predict))
    #print('Tree depth: {}\n'.format(model.get_depth()))
    #feat_importances = ['{}:{}%'.format(f, round(i*100)) for i, f in sorted(zip(model.feature_importances, model.features), reverse=True)]
    #print('Feature importances: {}'.format(', '.join(feat_importances)))
    
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    print('F1-Score: {:.2f}%'.format(f1_score(y_test, y_pred, average='weighted')*100))
    print('Accuracy: {:.2f}%\n'.format(accuracy_score(y_test, y_pred)*100))

def plot_feature_importance(features, importances):
    sorted_indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance')
    plt.bar(range(len(features)), np.array(importances)[sorted_indices], align='center')
    plt.xticks(range(len(features)), np.array(features)[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()

test_small()
test_medium()
test_large()



####################################################################################


# Sklearn CART
#from sklearn import tree as sklearnTree
#print('Evaluate Scikit-Learn CART')
#model = sklearnTree.DecisionTreeClassifier()
#evaluate_model(model, X_train, y_train, X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
# Train the mode
forest = RandomForestClassifier()
x, y = datasets.load_rice()
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
forest.fit(X_train, y_train)



# # TODO: here as reference
# # https://stackoverflow.com/a/31534542
# df = pd.DataFrame([1,2])
# bootstrap_size = 50
# randlist = pandas.DataFrame(index=np.random.randint(len(df), size=bootstrap_size))
# df.merge(randlist, left_index=True, right_index=True, how='right')

# # feature subspace sampling
# features = [0, 1, 2, 3, 4]
# n_features = 3
# selected_features = random.sample(features, n_features)

#x, y = datasets.load_lenses()
#evaluate_model(cart, x, y, x, y)


# Create a random subsample from the dataset with replacement
# def subsample(dataset, ratio=1.0):
# 	sample = list()
# 	n_sample = round(len(dataset) * ratio)
# 	while len(sample) < n_sample:
# 		index = randrange(len(dataset))
# 		sample.append(dataset[index])
# 	return sample