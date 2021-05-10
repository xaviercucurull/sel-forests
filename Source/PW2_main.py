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

import pandas as pd
import time
import math
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
    cart_acc = test_CART(x, y)
    results_rf = test_random_forest(x, y)
    results_df = test_decision_forest(x, y)
    return results_rf, results_df, cart_acc

def test_medium():
    # Load Mammographic Mass dataset
    print('###########################################################')
    print('################## Test Medium Dataset ####################')
    print('##################  Mammographic Mass  ####################')
    print('###########################################################\n')
    x, y = datasets.load_mammographic_mass()
    cart_acc = test_CART(x, y)
    results_rf = test_random_forest(x, y)
    results_df = test_decision_forest(x, y)
    return results_rf, results_df, cart_acc

def test_large():
    # Load Rice dataset
    print('###########################################################')
    print('#################### Test Large Dataset ###################')
    print('#########################   Rice  #########################')
    print('###########################################################\n')
    x, y = datasets.load_rice()
    cart_acc = test_CART(x, y)
    results_rf, results_df = None, None
    results_rf = test_random_forest(x, y)
    results_df = test_decision_forest(x, y)
    return results_rf, results_df, cart_acc

def split_dataset(x, y):
    """ Split data into 75% train and 25% test

    Args:
        x (DataFrame): data features
        y (array-like): data labels
    """    
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, X_test, y_train, y_test 
    
def test_CART(x, y):
    """ Test CART

    Args:
        x ([type]): [description]
        y ([type]): [description]
    """
    print(' -----------------------')
    print('|        Test CART      |')
    print(' -----------------------\n')
    # Split data into 75% train and 25% test
    X_train, X_test, y_train, y_test = split_dataset(x, y)

    print('Data attributes: {}'.format(len(X_train.keys())))
    print('Training size: {}'.format(len(y_train)))
    print('Test size: {}\n'.format(len(y_test)))
    
    cart = CART(verbose=0)
    f1, acc, fit_t, pred_t = evaluate_model(cart, X_train, y_train, X_test, y_test, return_scores=True)
    
    return acc
  
def test_forest(x, y, NT_values, F_values, forest_classifier):
    # Split data into 75% train and 25% test
    X_train, X_test, y_train, y_test = split_dataset(x, y)
    print('Training size: {}'.format(len(y_train)))
    print('Test size: {}\n'.format(len(y_test)))
    
    nt_vals = []
    f_vals = []
    accuracies = []
    fit_times = []
    pred_times = []
    importances = []
    
    for nt in NT_values:
        for f in F_values:
            # Instantiate Random Forest and evaluate it
            model = forest_classifier(NT=nt, F=f)
            model.fit(X_train, y_train)
            f1, acc, fit_t, pred_t = evaluate_model(model, X_train, y_train, X_test, y_test, print_classificiation_report=False, return_scores=True)

            # Save parameters used and results
            nt_vals.append(nt)
            f_vals.append(f)
            accuracies.append(acc)
            fit_times.append(fit_t)
            pred_times.append(pred_t)
            importances.append(model.feature_importances.to_dict())
            
            # Print results
            print('NT={}  F={}  -> Accuracy: {:.2f}%'.format(nt, f, acc*100))
            print('Feature importance:\n{}\n'.format(model.feature_importances))  
    
    # Save results in a table
    results_table = pd.DataFrame({'NT': nt_vals, 'F': f_vals, 'Accuracy': accuracies,
                  'Fit time': fit_times, 'Prediction times': pred_times, 
                  'Feature importance': importances})
    
    return results_table

def test_random_forest(x, y):
    """ Test random forest with proposed hyperparameters
    """     
    print(' -----------------------')
    print('|  Test Random Forest   |')
    print(' -----------------------\n')

    M = len(x.columns)
    print('Data attributes: {}'.format(M))
    
    # Hyperparameters to test
    NT_values = [1, 10, 25, 50, 75, 100]
    F_values = [1, 3, int(math.log2(M) + 1), int(math.sqrt(M))]
    # Remove duplicates
    F_values = set(F_values)
    
    # Evaluate model with all hyperparameter combinations
    results_table = test_forest(x, y, NT_values, F_values, RandomForest)
    
    return results_table

def test_decision_forest(x, y):
    """ Test decision forest with proposed hyperparameters
    """     
    print(' -----------------------')
    print('| Test Decision Forest  |')
    print(' -----------------------\n')
    
    M = len(x.columns)
    print('Data attributes: {}'.format(M))
    
    # Hyperparameters to test
    NT_values = [1, 10, 25, 50, 75, 100]
    F_values = [int(M/4), int(M/2), int(3*M/4), 'Runif(1/M)']
    # Remove duplicates
    F_values = set(F_values)
    
    # Evaluate model with all hyperparameter combinations
    results_table = test_forest(x, y, NT_values, F_values, DecisionForest)
    
    return results_table
    
def evaluate_model(model, x_train, y_train, x_test, y_test, print_classificiation_report=True, return_scores=False):
    # Train classifier
    print('Training model...')

    time0 = time.time()
    model.fit(x_train, y_train.to_list())

    time_fit = time.time() - time0
    print('Model trained in {:.1f}s'.format(time_fit))
        
    # Predict test data
    time0 = time.time()
    y_pred = model.predict(x_test)
    time_predict = time.time() - time0
    print('Prediction made in {:.1f}s'.format(time_predict))
    
    if print_classificiation_report:
        print('Classification report:')
        print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    print('F1-Score: {:.2f}%'.format(f1*100))
    print('Accuracy: {:.2f}%\n'.format(acc*100))
    
    if return_scores:
        return f1, acc, time_fit, time_predict

if __name__ == '__main__':
    small_results_rf, small_results_df, small_cart_acc = test_small()
    small_results_rf.to_csv(os.path.join('out', 'small_results_rf.csv'), sep=';')
    small_results_df.to_csv(os.path.join('out', 'small_results_df.csv'), sep=';')
    
    medium_results_rf, medium_results_df, medium_cart_acc = test_medium()
    medium_results_rf.to_csv(os.path.join('out', 'medium_results_rf.csv'), sep=';')
    medium_results_df.to_csv(os.path.join('out', 'medium_results_df.csv'), sep=';')
    
    large_results_rf, large_results_df, large_cart_acc = test_large()
    large_results_rf.to_csv(os.path.join('out', 'large_results_rf.csv'), sep=';')
    large_results_df.to_csv(os.path.join('out', 'large_results_df.csv'), sep=';')