# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:59:54 2017

@author: ahmed

ALMOST Same as "Concrete IH calculation.py", ut renamed for easier manipulation in other programs (made some changes later)

This program calculates concrete Instance Hardness (IH) values for each instance of a given dataset.
The instances are classifed by a a set of n classifiers using a leave-one-out method.
IH is the % of classifiers that misclassify the data instance
"""
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Create module for concrete IH calculation so that it can be used by other programs
def calculate_concrete_IH(X, y, full, clfList):
    ndata = X.shape[0]
    numClf = len(clfList) # Num of classifiers
    knn_clf = KNeighborsClassifier(np.floor(np.sqrt(ndata)/2))  # k = sqrt(n)/2
    tree_clf = DecisionTreeClassifier(max_depth=5) 
    nb_clf = GaussianNB()
    lr_clf = LogisticRegression()
    lda_clf = LinearDiscriminantAnalysis()
    qda_clf = QuadraticDiscriminantAnalysis()

    # Matrix that record misclassification
    misclf_matrix = np.zeros((ndata, numClf))

    # If full = True, perform Leave-one-out cross validation for all classifiers
    if full == True:
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):
    
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Classifier 0: kNN
            if 0 in clfList:
                knn_clf.fit(X_train, y_train)
                pred_knn = knn_clf.predict(X_test)
                if pred_knn != y_test:
                    misclf_matrix[test_index[0]][0] = 1
                             
            # Classifier 1: Decision Tree
            if 1 in clfList:             
                tree_clf.fit(X_train, y_train)
                pred_tree = tree_clf.predict(X_test)
                if pred_tree != y_test:
                    misclf_matrix[test_index[0]][1] = 1
                                                  
            # Classifier 2: Naive Bayes                 
            if 2 in clfList:
                nb_clf.fit(X_train, y_train)
                pred_nb = nb_clf.predict(X_test)
                if pred_nb != y_test:
                    misclf_matrix[test_index[0]][2] = 1
                                                               
            # Classifier 3: Logistic Regression
            if 3 in clfList:                 
                lr_clf.fit(X_train, y_train)
                pred_lr = lr_clf.predict(X_test)
                if pred_lr != y_test:
                    misclf_matrix[test_index[0]][3] = 1
                         
            # Classifier 4: LDA
            if 4 in clfList:
                lda_clf.fit(X_train, y_train)
                pred_lda = lda_clf.predict(X_test)
                if pred_lda != y_test:
                    misclf_matrix[test_index[0]][4] = 1
            
            # Classifier 5: QDA
            if 5 in clfList:
                qda_clf.fit(X_train, y_train)
                pred_qda = qda_clf.predict(X_test)
                if pred_qda != y_test:
                    misclf_matrix[test_index[0]][5] = 1
            
            ih_vector = np.zeros(ndata)
            for i in range(ndata):
                ih_vector[i] = sum(misclf_matrix[i,:])/numClf
                         
        return ih_vector, misclf_matrix  
    
    # else perform niter by nfolds (default is 5 by 10) fold cross validation
    else:
        niter = 5   # Num of iterations
        nfolds = 10      
        misclf = np.zeros((ndata, numClf, niter))  # For each data, misclassif by each classifier on each iteration
        
        for randseed in range(niter):
            np.random.seed(randseed)
            kf = KFold(n_splits=nfolds, shuffle=True)
            fold = 0
            for tr_idx, test_idx in kf.split(X):

                X_train, X_test = X[tr_idx], X[test_idx]
                y_train, y_test = y[tr_idx], y[test_idx] 
                
                # Classifier 0: kNN
                if 0 in clfList:
                    knn_clf.fit(X_train, y_train)
                    pred_knn = knn_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_knn[i] != y_test[i]:
                            misclf[test_idx[i]][0][randseed] = 1
                                 
                # Classifier 1: Decision Tree 
                if 1 in clfList:
                    tree_clf.fit(X_train, y_train)
                    pred_tree = tree_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_tree[i] != y_test[i]:
                            misclf[test_idx[i]][1][randseed] = 1
    
                                                  
                # Classifier 2: Naive Bayes  
                if 2 in clfList:
                    nb_clf.fit(X_train, y_train)
                    pred_nb = nb_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_nb[i] != y_test[i]:
                            misclf[test_idx[i]][2][randseed] = 1
            
                # Classifier 3: Logistic Regression 
                if 3 in clfList:                
                    lr_clf.fit(X_train, y_train)
                    pred_lr = lr_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_lr[i] != y_test[i]:
                            misclf[test_idx[i]][3][randseed] = 1
    
                             
                # Classifier 4: LDA
                if 4 in clfList:
                    lda_clf.fit(X_train, y_train)
                    pred_lda = lda_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_lda[i] != y_test[i]:
                            misclf[test_idx[i]][4][randseed] = 1
    
                
                # Classifier 5: QDA
                if 5 in clfList:
                    qda_clf.fit(X_train, y_train)
                    pred_qda = qda_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_qda[i] != y_test[i]:
                            misclf[test_idx[i]][5][randseed] = 1
                
                fold = fold + 1

        ih_vector = np.zeros(ndata)
        for i in range(ndata):
            ih_vector[i] = sum(sum(misclf[i]))/(numClf*niter)   # Avg of matrix with numClf classifiers and niter iterations
        
        return ih_vector, misclf             
            
#    return ih_vector, misclf_matrix

if __name__ == '__main__':
    #'''This portion is specific for stroke dataset
    data = pd.read_csv('Datasets//stroke_normalized.csv')
    X = data.ix[:,1:-1]  # Columns 1 to end - 1. Col 0 is Inst number 
    y = data.ix[:,-1]
    #'''
    X = X.as_matrix()
    y = y.as_matrix()

    #ih, misclf = calculate_concrete_IH(X, y)
    ih, misclf = calculate_concrete_IH(X, y, full=False)
    
