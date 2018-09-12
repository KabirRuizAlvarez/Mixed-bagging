# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:07:19 2018

Bias-variance estimation (Kohavi-Wolpert method)

Version 2
Added stratified k-fold for whole dataset

@author: akabir
"""
import os
import numpy as np
import pandas as pd
import random
import csv
import itertools
import winsound
from statistics import mode
import scipy.stats as st

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as mt

import ih_calc
import mix_bag_library as mixlib

N = 3  # No. of sample training sets to be picked from pool of training sets 
numfolds = 3 # No. of folds to divide data
numcvfolds = 3   # For validation sets in mixed bagging

""" Calculate bias (Kong & Dietterich, 1995), (Domingos, 2000) """
def calculate_bias_kong(y, full_pred):
    nsamples = len(y)
    bias_vector = [0 for i in range(nsamples)]
    
    for inst in range(nsamples):
        actual = y[inst] 
        all_pred = full_pred[:, inst]   # All the diff predictions from diff runs
        main_pred = int(mode(all_pred))   # Main prediction ym as defined by [Domingos, 2000] 
        
        bias_vector[inst] = 1 if actual != main_pred else 0
                    
    return bias_vector

""" Calculate variance (Kong & Dietterich, 1995), (Domingos, 2000) """
def calculate_variance_kong(y, full_pred):   # independet of y
    nsamples = len(y)
    variance_vector = [0 for i in range(nsamples)]

    for inst in range(nsamples):
        actual = y[inst]                # y* according to [Domingos, 2000]
        all_pred = full_pred[:, inst]   # All the diff predictions from diff runs
        main_pred = int(mode(all_pred))   # Main pred ym as defined by Domingos 
        
        dev = sum(yi != main_pred for yi in all_pred)   # deviation from main pred
        dev = dev/N
        
        variance_vector[inst] = dev if main_pred == actual else -dev
                    
    return variance_vector

""" Calculate bias (Kohavi & Wolpert, 1996) """
def calculate_bias_kohavi(y, full_pred):
    nsamples = len(y)
    bias_vector = [0 for i in range(nsamples)]
    
    for inst in range(nsamples):
        actual = y[inst]
        if actual == 0:
            Pf = np.array([1, 0])   # Pf = Actual Prob distribution as defined by Kohavi and Wolpert
        else:
            Pf = np.array([0, 1])
        
        all_pred = full_pred[:, inst]   # All the diff predictions from diff runs
        cnt0 = N - sum(all_pred)    # Count 0s and 1s
        cnt1 = sum(all_pred)
        
        Ph = np.array([cnt0/N, cnt1/N])      # # Ph = Predicted Prob distribution as defined by Kohavi and Wolpert

        bias_vector[inst] = 0.5 * np.dot(Pf - Ph, Pf - Ph)   #  Basically 1/2 * (Pf - Ph)^2 
                      
    return bias_vector

""" Calculate variance (Kohavi & Wolpert, 1996) """
def calculate_variance_kohavi(y, full_pred):
    nsamples = len(y)
    variance_vector = [0 for i in range(nsamples)]
    
    for inst in range(nsamples):      
        all_pred = full_pred[:, inst]   # All the diff predictions from diff runs
        cnt0 = N - sum(all_pred)    # Count 0s and 1s
        cnt1 = sum(all_pred)
        
        Ph = np.array([cnt0/N, cnt1/N])      # # Ph = Predicted Prob distribution as defined by Kohavi and Wolpert

        variance_vector[inst] = 0.5 * ( 1 - np.dot(Ph,Ph))   #  Basically 1/2 * (1 - Ph^2)  
            
    return variance_vector

""" Generate N training sets of size m samples from D, the pool of training samples. """
def generate_training_samples(D_X, D_Y, N, fraction = 0.75):
    m = np.int(np.round(len(D_Y)*fraction))    # fraction = fraction of pool size to be used for number instances in each sample
    Tr_X = []
    Tr_Y = []
    
    for i in range(N):
        idx = random.sample(range(len(D_Y)), m)   # Generate unique indices 
        Tr_X.append(D_X[idx])
        Tr_Y.append(D_Y[idx])
    Tr_X = np.asarray(Tr_X)
    Tr_Y = np.asarray(Tr_Y)
    return Tr_X, Tr_Y

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)  # To print the whole array instead of "..." in the middle parts
    
    """ Begin loop for each dataset """  
    for dataset_name in os.listdir('Datasets//temp//'):    # dataset_name now contains the trailing '.csv'
        if(dataset_name.endswith('csv')):
            print("Running",dataset_name[:-4],"dataset ...")
            # Create files to write out the prints
            outfilename = "outputs//bvd_kohavi_with_boost.csv"    
            outfile = open(outfilename,"a")
 
            data = pd.read_csv('Datasets//'+dataset_name)   # construct file name of dataset
            X_mis = data.ix[:,:-1]  # Columns 0 to end - 1
            y = data.ix[:,-1]         # Last column
            
            imp = preprocessing.Imputer(missing_values=999,strategy='most_frequent', axis=0)  # Impute missing values which are coded as'999' in all datasets
            X_unscaled = imp.fit_transform(X_mis)
            X = preprocessing.scale(X_unscaled)   # Scale/normalize the data
            #'''
          #  X = X.as_matrix()
            y = y.as_matrix()
            
            nsamples = len(y)
            
            # Create and evaluate ensemble methods                          
            base_pred = np.zeros((N, nsamples))
            reg_pred = np.zeros((N, nsamples))
            wag_pred = np.zeros((N, nsamples))
            boost_pred = np.zeros((N, nsamples))
            mix_pred = np.zeros((N, nsamples))
            grad_pred = np.zeros((N, nsamples))
            
            # Bias and variance vectors
            base_bias_vector = np.zeros((nsamples))
            reg_bias_vector = np.zeros((nsamples))
            wag_bias_vector = np.zeros((nsamples))
            boost_bias_vector = np.zeros((nsamples))
            mix_bias_vector = np.zeros((nsamples))
            grad_bias_vector = np.zeros((nsamples))
            
            base_variance_vector = np.zeros((nsamples))
            reg_variance_vector = np.zeros((nsamples))
            wag_variance_vector = np.zeros((nsamples))
            boost_variance_vector = np.zeros((nsamples))
            mix_variance_vector = np.zeros((nsamples))
            grad_variance_vector = np.zeros((nsamples))
            
            # Divide dataset into three folds. Then three times, repeat the following:
            # Use D (2/3rd of data), the pool of samples, to test E (1/3rd), the test set
            random.seed(1)   # arbitrary random seed           
            fold = 0                       
            skf = StratifiedKFold(n_splits=numfolds, shuffle=True)

            # Loop for each fold
            for idx_D, idx_E in skf.split(X, y):
                D_X, D_Y = X[idx_D], y[idx_D]
                E_X, E_Y = X[idx_E], y[idx_E] 
                        
                # Generate samples from pool
                Tr_X, Tr_Y = generate_training_samples(D_X, D_Y, N)
                m = len(Tr_Y)    # size of each training sample
                                
                # Base for ensemble methods
                clf2 = DecisionTreeClassifier(max_depth=5)  
                
                for t in range(N):   # For each training set from pool
                    print("Tr no.", t)
                    clf2 = DecisionTreeClassifier(max_depth=5)
                    # calculate ih and create concat set to feed into functions
                    ih, misclf = ih_calc.calculate_concrete_IH(Tr_X[t], Tr_Y[t], full=False, clfList=[0,1,2,3])
                    Tr_Y_reshaped = Tr_Y[t].reshape((len(Tr_Y[t]),1))
                    ih_reshaped = ih.reshape((len(ih),1))
                    trainvalset = np.concatenate((Tr_X[t], Tr_Y_reshaped, ih_reshaped), axis = 1) #trainset includes both X and Y 
                   
                    # Create base elarner and calculate predictions
                    clf_base = DecisionTreeClassifier(max_depth=5)  
                    clf_base.fit(Tr_X[t], Tr_Y[t])
                    base_pred[t, idx_E] = clf_base.predict(E_X)
                    # Create regular bags and calculate predictions
                    reg_bags = mixlib.create_regular_bags(trainvalset) 
                    reg_predictions, reg_predictions_proba = mixlib.make_models(reg_bags, clf2, E_X, E_Y)  # Reg = regular, Proba = probabilities
                    reg_final_pred, reg_final_pred_proba = mixlib.calculate_pred(reg_predictions, reg_predictions_proba)  
                    reg_pred[t, idx_E] = np.squeeze(reg_final_pred)     # Squeeze is like reshape. We just need reg_pred, so thats the only one indexed                                                        
                    
                    # Create Wagging bags and calculate predictions
                    wag_weights_bags = mixlib.create_wagging_weights(len(Tr_X[t]))
                    wag_predictions, wag_predictions_proba = mixlib.make_wagging_models(wag_weights_bags, clf2, Tr_X[t], Tr_Y[t], E_X, E_Y)  # wag = wagging, Proba = probabilities
                    wag_final_pred, wag_final_pred_proba = mixlib.calculate_pred(wag_predictions, wag_predictions_proba)
                    wag_pred[t, idx_E] = np.squeeze(wag_final_pred)     # Squeeze is like reshape. We just need reg_pred, so thats the only one indexed                                                        
                    
                    # Calculate predictions for boosting
                    boost = AdaBoostClassifier(base_estimator=clf2, n_estimators=10)
                    boost.fit(Tr_X[t], Tr_Y[t])
                    boost_pred[t, idx_E] = boost.predict(E_X)
                    
                    # Create Mixed bags and calculate predictions
                    cvfold = 0                       
                    skf2 = StratifiedKFold(n_splits=numcvfolds, shuffle=True)
                                        
                    numhqs = len(mixlib.hqs)
                    numratios = len(mixlib.mix_ratios)
    
                    cv_acc_mx = np.zeros((numcvfolds, numhqs*numratios))  # one for each combo within this cv fold
                    cv_acc_gr = np.zeros((numcvfolds, len(mixlib.hardness_intervals)))
                    
                    cvfold = 0
                    # Loop for cross validation for model selection
                    for tr_idx, val_idx in skf2.split(Tr_X[t], Tr_Y[t]):
                        X_train, X_validate = Tr_X[t][tr_idx], Tr_X[t][val_idx]
                        Y_train, Y_validate = Tr_Y[t][tr_idx], Tr_Y[t][val_idx] 
                        
                        # Calculate IH of training set from external file
                        ih_tr, misclf = ih_calc.calculate_concrete_IH(X_train, Y_train, full=False, clfList=[0,1,2,3])
                                    
                        # Join the X, Y and IH of training set to make one full dataset for easier manipulation
                        Y_train_reshaped = Y_train.reshape((len(Y_train),1))
                        ih_tr_reshaped = ih_tr.reshape((len(ih_tr),1))
                        trainset = np.concatenate((X_train, Y_train_reshaped, ih_tr_reshaped), axis = 1) #trainset includes both X and Y 
                        
                        mixed_bags = list()
                        for hq, mx in itertools.product(mixlib.hqs, mixlib.mix_ratios):                            
                            mixed_bags.append(mixlib.create_mixed_bags(trainset, ih_tr, mix_ratio=mx, hard_quotient=hq)) # Mixed bags with different mixed ratios
                        
                        grad_bags = list()
                        for h_i in mixlib.hardness_intervals:   
                            grad_bags.append(mixlib.create_gradually_mixed_bags(trainset, ih_tr, low_bag_hardness=h_i[0], high_bag_hardness=h_i[1])) # GRadually mixed bags with different intervals
                    
                        mix_predictions = list()
                        mix_predictions_proba = list()
                        mix_final_pred = list()
                        mix_final_pred_proba = list()
                        
                        grad_predictions = list()
                        grad_predictions_proba = list()
                        grad_final_pred = list()
                        grad_final_pred_proba = list()
    
                        # For mixed bagging, evaluate the validation set 
                        for i in range(numhqs*numratios):     # For each mix_ratio for each hq
                            pred, pred_proba = mixlib.make_models(mixed_bags[i], clf2, X_validate, Y_validate)
                            mix_predictions.append(pred)
                            mix_predictions_proba.append(pred_proba)
                            fin_pred, fin_pred_proba = mixlib.calculate_pred(pred, pred_proba) 
                            mix_final_pred.append(fin_pred)
                            mix_final_pred_proba.append(fin_pred_proba)  
                        
                        # For gradually mixed bagging, evaluate the validation set 
                        for i in range(len(mixlib.hardness_intervals)):     
                            gpred, gpred_proba = mixlib.make_models(grad_bags[i], clf2, X_validate, Y_validate)   # gpred = pred for gradually mixed bags 
                            grad_predictions.append(gpred)
                            grad_predictions_proba.append(gpred_proba)
                            gfin_pred, gfin_pred_proba = mixlib.calculate_pred(gpred, gpred_proba) 
                            grad_final_pred.append(gfin_pred)              # gfin_pred = final prediction for gradually mixed bags
                            grad_final_pred_proba.append(gfin_pred_proba)
        
                        # Check all MR-HQ combos for mixed bags
                        for i in range(numhqs*numratios):
                            cv_acc_mx[cvfold][i] = mt.accuracy_score(Y_validate, mix_final_pred[i])
                        # Check all hardness intervals for gradually mixed bags
                        for i in range(len(mixlib.hardness_intervals)):
                            cv_acc_gr[cvfold][i] = mt.accuracy_score(Y_validate, grad_final_pred[i])
                        cvfold = cvfold + 1
                    """ End of loop of cv """  
                 
                    # Use the evaluation matrices to find best parameters for mixed bags
                    val_acc_mx = np.mean(cv_acc_mx, axis=0) # Take mean validation accuracy of each combo by taking mean of each column
                    val_acc_gr = np.mean(cv_acc_gr, axis=0)
                    
                    # Find indices of best combo/interval
                    idx_max_acc_mx = np.argmax(val_acc_mx)    # Index of max accuracy in mixed bags
                    idx_max_acc_gr = np.argmax(val_acc_gr)    # Index of max accuracy in grad bags
                    
                    # Fianlly, create mixed bags (GrpMixBag) and calculate predictions
                    mix_bags = mixlib.create_mixed_bags(trainvalset, ih, mix_ratio=mixlib.mix_ratios[idx_max_acc_mx % numratios], hard_quotient=mixlib.hqs[idx_max_acc_mx // numratios]) # Mixed bags with different mixed ratios
                    mix_predictions, mix_predictions_proba = mixlib.make_models(mix_bags, clf2, E_X, E_Y)  # Reg = regular, Proba = probabilities
                    mix_final_pred, mix_final_pred_proba = mixlib.calculate_pred(mix_predictions, mix_predictions_proba)  
                    mix_pred[t, idx_E] = np.squeeze(mix_final_pred)     # Squeeze is like reshape. We just need reg_pred, so thats the only one indexed                                                        
                    
                    # Create grad bags (IncMixBag) and calculate predictions
                    grad_bags = mixlib.create_gradually_mixed_bags(trainvalset, ih, low_bag_hardness=mixlib.hardness_intervals[idx_max_acc_gr][0], high_bag_hardness=mixlib.hardness_intervals[idx_max_acc_gr][1])
                    grad_predictions, grad_predictions_proba = mixlib.make_models(grad_bags, clf2, E_X, E_Y)  # Reg = regular, Proba = probabilities
                    grad_final_pred, grad_final_pred_proba = mixlib.calculate_pred(grad_predictions, grad_predictions_proba)  
                    grad_pred[t, idx_E] = np.squeeze(grad_final_pred)     # Squeeze is like reshape. We just need reg_pred, so thats the only one indexed                                                        
                
                """ End of training sample from pool """
                
                # Calculate and print bias and variance for all methods
                print(dataset_name[:-4])

                base_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, base_pred[:, idx_E])
                base_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, base_pred[:, idx_E])
   
                reg_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, reg_pred[:, idx_E])
                reg_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, reg_pred[:, idx_E])
                               
                wag_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, wag_pred[:, idx_E])
                wag_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, wag_pred[:, idx_E])
                
                boost_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, boost_pred[:, idx_E])
                boost_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, boost_pred[:, idx_E])
                
                mix_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, mix_pred[:, idx_E])
                mix_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, mix_pred[:, idx_E])
                    
                grad_bias_vector[idx_E] = calculate_bias_kohavi(E_Y, grad_pred[:, idx_E])
                grad_variance_vector[idx_E] = calculate_variance_kohavi(E_Y, grad_pred[:, idx_E])

                fold = fold + 1
                
            """ End of fold """
            print("Base   %.3f   %.3f" % (np.mean(base_bias_vector), np.mean(base_variance_vector)))
            print("Reg    %.3f   %.3f" % (np.mean(reg_bias_vector), np.mean(reg_variance_vector)))
            print("Wag    %.3f   %.3f" % (np.mean(wag_bias_vector), np.mean(wag_variance_vector)))
            print("Boost  %.3f   %.3f" % (np.mean(wag_bias_vector), np.mean(wag_variance_vector)))
            print("Mix    %.3f   %.3f" % (np.mean(mix_bias_vector), np.mean(mix_variance_vector)))
            print("Grad   %.3f   %.3f" % (np.mean(grad_bias_vector), np.mean(grad_variance_vector)))
            
            reg_bias_sigtest = st.wilcoxon(base_bias_vector, reg_bias_vector)
            wag_bias_sigtest = st.wilcoxon(base_bias_vector, wag_bias_vector)
            boost_bias_sigtest = st.wilcoxon(base_bias_vector, boost_bias_vector)
            mix_bias_sigtest = st.wilcoxon(base_bias_vector, mix_bias_vector)
            grad_bias_sigtest = st.wilcoxon(base_bias_vector, grad_bias_vector)
            
            reg_variance_sigtest = st.wilcoxon(base_variance_vector, reg_variance_vector)
            wag_variance_sigtest = st.wilcoxon(base_variance_vector, wag_variance_vector)
            boost_variance_sigtest = st.wilcoxon(base_variance_vector, boost_variance_vector)
            mix_variance_sigtest = st.wilcoxon(base_variance_vector, mix_variance_vector)
            grad_variance_sigtest = st.wilcoxon(base_variance_vector, grad_variance_vector)
            
            print("Bias", reg_bias_sigtest.pvalue, wag_bias_sigtest.pvalue, boost_bias_sigtest.pvalue, mix_bias_sigtest.pvalue, grad_bias_sigtest.pvalue)
            print("Variance", reg_variance_sigtest.pvalue, wag_variance_sigtest.pvalue, boost_variance_sigtest.pvalue, mix_variance_sigtest.pvalue, grad_variance_sigtest.pvalue)
            
            outfile.write("\n%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" % 
                      (dataset_name[:-4], np.mean(base_bias_vector), np.mean(reg_bias_vector), np.mean(wag_bias_vector), np.mean(boost_bias_vector), np.mean(mix_bias_vector), np.mean(grad_bias_vector), 
                       np.mean(base_variance_vector), np.mean(reg_variance_vector), np.mean(wag_variance_vector), np.mean(boost_variance_vector), np.mean(mix_variance_vector), np.mean(grad_variance_vector)))
            winsound.Beep(300, 500)
        """ End of dataset """
    outfile.close()
    winsound.Beep(300, 2500)      # Ending notification sound     
