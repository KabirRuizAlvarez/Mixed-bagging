# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 01:24:29 2017

Visualize the datasets

@author: ahmed
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from random import randrange
import time
import random
import itertools
import winsound
import csv
import os

import ih_calc   # External file where IH is calculated

import matplotlib 
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import sklearn.metrics as mt

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


synthdata = True
classifier = 'KNN'

""" Parameter values of make_classification defined globally for ease of modification """
nsamples = 200
nfeatures = 2
ninformative = 2 #np.round(nfeatures * 0.6) # 60% of features are informative
nredundant = 0 #np.round(nfeatures * 0.1) # 10% of features are redundant.   Remaining 30% are useless (noise) features
flipy = 0.05 
balance = [0.5, 0.5]
randomstate = 5
classsep = 0.6
nclusperclass = 2

""" Some global values declared so that they can be easily changed """

#dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model',
#                'credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere', 
#                'pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'titanic', 'voting_records']

#dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model']
#dataset_list = ['credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere']
#dataset_list = ['pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'titanic', 'voting_records']

#dataset_list = ['hepatitis', 'kr-vs-kp', 'labor', 'sick']
dataset_list = ['balance_scale_LR', 'balance_scale_BL', 'balance_scale_BR']
dataset_list = ['teaching_assistant_LH', 'teaching_assistant_LM','teaching_assistant_MH']
dataset_list = ['wine_c1c2', 'wine_c1c3', 'wine_c2c3']
dataset_list = ['car_evaluation']



""" Visualize the decision boundaries for different datasets """
def visualize_decision_boundary(X, y, ih):
    h = .02  # step size in the mesh
    
    # Divide the plot space into grids
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    ax = plt.subplot()
    
    # just plot the dataset first
    cm = plt.cm.coolwarm
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    ax.set_title("Input data")
    # Plot the training points
#    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    # Perform classification
    if classifier == 'DT':
        clf = DecisionTreeClassifier()
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        clf = GaussianNB()
    clf.fit(X, y)
#    score = clf.score(X_test, y_test)
    
    # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        
#    if hasattr(clf, "decision_function"):
#        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#    else:
#        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
   # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.1)
    
    # Plot also the training points
    yy = y
    yy[yy==0] = -1
    # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
    # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)    
    ax.scatter(X[:, 0], X[:, 1], s = 25, c = yy*(ih + 0.3), vmin = -1, vmax = 1, cmap=plt.cm.coolwarm)

  #  ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
#    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')


""" MDS and t-SNE visualization of dataset """
def visualize_dataset(X, y, ih, method = 'both'):
    if method == 'both':   # Draw both in same plot    
        plt.figure(figsize=(4, 9))
        plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
        
        plt.subplot(2,1,1)
        plt.title("MDS of %s" % dataset_name[:-4])
        mds = MDS(n_components = 2, max_iter=100, n_init=1)
        X_transformed = mds.fit_transform(X)
        y[y==0] = -1
        # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
        # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)    
    
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s = 5, c = y*(ih + 0.1), vmin = -1, vmax = 1, cmap=plt.cm.coolwarm)
        
        plt.subplot(2,1,2)
        plt.title("t-SNE of %s" % dataset_name[:-4])
        tsne = TSNE(n_components = 2)
        X_transformed = tsne.fit_transform(X)
        yy = y
        yy[yy==0] = -1
        # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
        # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)    
    
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s = 5, c = yy*(ih + 0.1), vmin = -1, vmax = 1, cmap=plt.cm.coolwarm)

        figname = 'visualizations//' + dataset_name[:-4] + '.png'
        plt.savefig(figname, bbox_inches='tight')
        return
        
        
    else:  # Plot either MDS or t-SNE
        if (method == 'MDS'):
            transformer = MDS(n_components = 2, max_iter=100, n_init=1)
        elif method == 't-SNE': # t-SNE
            transformer = TSNE(n_components = 2)
        fig, plot = plt.subplots()
        fig.set_size_inches(8, 8)
        plt.prism()
        
        # Blue for low (good) mRS-90, Red for High
        colors = ['blue', 'red']
        
        X_transformed = transformer.fit_transform(X)
        y[y==0] = -1
        # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
        # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)    
    
        plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c = y*(ih + 0.2), vmin = -1, vmax = 1, cmap=plt.cm.coolwarm)
        plot.set_xticks(())
        plot.set_yticks(())
        
        count=0;
        plt.tight_layout()
        if (method == 'MDS'):
            title = "MDS of " + dataset_name + " dataset"
       #     title = "MDS"
            plt.suptitle(title, fontsize = 20)
        else:
            title = "t-SNE of " + dataset_name + " dataset"
          #  title = "t-SNE"
            plt.suptitle(title, fontsize = 20)
             
        plt.show()
        return
  
""" Visualize IH of the dataset and the two classes seprately along with displaying some key info """
def visualize_ih(X,y,ih):
    # Find out which is the majority class so that can be addressed in histogram
    num_c0 = len(np.where(y==0)[0])
    num_c1 = len(np.where(y==1)[0])
    
    #   perc_c0 = round(num_c0 / (num_c0 + num_c1), 2)
  #  perc_c1 = round(num_c1 / (num_c0 + num_c1), 2)
    
  #  ret_list = [] 
    # Find beta distribution parameters 
    print(y)
    beta_params_whole = st.beta.fit(ih)
    beta_params_c0 = st.beta.fit(ih[np.where(y==0)])
    beta_params_c1 = st.beta.fit(ih[np.where(y==1)])
    
    
    plt.figure(figsize=(5, 9))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95, hspace = 0.3)
   
    # Plot IH of whole dataset
    plt.subplot(3,1,1)
    plt.title("Whole dataset - Beta distr. a = %.3f b = %.3f" % (beta_params_whole[0], beta_params_whole[1]), fontsize=10)
    plot_histogram(ih, len(y), 'g')  # All should range from 0 to len(y)
    
    # Then plot majority class followed by minority class
    plt.subplot(3,1,2)
    
    if num_c0 > num_c1:  
        plt.title("Majority class (c0) - Beta distr. a = %.3f   b = %.3f" % (beta_params_c0[0], beta_params_c0[1]), fontsize=10)                    
        #plot_histogram(ih[np.where(y==0)], len(y), 'b')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, len(y))
        plt.hist(ih[np.where(y==0)], range=(0,1), bins=21, color='b')
    else:
        plt.title("Majority class (c1) - Beta distr. a = %.3f   b = %.3f" % (beta_params_c1[0], beta_params_c1[1]), fontsize=10)  
       # plot_histogram(ih[np.where(y==1)], len(y), 'r')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, len(y))
        plt.hist(ih[np.where(y==1)], range=(0,1), bins=21, color='r')
    
    plt.subplot(3,1,3)
    if num_c0 > num_c1:   
        plt.title("Minority class (c1) - Beta distr. a = %.3f   b = %.3f" % (beta_params_c1[0], beta_params_c1[1]), fontsize=10)                  
        #plot_histogram(ih[np.where(y==1)], len(y), 'r')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, len(y))
        plt.hist(ih[np.where(y==1)], range=(0,1), bins=21, color='r')
    else:
        plt.title("Minority class (c0) - Beta distr. a = %.3f   b = %.3f" % (beta_params_c0[0], beta_params_c0[1]), fontsize=10)  
        #plot_histogram(ih[np.where(y==0)], len(y), 'b')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, len(y))
        plt.hist(ih[np.where(y==0)], range=(0,1), bins=21, color='b')

    figname = 'visualizations//' + dataset_name[:-4] + '_IH.png'
    plt.savefig(figname, bbox_inches='tight')        



""" Plot histogram of dataset """
def plot_histogram(ih, ymax, color='k'):
    plt.suptitle("%s\nn = %d,  f = %d,   bal = %s,  avg_IH = %.3f +/- %.3f,  corr at k=%d is %.3f" % 
                 (dataset_name[:-4], X.shape[0], X.shape[1], findClassBalance(y), np.mean(ih), np.std(ih), findCorr(X, y, ih)[0], findCorr(X, y, ih)[1]), fontsize=12)
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, ymax)
    plt.hist(ih, range=(0,1), bins=21, color=color)
    #plt.show()        


""" Find class balance """
def findClassBalance(y):
    num_c0 = len(np.where(y==0)[0])
    num_c1 = len(np.where(y==1)[0])
    
    perc_c0 = round(num_c0 / (num_c0 + num_c1), 2)
    perc_c1 = round(num_c1 / (num_c0 + num_c1), 2)
    
    ret_list = []
        
    if num_c0 > num_c1:             # Put majority class first
        ret_list.append(perc_c0)
        ret_list.append(perc_c1)
    else:
        ret_list.append(perc_c1)
        ret_list.append(perc_c0)
    return ret_list    


""" Find correlation between IH and and avg IH of neighbors """
def findCorr(X, y, ih):
    maxcorr = -1
    maxk = 0
    
    for k in range(1,int(np.round(np.sqrt(len(X))))):
        avgIH_of_nbrs = avg_IH_of_kNearestNeighbors(X, y, ih, k)
        corr = np.corrcoef(ih, avgIH_of_nbrs)[0,1]
        if corr > maxcorr:
            maxcorr = corr
            maxk = k
    
    return maxk, maxcorr


""" Return the avg IH of k nearest neighbors (apart from itself) """
def avg_IH_of_kNearestNeighbors(X, y, ih, k = 5):
    # Build NearestNeighbors object and fit to X
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs = nbrs.fit(X)
    
    dist, idx = nbrs.kneighbors(X, n_neighbors=k+1)   # Since it returns itself as a neighbor, we take k+1 neighbors to get the other k apart from itslef
    
    idx = idx[:, 1:]    # Exclude the instance itself, take the neighbors' indices only

    ih_of_nbrs = ih[idx]

    avgIH_of_nbrs = np.mean(ih_of_nbrs, axis=1)
                              
    return avgIH_of_nbrs




""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
 
    np.set_printoptions(threshold=np.inf)  # To print the whole array instead of "..." in the middle parts
    
    if synthdata == True:
        #  for classsep, balance, flipy in itertools.product(classseps, balances, flipys):  
        X, y = make_classification(n_samples = nsamples, n_features=nfeatures, n_redundant=nredundant, n_informative=ninformative, flip_y=flipy, 
                           weights = balance, random_state=randomstate, class_sep=classsep, n_clusters_per_class=nclusperclass)
   
        ih, misclf = ih_calc.calculate_concrete_IH(X, y, full=False, clfList=[0,1,2,3])    
        visualize_decision_boundary(X, y, ih)
    else:
        
        """ Begin loop for each dataset """                   
      #  for dataset_name in dataset_list:
        for dataset_name in os.listdir('Datasets//added_later//'):    # dataset_name now contains the trailing '.csv'
            if(dataset_name.endswith('csv')):    
                print("Dataset: %s" % dataset_name[:-4])         
               
                data = pd.read_csv('Datasets//added_later//'+dataset_name)   # construct file name of dataset
                X_mis = data.ix[:,:-1]  # Columns 0 to end - 1
                y = data.ix[:,-1]         # Last column
                
                imp = preprocessing.Imputer(missing_values=999,strategy='most_frequent', axis=0)  # Impute missing values which are coded as'999' in all datasets
                X_unscaled = imp.fit_transform(X_mis)
                if classifier != 'NB':
                    X = preprocessing.scale(X_unscaled)   # Scale/normalize the data, but not if we're working with Naive Bayes
                else:
                    X = X_unscaled
                #'''
              #  X = X.as_matrix()
                y = y.as_matrix()
                
                ih, misclf = ih_calc.calculate_concrete_IH(X, y, full=False, clfList=[0,1,2,3])
                
             #   visualize_dataset(X, y, ih, 'both')
               
                visualize_ih(X, y, ih)
         #       visualize_dataset(X, y, ih, 'MDS')
         #       visualize_dataset(X, y, ih, 't-SNE')
                
         