# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:43:13 2017

Version 6

Create final Mixed bagging models from whole dataset
Find performance of params for the dataset 
Visualize the dataset with MDS/T-SNE
Print to csv file
Visualize decision boundaries of bags

@author: Ahmedul Kabir
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
import sys

import ih_calc   # External file where IH is calculated

import matplotlib 
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import sklearn.metrics as mt

import matplotlib.pyplot as plt


""" Some global values declared so that they can be easily changed """

dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model',
                'credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere', 
                'pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'titanic', 'voting_records']

#dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model']
#dataset_list = ['credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere']
#dataset_list = ['pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'titanic', 'voting_records']

#dataset_list = ['breast_cancer', 'voting_records']

classifier = 'DT'   # Which classifier to use. DT = Decision Tree, NB = Naive Bayes, KNN = k-nearest neighbors

# Mix ratios and hardness quotients (hq) to be tested
#mix_ratios = [[0.5, 0.0, 0.5],[0.4, 0.2, 0.4],[0.3, 0.4, 0.3],[0.2, 0.6, 0.2],[0.1, 0.8, 0.1],[0.0, 1.0, 0.0]]
#hqs = [1.0, 2.0, 3.0] 
mix_ratios=[[0.3,0.4,0.3]]
hqs=[3.0]
# Hardness intervals for gradually mixed bags to be tested
hardness_intervals = [[-4.0, 4.0], [-3.0, 3.0], [-2.5, 2.5], [-2.0, 2.0], [-1.5, 1.5], [-1.0, 1.0], 
                      [-4.0, 3.0], [-3.0, 2.0], [-2.0, 1.0], 
                      [-3.0, 4.0], [-2.0, 3.0], [-1.0, 2.0], 
                      [-4.0, 2.0], [-3.0, 1.0], [-2.0, 0.0], 
                      [-2.0, 4.0], [-1.0, 3.0], [0.0, 2.0],
                      [-0.5, 0.5], [-0.25, 0.25], [0.0, 0.0]]


#numratios = len(mix_ratios)
#numhqs = len(hqs)

synthdata = True  # True whe working with synthetic data
vizbags = False # True when we want to visualize bags
vizdataset = False # True when we want to visualize dataset with dimensionality reduction
vizonebag = False # Visualize one bag at a time
verbose = True # Whether to print info of each fold
vizonebagboundary = True

defnumbags = 10 # default number of bags
numcvruns = 5
numcvfolds = 5    # No. of folds for cross-validation (for model selection)
clfList = [0,1,2,3] #Identify which of the classifiers to use for IH calculation. THe numbers associated with each clf can be found in ih_calc
 
""" Parameter values of make_classification defined globally for ease of modification """
nsamples = 400
nfeatures = 2
ninformative = 2
#ninformative = np.round(nfeatures * 0.6) # 60% of features are informative
nredundant = np.round(nfeatures * 0.1) # 10% of features are redundant.   Remaining 30% are useless (noise) features
#flipys = [0.05, 0.1]  # Experiment with fliping 1%, 5% and 10% data points 
#balances = [[0.5, 0.5], [0.67, 0.33], [0.80, 0.20]] # Experiment with 1:1, 2:1, 4:1 class balance
#randomstate = 5
#classseps = [1.0, 0.6, 0.2] # Experiment with different class balance. Starting with easy problems, then gradually going hard
flipys = [0.01]  # Experiment with fliping 1%, 5% and 10% data points 
balances = [[0.5, 0.5]] # Experiment with 1:1, 2:1, 4:1 class balance
randomstate = 7
classseps = [0.6] # Experiment with different class balance. Starting with easy problems, then gradually going hard

nclusperclass = 2


""" Create 'normal' bags: a random subsample from the dataset with replacement """
#def subsample(dataset, ratio=1.0):
#    sample = list()
#    n_sample = round(len(dataset) * ratio)
#    random.seed(10)
#    while len(sample) < n_sample:
#        index = randrange(len(dataset))
#        sample.append(dataset[index])
#    sample_asarray = np.asarray(sample)
#    return sample_asarray

def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray

""" Create 'hard' bags: a random subsample from the dataset with replacement """
""" ih contains the hardness of each instance in the dataset """
def hard_subsample(dataset, ih, hard_quotient, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # Set weights to be a function of IH by adding a certain percentage of IH (as specified by hard_quotient) to weights
    w = np.ones(len(dataset)) + [hard_quotient*i for i in ih]   # This way the weights are between 1 and 1.25 if hard_quotient = 0.25 (default)
    weights = w / sum(w) # Now sum of all weights equal to one
    
    np.random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample, p = weights)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray

""" Create 'easy' bags: a random subsample from the dataset with replacement """
""" ih contains the hardness of each instance in the dataset. We create 'easiness' as 1 - IH or opposite of hardness """
def easy_subsample(dataset, ih, hard_quotient, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # Set weights to be a function of IH by adding a certain percentage of IH (as specified by hard_quotient) to weights
    easiness = [1 - i for i in ih]
    w = np.ones(len(dataset)) + [hard_quotient*i for i in easiness]   # This way the weights are between 1 and 1.2 if hard_quotient = 0.2 (default)
    weights = w / sum(w) # Now sum of all weights equal to one
    
    np.random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample, p = weights)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray

""" Create wagging subsamples: a random subsample from the dataset with replacement where random noise is added to the weights """
def wagging_subsample(dataset, stddev = 2.0, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    random.seed(10)
     # Set weights to be 1 + random noise with mean at 0 and std. dev. = stddev
    w = np.ones(len(dataset)) + np.random.normal(0.0, stddev, len(dataset)) 
    w[w<0] = 0   # Some weights become less than 0, so they are brought back to just zero since probability cannot be negative 
    weights = w / sum(w) # Now sum of all weights equal to one
    
    np.random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample, p = weights)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray

""" Create regular bootstraps """
def create_regular_bags(trainset, nbags = defnumbags):
    bags = list()
    
    for i in range(nbags):
        bags.append(subsample(trainset))
    
    bags_asarray = np.asarray(bags)
    return bags_asarray

""" Create bootstraps for wagging """
def create_wagging_bags(trainset, nbags = defnumbags):
    bags = list()
    
    for i in range(nbags):
        bags.append(wagging_subsample(trainset, 2.0))
    
    bags_asarray = np.asarray(bags)
    return bags_asarray

""" Create bootstraps (bags) from a given dataset (training set) """
""" Mix ratio: ratio of easy, normal and hard bags (default 0.33, 0.33, 0.33) """
""" hard_quotient: How much oversampling or undersampling to be done """
""" Regular (conventional) bagging is a special case where mix ratio is 0:1:0 """
def create_mixed_bags(trainset, ih, nbags = defnumbags, mix_ratio = [0.33, 0.33, 0.33], hard_quotient = 0.5):
    neasy = round(nbags*mix_ratio[0]) 
    nnormal = round(nbags*mix_ratio[1]) 
    nhard = round(nbags*mix_ratio[2]) 
    
    bags = list()
    # Make easy bags
    for i in range(neasy):     # First nneasy easy bags    
        bags.append(easy_subsample(trainset, ih, hard_quotient))      
    # Make normal bags
    for i in range(neasy, neasy + nnormal):      # The next nnormal normal bags      
        bags.append(subsample(trainset))
    # Make hard bags
    for i in range(neasy + nnormal, neasy + nnormal + nhard):   # The next nhard hard bags
        bags.append(hard_subsample(trainset, ih, hard_quotient))
    
    bags_asarray = np.asarray(bags)
    return bags_asarray


""" Create bootstraps for gradually-changing mixed bagging """
def create_gradually_mixed_bags(trainset, ih, nbags = defnumbags, low_bag_hardness = -1.0, high_bag_hardness = 1.0):
    bag_hardness_values = np.linspace(low_bag_hardness, high_bag_hardness, nbags)   # Divide up the range to find hardness value for each bag
    
    bags = list()
    
    for i in range(nbags):   # nbags = len(hardness_values) 
        if bag_hardness_values[i] < 0:
            bags.append(easy_subsample(trainset, ih, 0 - bag_hardness_values[i]))  # say, bag hardness is -0.3. Thats the same as creating an easy bag with HC = 0.3
        elif bag_hardness_values[i] == 0:
            bags.append(subsample(trainset))
        else:   # if > 1
            bags.append(hard_subsample(trainset, ih, bag_hardness_values[i]))  # say, bag_hardness is 0.4. Thats the same as creating a hard bag with HC = 0.4
    
    bags_asarray = np.asarray(bags)
    return bags_asarray        


""" Make scatterplots of each bag """
def visualize_bags(bags):
    nbags = bags.shape[0]
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    
  #  mycolor = bags[i][:,2]*(ih + 0.2)  
    for i in range(nbags):
        plt.subplot(3,3, i+1)    # subplots are numbered 1->n not 0->n
        
        # We are usually plotting the default 3:3:3 case
        if i < 3:
            bagtype = 'easy'
        elif i < 6:
            bagtype = 'regular'
        else:
            bagtype = 'hard'
            
                   
        plt.title("Bag %d (%s)" % (i+1, bagtype),fontsize = 10)
        # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
        # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)
        y = bags[i][:,2]
        y[y==0] = -1
        ih = bags[i][:,3]
        plt.scatter(bags[i][:,0], bags[i][:,1], c= y*(ih + 0.2), cmap=plt.cm.coolwarm, s=10)
    plt.show()


""" MDS and t-SNE visualization of dataset """
def visualize_dataset(X, y, ih, method = 't-SNE'):
    if (method == 'MDS'):
        transformer = MDS(n_components = 2, max_iter=100, n_init=1)
    else: # t-SNE
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
    plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c = y*(ih + 0.2), cmap=plt.cm.coolwarm)
    plot.set_xticks(())
    plot.set_yticks(())
    
    count=0;
    plt.tight_layout()
    if (method == 'MDS'):
#        title = "MDS of " + dataset_name + " dataset"
        title = "MDS"
        plt.suptitle(title, fontsize = 20)
    else:
#        title = "t-SNE of " + dataset_name + " dataset"
        title = "t-SNE"
        plt.suptitle(title, fontsize = 20)

    plt.show()


""" Make scatterplots of one specified bag """
def visualize_one_bag(bag):
    print("inside here")
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    
  #  mycolor = bags[i][:,2]*(ih + 0.2)  
                       
   # plt.title("Bag %d (%s)" % (i+1, bagtype),fontsize = 10)
    # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
    # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)
    y = bag[:,2]
    y[y==0] = -1
    ih = bag[:,3]
    plt.scatter(bag[:,0], bag[:,1], c= y*(ih + 0.2), cmap=plt.cm.coolwarm, s=80)
    plt.show()
    

""" Make scatterplots with decision boundary of one specified bag """
def visualize_one_bag_decision_boundary(bag):

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    
  #  mycolor = bags[i][:,2]*(ih + 0.2)  
                       
   # plt.title("Bag %d (%s)" % (i+1, bagtype),fontsize = 10)
    # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH 
    # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)
    X = bag[:,[0,1]]
    y = bag[:,2]

    h = .02  # step size in the mesh
    
    # Divide the plot space into grids
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    ax = plt.subplot()
    
    # just plot the dataset first
    cm = plt.cm.coolwarm
 #   cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
  #  ax.set_title("Input data")
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
        clf = DecisionTreeClassifier(max_depth=4)
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        clf = GaussianNB()
    
    clf.fit(X, y)
#    score = clf.score(X_test, y_test)
    
    # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.1)
    
    y[y==0] = -1
    ih = bag[:,3]
    plt.scatter(bag[:,0], bag[:,1], c= y*(ih + 0.3), cmap=plt.cm.coolwarm, s=80)
    plt.show()
    
    
""" Make models from each bag and test them on the testing set """    
def make_models(bags, clf, X_test, Y_test):
    nbags = bags.shape[0]
    nfeatures = bags.shape[2] - 2 # because the last two columns are y and ih
    
    predictions = list()  
    predictions_proba = list()

    for i in range(nbags):
        # Fit and predict (hard as well as soft predictions)
        clf.fit(bags[i][:,0:nfeatures], bags[i][:,nfeatures])
        predicted = clf.predict(X_test)
       # print(predicted.shape)
        predictions.append(predicted)
        pred_proba = clf.predict_proba(X_test)
        predictions_proba.append(pred_proba)

        # Export tree as DOT file for visualization
    #    outfilename = "tree_bag"+ str(i) + ".dot" 
    #    export_graphviz(clf, out_file=outfilename)   # tree_bag1.dot, tree_bag2.dot etc.
        
        predictions_asarray = np.asarray(predictions)
        predictions_proba_asarray = np.asarray(predictions_proba)
    return predictions_asarray, predictions_proba_asarray   


""" Calculate final prediction from the predictions of several bag """
def calculate_pred(predictions, predictions_proba):
    ninst = predictions.shape[1] # No. of instances
    nbags = predictions.shape[0] # No. of bags
    
    # hard (binary) predictions
    final_pred = np.zeros((ninst, 1))
    for j in range(ninst):    # for each instance
        count = 0
        for i in range(nbags):    # find prediction in each bag and do majority vote
            if predictions[i, j] == 1:
                count = count + 1
        final_pred[j] = 1 if count > nbags/2 else 0   # Majority vote

    # soft (with probabilities) predictions
    final_pred_proba = np.zeros((ninst, 1))
    for j in range(ninst):    # for each instance
        s = 0  #sum
        for i in range(nbags):    # find prediction in each bag and take average
            s = s + predictions_proba[i, j , 1]  # Probabilities of class 1
        avg = s/nbags
        final_pred_proba[j] = 1 if avg > 0.5 else 0   # Binarize the predictions

    return final_pred, final_pred_proba
           

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    start = time.time()
    np.set_printoptions(threshold=np.inf)  # To print the whole array instead of "..." in the middle parts
    # Prepare file for output
    outfilename = 'complex_synthdata_' + classifier + "_depth10_" + str(defnumbags) + 'bags_'+str(randomstate)+'.csv'     # e.g., new_params_DT_50bags.csv
    csvfile = open(outfilename, 'w')
    myWriter = csv.writer(csvfile, delimiter=",", lineterminator='\n')
     
    
    """ Begin loop for each dataset """                   
    #for dataset_name in dataset_list:
        
        #print("Dataset: %s" % dataset_name)
    for classsep, balance, flipy in itertools.product(classseps, balances, flipys):  
        if synthdata == True:
            X, y = make_classification(n_samples = nsamples, n_features=nfeatures, n_redundant=nredundant, n_informative=ninformative, flip_y=flipy, 
                                   weights = balance, random_state=randomstate, class_sep=classsep, n_clusters_per_class=nclusperclass)
           
        else: # Work with real data here
            data = pd.read_csv('Datasets//'+dataset_name+'.csv')   # construct file name of dataset
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
            
            #X, y = make_hastie_10_2(1000)
            #y[y==-1] = 0
            
        #    X, y = load_breast_cancer(True)
        
#        cv_acc_mx = np.zeros((numcvruns*numcvfolds, numhqs*numratios))  # one for each combo within this cv fold
#        cv_f1_mx   = np.zeros((numcvruns*numcvfolds, numhqs*numratios))
#        cv_auc_mx = np.zeros((numcvruns*numcvfolds, numhqs*numratios))
                
        cv_acc_gr = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))   # one for each interval within this cv fold
        cv_f1_gr   = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))
        cv_auc_gr = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))
        
        for cvrun in range(numcvruns):
            print ("\nCV Run %d of %d" % (cvrun+1, numcvruns))                 
        
            cvfold = 0
            skf2 = StratifiedKFold(n_splits=numcvfolds, shuffle=True)
            
            # Loop for cross validation for model selection
            for tr_idx, val_idx in skf2.split(X, y):
                X_train, X_validate = X[tr_idx], X[val_idx]
                Y_train, Y_validate = y[tr_idx], y[val_idx] 
                
                print("CV Fold %d of %d" % (cvfold+1, numcvfolds))
                
    #            print("Calculating IH ...")
                # Calculate IH of training set from external file
                ih_tr, misclf = ih_calc.calculate_concrete_IH(X_train, Y_train, full=False, clfList=clfList)
                            
                # Join the X, Y and IH of training set to make one full dataset for easier manipulation
                Y_train_reshaped = Y_train.reshape((len(Y_train),1))
                ih_tr_reshaped = ih_tr.reshape((len(ih_tr),1))
                trainset = np.concatenate((X_train, Y_train_reshaped, ih_tr_reshaped), axis = 1) #trainset includes both X and Y 
                
                mixed_bags = list()
                for hq, mx in itertools.product(hqs, mix_ratios):                           
                    mixed_bags.append(create_mixed_bags(trainset, ih_tr, mix_ratio=mx, hard_quotient=hq)) # Mixed bags with different mixed ratios
#                if vizonebag == True:
#                    visualize_one_bag(mixed_bags[0][0]) 
#                    visualize_one_bag(mixed_bags[0][5]) 
#                    visualize_one_bag(mixed_bags[0][9]) 
#                    sys.exit(0)

                if vizonebagboundary == True:
                    visualize_one_bag_decision_boundary(mixed_bags[0][0]) 
                    visualize_one_bag_decision_boundary(mixed_bags[0][5]) 
                    visualize_one_bag_decision_boundary(mixed_bags[0][9]) 
                    sys.exit(0)

                grad_bags = list()
                for h_i in hardness_intervals:   
                    grad_bags.append(create_gradually_mixed_bags(trainset, ih_tr, low_bag_hardness=h_i[0], high_bag_hardness=h_i[1])) # GRadually mixed bags with different intervals
            
                # Visualize bags when necessary. Code just has mixed bags, but can try others
                if vizbags == True:
                    visualize_bags(mixed_bags[2*numratios])   # MR = 3:3:3, HQ = 0.75
        
    #               print("Building models ...")
    
                # Define classifier models
                if classifier == 'DT':
                    clf = DecisionTreeClassifier(max_depth=5, random_state=1)
                elif classifier == 'KNN':
                    clf = KNeighborsClassifier(np.floor(np.sqrt(nsamples)/2))  # k = sqrt(n)/2
                else:   # NB
                    clf = GaussianNB()    
            
                mix_predictions = list()
                mix_predictions_proba = list()
                mix_final_pred = list()
                mix_final_pred_proba = list()
                
                grad_predictions = list()
                grad_predictions_proba = list()
                grad_final_pred = list()
                grad_final_pred_proba = list()
            
#                # For mixed bagging, evaluate the validation set 
#                for i in range(numhqs*numratios):     # For each mix_ratio for each hq
#                    pred, pred_proba = make_models(mixed_bags[i], clf, X_validate, Y_validate)
#                    mix_predictions.append(pred)
#                    mix_predictions_proba.append(pred_proba)
#                    fin_pred, fin_pred_proba = calculate_pred(pred, pred_proba) 
#                    mix_final_pred.append(fin_pred)
#                    mix_final_pred_proba.append(fin_pred_proba)
#                
                # For gradually mixed bagging, evaluate the validation set 
                for i in range(len(hardness_intervals)):     
                    gpred, gpred_proba = make_models(grad_bags[i], clf, X_validate, Y_validate)   # gpred = pred for gradually mixed bags 
                    grad_predictions.append(gpred)
                    grad_predictions_proba.append(gpred_proba)
                    gfin_pred, gfin_pred_proba = calculate_pred(gpred, gpred_proba) 
                    grad_final_pred.append(gfin_pred)              # gfin_pred = final prediction for gradually mixed bags
                    grad_final_pred_proba.append(gfin_pred_proba)
    
    #            print("Evaluating mixed bagging models ...")
    
                # Check all MR-HQ combos for mixed bags
#                for i in range(numhqs*numratios):
#                    cv_acc_mx[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, mix_final_pred_proba[i])
#                    cv_f1_mx[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, mix_final_pred_proba[i], average='weighted')
#                    cv_auc_mx[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, mix_final_pred_proba[i])
#                
#                # Check all hardness intervals for gradually mixed bags
                for i in range(len(hardness_intervals)):
                    cv_acc_gr[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, grad_final_pred_proba[i])
                    cv_f1_gr[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, grad_final_pred_proba[i], average='weighted')
                    cv_auc_gr[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, grad_final_pred_proba[i])
                
                cvfold = cvfold + 1
        """ End of fold for cross validation for model selection """
        """ We now make models for each method and evaluate on test set """
                
                 
        # Use the evaluation matrices to find best parameters for mixed bags
#        val_acc_mx = np.mean(cv_acc_mx, axis=0) # Take mean validation accuracy of each combo by taking mean of each column
#        val_f1_mx = np.mean(cv_f1_mx, axis=0)   # val = validation
#        val_auc_mx = np.mean(cv_auc_mx, axis=0)
        
        val_acc_gr = np.mean(cv_acc_gr, axis=0)
        val_f1_gr = np.mean(cv_f1_gr, axis=0)
        val_auc_gr = np.mean(cv_auc_gr, axis=0)
        
        # Find indices of best combo/interval
#        idx_max_acc_mx = np.argmax(val_acc_mx)    # Index of max accuracy in mixed bags
#        idx_max_f1_mx = np.argmax(val_f1_mx)
#        idx_max_auc_mx = np.argmax(val_auc_mx)

        idx_max_acc_gr = np.argmax(val_acc_gr)    # Index of max accuracy in grad bags
        idx_max_f1_gr = np.argmax(val_f1_gr)
        idx_max_auc_gr = np.argmax(val_auc_gr)
    
        # Calculate IH of whole data (training + vaidation) from external file
        ih, misclf = ih_calc.calculate_concrete_IH(X, y, full=False, clfList=clfList)
        
        # Visualize when needed
        if vizdataset==True:
            visualize_dataset(X, y, ih, 'MDS')
            visualize_dataset(X, y, ih, 't-SNE')
                
        # Join the X, Y and IH of training+validation set to make one full dataset for easier manipulation
        y_reshaped = y.reshape((len(y),1))
        ih_reshaped = ih.reshape((len(ih),1))
        fulldataset = np.concatenate((X, y_reshaped, ih_reshaped), axis = 1) #trainset includes both X and Y 

    # Create bags for everyone. For mixed bags, use the combo/interval we found to be best according to validation set
        # Regular bagging 
    #    reg_bags = create_regular_bags(fulldataset) 
        
        # Wagging
    #    wag_bags = create_wagging_bags(fulldataset) 
        
        # Mixed bags
        # idx_max_xxx_mx now has index of best mixture in terms of acc, f1, auc. But idx contains info of both HQ and mix_ratio. 
        # Example: numratios = 10, numhqs = 3, so idx= 24 actually means hq = 24/10 = 2 and mix_ratio = 24 % 10 = 4
#        mixed_bags_acc = create_mixed_bags(fulldataset, ih, mix_ratio=mix_ratios[idx_max_acc_mx % numratios], hard_quotient=hqs[idx_max_acc_mx // numratios]) # Mixed bags with different mixed ratios
#        mixed_bags_f1 = create_mixed_bags(fulldataset, ih, mix_ratio=mix_ratios[idx_max_f1_mx % numratios], hard_quotient=hqs[idx_max_f1_mx // numratios]) # Mixed bags with different mixed ratios
#        mixed_bags_auc = create_mixed_bags(fulldataset, ih, mix_ratio=mix_ratios[idx_max_auc_mx % numratios], hard_quotient=hqs[idx_max_auc_mx // numratios]) # Mixed bags with different mixed ratios
#        
       
        # Grad bags                    
        grad_bags_acc = create_gradually_mixed_bags(fulldataset, ih, low_bag_hardness=hardness_intervals[idx_max_acc_gr][0], high_bag_hardness=hardness_intervals[idx_max_acc_gr][1])
        grad_bags_f1 = create_gradually_mixed_bags(fulldataset, ih, low_bag_hardness=hardness_intervals[idx_max_f1_gr][0], high_bag_hardness=hardness_intervals[idx_max_f1_gr][1])
        grad_bags_auc = create_gradually_mixed_bags(fulldataset, ih, low_bag_hardness=hardness_intervals[idx_max_auc_gr][0], high_bag_hardness=hardness_intervals[idx_max_auc_gr][1])
        
         
        # Visualize bags when necessary. Code just has mixed bags, but can try others
        if vizbags == True:
            visualize_bags(mixed_bags[2*numratios])   # MR = 3:3:3, HQ = 0.75
           
        end = time.time()
       
        # Calculate time. MIx bag and grad bag do the same thing. The time only depends on how many params to check
        # So we divide the total time based on the ratio of param sets to check
#        total_time = end - start
#        mix_time = total_time * (numratios*numhqs) / ((numratios*numhqs) + len(hardness_intervals))
#        grad_time = total_time * len(hardness_intervals) / ((numratios*numhqs) + len(hardness_intervals))
        
        #Print the avg of all the metrics
        print("\nFinal results:\n")
        if synthdata == True:
            print("Inst, Features:", nsamples,",", nfeatures)
            print("Class balance:", balance)
            print("Class sep:", classsep)
            print("Rand state:", randomstate)

        else:  # Real data
            print("Dataset: " + dataset_name)
            print("Dataset with", X.shape[0], "instances and", X.shape[1], "predictive features")
        
#        print("\nTime for Mixed bags: %.2f" % mix_time, "s")
#        print("Time for Grad bags: %.2f" % grad_time, "s")
#        
#        print("\nMixed bag accuracies", np.round(val_acc_mx,3))
#        print("Mixed bag F1", np.round(val_f1_mx,3))
#        print("Mixed bag AUC", np.round(val_auc_mx,3))
#        if verbose == True:    # Print when required
#            print("\nBest mixture (accuracy):", mix_ratios[idx_max_acc_mx % numratios], "   with HQ =", hqs[idx_max_acc_mx // numratios])    # integer division 
#            print("Best mixture (F-1):", mix_ratios[idx_max_f1_mx % numratios], "   with HQ =", hqs[idx_max_f1_mx // numratios])     
#            print("Best mixture (AUC):", mix_ratios[idx_max_auc_mx % numratios], "   with HQ =", hqs[idx_max_auc_mx // numratios])
 
        print("\nGrad bag accuracies", np.round(val_acc_gr,3))
        print("Grad bag F1", np.round(val_f1_gr,3))
        print("Grad bag AUC", np.round(val_auc_gr,3))     
        if verbose == True:
            print("\nBest interval (accuracy):", hardness_intervals[idx_max_acc_gr])
            print("Best interval (F-1):", hardness_intervals[idx_max_f1_gr])     
            print("Best interval (AUC):", hardness_intervals[idx_max_auc_gr])
            
        """ Now print everything in the CSV file """
       
#        myWriter.writerows([np.round(val_acc_mx,3)])
#        myWriter.writerows([np.round(val_f1_mx,3)])
#        myWriter.writerows([np.round(val_auc_mx,3)])
        print(classsep, balance, flipy)
        myWriter.writerow([classsep, balance, flipy])
        myWriter.writerows([hardness_intervals])
        myWriter.writerows([np.round(val_acc_gr,3)])
        myWriter.writerows([np.round(val_f1_gr,3)])
        myWriter.writerows([np.round(val_auc_gr,3)])
        myWriter.writerow([])
        
        winsound.Beep(300, 500)
    """ End of loop for all dataset """   
    csvfile.close()

    winsound.Beep(300, 2500)      # Ending notification sound