# -*- coding: utf-8 -*-
"""
Created on Feb 01 2018

Copy of Mixed bagging Version 8
Created for easier inclusion of functions into other programs

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
import os

import ih_calc   # External file where IH is calculated

import matplotlib 
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import sklearn.metrics as mt

import matplotlib.pyplot as plt

start = time.time()

""" Some global values declared so that they can be easily changed """
#dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model', 'credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere','pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'voting_records']
dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model']
#dataset_list = ['credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere']
#dataset_list = ['pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'voting_records']
#dataset_list = ['breast_cancer', 'heart_disease']
#dataset_list = ['breast-w','heart-statlog']
dataset_list = ['acute_inflammations', 'acute_nephritis', 'hepatitis', 'kr-vs-kp', 'labor', 'sick']
dataset_list = ['balance_scale_LR', 'balance_scale_BL', 'balance_scale_BR']
dataset_list = ['teaching_assistant_LH', 'teaching_assistant_LM','teaching_assistant_MH']
dataset_list = ['titanic']

classifier = 'DT'   # Which classifier to use. DT = Decision Tree, NB = Naive Bayes, KNN = k-nearest neighbors

#bestdepth = [10,2,4,1,1,1,3,3,4,1,2,2,1,1,4]   # Best depth in terms of accuracy (where best accuracy is achieved) for each of the 15 datasets. Precomputed in "Datasets DT experiments.py" 
bestdepth = [7,2,4,1,3,3,3,3,3,4,2,4,4,2,2]   # Best depth in terms AUC (precomputed)
#bestdepth = [2, 3]

synthdata = False  # True whe working with synthetic data
vizbags = False # True when we want to visualize bags
vizdataset = False # True when we want to visualize dataset with dimensionality reduction
verbose = True # Whether to print info of each fold

defnumbags = 10 # default number of bags
numruns = 10 # No. of runs (with diff rand seeds) for k-fold evaluation
numfolds = 5   # No. of folds for test set evaluation
numcvruns = 5  # No. of runs for cross validation
numcvfolds = 5    # No. of folds for cross-validation (for model selection)
numsynthruns = 10 # No. of runs for synthetic dataset experiments
clfList = [0,1,2,3] #Identify which of the classifiers to use for IH calculation. THe numbers associated with each clf can be found in ih_calc
         
# Mix ratios and hardness quotients (hq) to be tested
mix_ratios = [[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0], 
              [0.2, 0.4, 0.4], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2], [0.0, 0.8, 0.2], [0.0, 0.7, 0.3], 
              [0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.0, 1.0, 0.0]]
hqs = [1.0, 2.0, 3.0] 

# Hardness intervals for gradually mixed bags to be tested
hardness_intervals = [[-6.0, 6.0], [-5.0, 5.0], [-4.0, 4.0], [-3.0, 3.0], 
                      [-4.0, 3.0], [-3.0, 2.0], [-2.0, 1.0], 
                      [-3.0, 4.0], [-2.0, 3.0], [-1.0, 2.0], 
                      [-4.0, 2.0], [-3.0, 1.0], [-2.0, 0.0], 
                      [-2.0, 4.0], [-1.0, 3.0], [0.0, 2.0],
                      [-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0]]


numratios = len(mix_ratios)
numhqs = len(hqs)
""" Parameter values of make_classification defined globally for ease of modification """
nsamples = 500
nfeatures = 10
ninformative = np.round(nfeatures * 0.6) # 60% of features are informative
nredundant = np.round(nfeatures * 0.1) # 10% of features are redundant.   Remaining 30% are useless (noise) features
flipy = 0.05 
balance = [0.5, 0.5]
#randomstate = 1
classsep = 1.0
nclusperclass = 2


""" Create 'normal' bags: a random subsample from the dataset with replacement """

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
    w = np.ones(len(dataset)) + [hard_quotient*i for i in ih]   # This way the weights are between 1 and 1.2 if hard_quotient = 0.2 (default)
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
def wagging_subsample(dataset, stddev = 1.0, ratio=1.0):
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
        bags.append(wagging_subsample(trainset, 1.0))
    
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
def create_gradually_mixed_bags(trainset, ih, nbags = defnumbags, low_bag_hardness = 0.5, high_bag_hardness = 1.5):
    bag_hardness_values = np.linspace(low_bag_hardness, high_bag_hardness, nbags)   # Divide up the range to find hardness value for each bag
    
    bags = list()
    
    for i in range(nbags):   # nbags = len(hardness_values) 
        if bag_hardness_values[i] < 0:
            bags.append(easy_subsample(trainset, ih, 0 - bag_hardness_values[i]))  # say, bag hardness is -0.3 Thats the same as creating an easy bag with HC =  0.3
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


""" MDS visualization of dataset """
def visualize_dataset(X, y, ih):
    transformer = MDS(n_components = 2, max_iter=100, n_init=1)
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
    plt.suptitle("MDS of dataset", fontsize = 20)
    #for label , x, y in zip(y_train, X_transformed[:, 0], X_transformed[:, 1]):
    #Lets annotate every 1 out of 20 samples, otherwise graph will be cluttered with anotations
    #  if count % 20 == 0:
    #    plt.annotate(str(int(label)),xy=(x,y), color='black', weight='normal',size=10,bbox=dict(boxstyle="round4,pad=.5", fc="0.8"))
    #  count = count + 1
    #plt.savefig("mnist_pca.png")
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
    np.set_printoptions(threshold=np.inf)  # To print the whole array instead of "..." in the middle parts
    
        
    datasetcount = 0
    
    """ Begin loop for each dataset """  
    for dataset_name in os.listdir('Datasets//temp_rejected_datasets//'):    # dataset_name now contains the trailing '.csv'
        if(dataset_name.endswith('csv')):
            # Create files to write out the prints
            outfilename = "outputs//all_DT_10X5_5X5_dNoLim_10bags//temp_rejected//" + dataset_name[:-4] +"_summary.txt"    
            outfile = open(outfilename,"w")
            
            csvfilename = "outputs//all_DT_10X5_5X5_dNoLim_10bags//temp_rejected//" + dataset_name[:-4] +"_details.csv"     
            csvfile = open(csvfilename, 'w')
            csvwriter = csv.writer(csvfile, delimiter=",", lineterminator='\n')
       
            data = pd.read_csv('Datasets//temp_rejected_datasets//'+dataset_name)   # construct file name of dataset
            X_mis = data.ix[:,:-1]  # Columns 0 to end - 1
            y = data.ix[:,-1]         # Last column
            
            imp = preprocessing.Imputer(missing_values=999,strategy='most_frequent', axis=0)  # Impute missing values which are coded as'999' in all datasets
            X_unscaled = imp.fit_transform(X_mis)
            X = preprocessing.scale(X_unscaled)   # Scale/normalize the data
            #'''
          #  X = X.as_matrix()
            y = y.as_matrix()
            
        
            # Initialize arrays to keep results of metrics
            accuracy_base = [0 for i in range(numruns*numfolds)]   # Accuracy of base learner for each fold in each run   
            accuracy_reg = [0 for i in range(numruns*numfolds)]   # Accuracy of regular bagging for each fold in each run    
            accuracy_pybag = [0 for i in range(numruns*numfolds)] # Accuracy for Python's implementation of bagging
            accuracy_mix = [0 for i in range(numruns*numfolds)]  # Accuracy of mixed bagging for each fold in each run 
            accuracy_wag = [0 for i in range(numruns*numfolds)]   # Accuracy of wagging for each fold in each run    
            accuracy_grad = [0 for i in range(numruns*numfolds)]  # Accuracy of gradually mixed bagging for each fold in each run 
            
            f1_base = [0 for i in range(numruns*numfolds)]
            f1_reg = [0 for i in range(numruns*numfolds)]
            f1_pybag = [0 for i in range(numruns*numfolds)]
            f1_mix = [0 for i in range(numruns*numfolds)]
            f1_wag = [0 for i in range(numruns*numfolds)]
            f1_grad = [0 for i in range(numruns*numfolds)]
        
            auc_base = [0 for i in range(numruns*numfolds)]
            auc_reg = [0 for i in range(numruns*numfolds)]
            auc_pybag = [0 for i in range(numruns*numfolds)]
            auc_mix = [0 for i in range(numruns*numfolds)]
            auc_wag = [0 for i in range(numruns*numfolds)]
            auc_grad = [0 for i in range(numruns*numfolds)]
            
         
            """ Loop for one run of k-fold  evaluation """
            for run in range(numruns):            
                
                print("Run %d of %d" % (run+1, numruns))
                fold = 0
                
                # Split into kfold folds for evaluation 
                skf = StratifiedKFold(n_splits=numfolds, shuffle=True, random_state=run)     # A different random state for each run
                
                ''' Loop for evaluation of test set '''
                for tr_val_idx, test_idx in skf.split(X, y):
                    X_tr_val, X_test = X[tr_val_idx], X[test_idx]
                    Y_tr_val, Y_test = y[tr_val_idx], y[test_idx] 
                    
                    print("\n Still running dataset %s ........" % dataset_name)
                    print("\nFold %d of %d" % (fold+1, numfolds))
        
                    # Initialize some arrays to store cv results for each combo/interval in each cv fold
                    # Basically its a (numcvruns*numcvfold) * numCombo array (numCombo = numhqs*numratios or numCombo = len(hardness_intervals))
                    cv_acc_mx = np.zeros((numcvruns*numcvfolds, numhqs*numratios))  # one for each combo within this cv fold
                    cv_f1_mx   = np.zeros((numcvruns*numcvfolds, numhqs*numratios))
                    cv_auc_mx = np.zeros((numcvruns*numcvfolds, numhqs*numratios))
                            
                    cv_acc_gr = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))   # one for each interval within this cv fold
                    cv_f1_gr   = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))
                    cv_auc_gr = np.zeros((numcvruns*numcvfolds, len(hardness_intervals)))
                    
                    
                    """ Loop for one run of cross validation """
                    for cvrun in range(numcvruns):
                        
                        print ("\nCV run %d of %d" % (cvrun+1, numcvruns))
                        cvfold = 0
                        
                        skf2 = StratifiedKFold(n_splits=numcvfolds, shuffle=True, random_state=cvrun)
                        
                        # Loop for cross validation for model selection
                        for tr_idx, val_idx in skf2.split(X_tr_val, Y_tr_val):
                            X_train, X_validate = X_tr_val[tr_idx], X_tr_val[val_idx]
                            Y_train, Y_validate = Y_tr_val[tr_idx], Y_tr_val[val_idx] 
                            
               #             print("CV Fold %d of %d" % (cvfold+1, numcvfolds))
                            
                            # Calculate IH of training set from external file
                            ih_tr, misclf = ih_calc.calculate_concrete_IH(X_train, Y_train, full=False, clfList=clfList)
                                        
                            # Join the X, Y and IH of training set to make one full dataset for easier manipulation
                            Y_train_reshaped = Y_train.reshape((len(Y_train),1))
                            ih_tr_reshaped = ih_tr.reshape((len(ih_tr),1))
                            trainset = np.concatenate((X_train, Y_train_reshaped, ih_tr_reshaped), axis = 1) #trainset includes both X and Y 
                            
                            mixed_bags = list()
                            for hq, mx in itertools.product(hqs, mix_ratios):                            
                                mixed_bags.append(create_mixed_bags(trainset, ih_tr, mix_ratio=mx, hard_quotient=hq)) # Mixed bags with different mixed ratios
                            
                            grad_bags = list()
                            for h_i in hardness_intervals:   
                                grad_bags.append(create_gradually_mixed_bags(trainset, ih_tr, low_bag_hardness=h_i[0], high_bag_hardness=h_i[1])) # GRadually mixed bags with different intervals
                        
                            # Visualize bags when necessary. Code just has mixed bags, but can try others
                            if vizbags == True:
                                visualize_bags(mixed_bags[2*numratios])   # MR = 3:3:3, HQ = 0.75
                    
            #               print("Building models ...")
            
                             # Define classifier models
                            if classifier == 'DT':
                                clf = DecisionTreeClassifier(random_state= cvrun*numcvfolds + cvfold)   # Best depth known for that dataset. Random state uniquely identifying this cvfold of this cvrun
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
                        
                            # For mixed bagging, evaluate the validation set 
                            for i in range(numhqs*numratios):     # For each mix_ratio for each hq
                                pred, pred_proba = make_models(mixed_bags[i], clf, X_validate, Y_validate)
                                mix_predictions.append(pred)
                                mix_predictions_proba.append(pred_proba)
                                fin_pred, fin_pred_proba = calculate_pred(pred, pred_proba) 
                                mix_final_pred.append(fin_pred)
                                mix_final_pred_proba.append(fin_pred_proba)  
                            
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
                            for i in range(numhqs*numratios):
    #                            cv_acc_mx[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, mix_final_pred_proba[i])
    #                            cv_f1_mx[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, mix_final_pred_proba[i], average='weighted')
    #                            cv_auc_mx[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, mix_final_pred_proba[i])                         
                                cv_acc_mx[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, mix_final_pred[i])
                                cv_f1_mx[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, mix_final_pred[i], average='weighted')
                                cv_auc_mx[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, mix_final_pred[i])
                            # Check all hardness intervals for gradually mixed bags
                            for i in range(len(hardness_intervals)):
    #                            cv_acc_gr[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, grad_final_pred_proba[i])
    #                            cv_f1_gr[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, grad_final_pred_proba[i], average='weighted')
    #                            cv_auc_gr[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, grad_final_pred_proba[i])
                                cv_acc_gr[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, grad_final_pred[i])
                                cv_f1_gr[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, grad_final_pred[i], average='weighted')
                                cv_auc_gr[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, grad_final_pred[i])
                            cvfold = cvfold + 1
                        """ End of one run of cv """
                    """ End of fold for cross validation for model selection """
                    
                    """ We now make models for each method and evaluate on test set """
                     
                    # Use the evaluation matrices to find best parameters for mixed bags
                    val_acc_mx = np.mean(cv_acc_mx, axis=0) # Take mean validation accuracy of each combo by taking mean of each column
                    val_f1_mx = np.mean(cv_f1_mx, axis=0)   # val = validation
                    val_auc_mx = np.mean(cv_auc_mx, axis=0)
                    
                    val_acc_gr = np.mean(cv_acc_gr, axis=0)
                    val_f1_gr = np.mean(cv_f1_gr, axis=0)
                    val_auc_gr = np.mean(cv_auc_gr, axis=0)
                    
                    # Write these values to fie for later inspection
                    
                    csvwriter.writerows([np.round(val_acc_mx,3)])
                    csvwriter.writerows([np.round(val_f1_mx,3)])
                    csvwriter.writerows([np.round(val_auc_mx,3)])
                    csvwriter.writerow([])
                    csvwriter.writerows([np.round(val_acc_gr,3)])
                    csvwriter.writerows([np.round(val_f1_gr,3)])
                    csvwriter.writerows([np.round(val_auc_gr,3)])
                    csvwriter.writerow([])
                    
                    # Find indices of best combo/interval
                    idx_max_acc_mx = np.argmax(val_acc_mx)    # Index of max accuracy in mixed bags
                    idx_max_f1_mx = np.argmax(val_f1_mx)
                    idx_max_auc_mx = np.argmax(val_auc_mx)
        
                    idx_max_acc_gr = np.argmax(val_acc_gr)    # Index of max accuracy in grad bags
                    idx_max_f1_gr = np.argmax(val_f1_gr)
                    idx_max_auc_gr = np.argmax(val_auc_gr)
                
                    # Calculate IH of whole data (training + vaidation) from external file
                    ih, misclf = ih_calc.calculate_concrete_IH(X_tr_val, Y_tr_val, full=False, clfList=clfList)
                    
                    # Visualize when needed
                    if vizdataset==True:
                        visualize_dataset(X_tr_val, Y_tr_val, ih)
                            
                    # Join the X, Y and IH of training+validation set to make one full dataset for easier manipulation
                    Y_tr_val_reshaped = Y_tr_val.reshape((len(Y_tr_val),1))
                    ih_reshaped = ih.reshape((len(ih),1))
                    trainvalset = np.concatenate((X_tr_val, Y_tr_val_reshaped, ih_reshaped), axis = 1) #trainset includes both X and Y 
        
                # Create bags for everyone. For mixed bags, use the combo/interval we found to be best according to validation set
                    # Regular bagging 
                    reg_bags = create_regular_bags(trainvalset) 
                    
                    # Wagging
                    wag_bags = create_wagging_bags(trainvalset) 
                    
                    # Mixed bags
                    # idx_max_xxx_mx now has index of best mixture in terms of acc, f1, auc. But idx contains info of both HQ and mix_ratio. 
                    # Example: numratios = 10, numhqs = 3, so idx= 24 actually means hq = 24/10 = 2 and mix_ratio = 24 % 10 = 4
                    mixed_bags_acc = create_mixed_bags(trainvalset, ih, mix_ratio=mix_ratios[idx_max_acc_mx % numratios], hard_quotient=hqs[idx_max_acc_mx // numratios]) # Mixed bags with different mixed ratios
                    mixed_bags_f1 = create_mixed_bags(trainvalset, ih, mix_ratio=mix_ratios[idx_max_f1_mx % numratios], hard_quotient=hqs[idx_max_f1_mx // numratios]) # Mixed bags with different mixed ratios
                    mixed_bags_auc = create_mixed_bags(trainvalset, ih, mix_ratio=mix_ratios[idx_max_auc_mx % numratios], hard_quotient=hqs[idx_max_auc_mx // numratios]) # Mixed bags with different mixed ratios
                   
                    if verbose == True:    # Print when required
                        outfile.write("\nFor run %d fold %d:" % (run + 1, fold + 1))
                        outfile.write("\t%s  %s" % (mix_ratios[idx_max_acc_mx % numratios], hqs[idx_max_acc_mx // numratios]))    # integer division 
                        outfile.write("\t%s  %s" % (mix_ratios[idx_max_f1_mx % numratios], hqs[idx_max_f1_mx // numratios]))  
                        outfile.write("\t%s  %s" % (mix_ratios[idx_max_auc_mx % numratios], hqs[idx_max_auc_mx // numratios]))
                    
                    # Grad bags                    
                    grad_bags_acc = create_gradually_mixed_bags(trainvalset, ih, low_bag_hardness=hardness_intervals[idx_max_acc_gr][0], high_bag_hardness=hardness_intervals[idx_max_acc_gr][1])
                    grad_bags_f1 = create_gradually_mixed_bags(trainvalset, ih, low_bag_hardness=hardness_intervals[idx_max_f1_gr][0], high_bag_hardness=hardness_intervals[idx_max_f1_gr][1])
                    grad_bags_auc = create_gradually_mixed_bags(trainvalset, ih, low_bag_hardness=hardness_intervals[idx_max_auc_gr][0], high_bag_hardness=hardness_intervals[idx_max_auc_gr][1])
                    
                    if verbose == True:
                     #   outfile.write("\nFor run %d fold %d:" % (run + 1, fold + 1))
                        outfile.write("\t\t%s" % hardness_intervals[idx_max_acc_gr])
                        outfile.write("\t%s" % hardness_intervals[idx_max_f1_gr])     
                        outfile.write("\t%s" % hardness_intervals[idx_max_auc_gr])
                    
                    
                    # Visualize bags when necessary. Code just has mixed bags, but can try others
                    if vizbags == True:
                        visualize_bags(mixed_bags[2*numratios])   # MR = 3:3:3, HQ = 0.75
                
        #            print("Building models ...")
        
                    # Build classifier models for each bag
                    if classifier == 'DT':
                        clf2 = DecisionTreeClassifier(random_state= run*numfolds + fold)
                    elif classifier == 'KNN':
                        clf2 = KNeighborsClassifier(np.floor(np.sqrt(nsamples)/2))  # k = sqrt(n)/2
                    else:   # NB
                        clf2 = GaussianNB() 
                    #clf = KNeighborsClassifier(np.floor(np.sqrt(nsamples)/2))  # k = sqrt(n)/2
                    #clf = GaussianNB()
                    
                    # Create and evaluate a base classifier (NO bagging, just the base learner)
                    clf_base = DecisionTreeClassifier(random_state= run*numfolds + fold)   # random state uniquely identifying this fold of this run
                    clf_base.fit(X_tr_val, Y_tr_val)
                    base_predictions = clf_base.predict(X_test)
               #     csvwriter.writerows([base_predictions])
                    
                    accuracy_base[run*numfolds + fold] = mt.accuracy_score(Y_test, base_predictions)          # all folds in all runs are being kept in aa 1-D array. So (run*numfolds + fold) gives index of thye fold within the array
                    f1_base[run*numfolds + fold] = mt.f1_score(Y_test, base_predictions, average='weighted')
                    auc_base[run*numfolds + fold] = mt.roc_auc_score(Y_test, base_predictions)
                    
                    # Create and evaluate regular bagging                          
                    reg_predictions, reg_predictions_proba = make_models(reg_bags, clf2, X_test, Y_test)  # Reg = regular, Proba = probabilities
                    reg_final_pred, reg_final_pred_proba = calculate_pred(reg_predictions, reg_predictions_proba)
              #      csvwriter.writerows([reg_final_pred_proba])
                    
                    accuracy_reg[run*numfolds + fold] = mt.accuracy_score(Y_test, reg_final_pred_proba)
                    f1_reg[run*numfolds + fold] = mt.f1_score(Y_test, reg_final_pred_proba, average='weighted')
                    auc_reg[run*numfolds + fold] = mt.roc_auc_score(Y_test, reg_final_pred_proba)
                    
                    # Create and evaluate python bagging                          
                    pybag_model = BaggingClassifier(base_estimator=clf2, n_estimators=defnumbags, random_state=run*numfolds + fold)  # random seed uniquely identifying this fold of this run
                    pybag_model.fit(X_tr_val, Y_tr_val)
                    pybag_pred = pybag_model.predict(X_test)
              #      csvwriter.writerows([pybag_pred])
                    
                    accuracy_pybag[run*numfolds + fold] = mt.accuracy_score(Y_test, pybag_pred)
                    f1_pybag[run*numfolds + fold] = mt.f1_score(Y_test, pybag_pred, average='weighted')
                    auc_pybag[run*numfolds + fold] = mt.roc_auc_score(Y_test, pybag_pred)
                    
                    # Create and evaluate wagging 
                    wag_predictions, wag_predictions_proba = make_models(wag_bags, clf2, X_test, Y_test)  # wag = wagging, Proba = probabilities
                    wag_final_pred, wag_final_pred_proba = calculate_pred(wag_predictions, wag_predictions_proba)
              #      csvwriter.writerows([wag_final_pred_proba])
                    
                    accuracy_wag[run*numfolds + fold] = mt.accuracy_score(Y_test, wag_final_pred_proba)
                    f1_wag[run*numfolds + fold] = mt.f1_score(Y_test, wag_final_pred_proba, average='weighted')
                    auc_wag[run*numfolds + fold] = mt.roc_auc_score(Y_test, wag_final_pred_proba)
                    
                    # Create and evaluate mixed bagging
                    mix_predictions_acc, mix_predictions_proba_acc = make_models(mixed_bags_acc, clf2, X_test, Y_test)  # Mix = Mixed bags, Proba = probabilities
                    mix_final_pred_acc, mix_final_pred_proba_acc = calculate_pred(mix_predictions_acc, mix_predictions_proba_acc)
              #      csvwriter.writerows([mix_final_pred_proba_acc])
                    
                    mix_predictions_f1, mix_predictions_proba_f1 = make_models(mixed_bags_f1, clf2, X_test, Y_test)  
                    mix_final_pred_f1, mix_final_pred_proba_f1 = calculate_pred(mix_predictions_f1, mix_predictions_proba_f1)
                    
                    mix_predictions_auc, mix_predictions_proba_auc = make_models(mixed_bags_auc, clf2, X_test, Y_test)  
                    mix_final_pred_auc, mix_final_pred_proba_auc = calculate_pred(mix_predictions_auc, mix_predictions_proba_auc)
                    
                    accuracy_mix[run*numfolds + fold] = mt.accuracy_score(Y_test, mix_final_pred_proba_acc)
                    f1_mix[run*numfolds + fold] = mt.f1_score(Y_test, mix_final_pred_proba_f1, average='weighted')
                    auc_mix[run*numfolds + fold] = mt.roc_auc_score(Y_test, mix_final_pred_proba_auc)
                    
                    # Create and evaluate gradually mixed bagging
                    grad_predictions_acc, grad_predictions_proba_acc = make_models(grad_bags_acc, clf2, X_test, Y_test)  # grad = gradually graded bags, Proba = probabilities
                    grad_final_pred_acc, grad_final_pred_proba_acc = calculate_pred(grad_predictions_acc, grad_predictions_proba_acc)
              #      csvwriter.writerows([grad_final_pred_proba_acc])
                    
                    grad_predictions_f1, grad_predictions_proba_f1 = make_models(grad_bags_f1, clf2, X_test, Y_test)  
                    grad_final_pred_f1, grad_final_pred_proba_f1 = calculate_pred(grad_predictions_f1, grad_predictions_proba_f1)
                    
                    grad_predictions_auc, grad_predictions_proba_auc = make_models(grad_bags_auc, clf2, X_test, Y_test)  
                    grad_final_pred_auc, grad_final_pred_proba_auc = calculate_pred(grad_predictions_auc, grad_predictions_proba_auc)
                    
                    accuracy_grad[run*numfolds + fold] = mt.accuracy_score(Y_test, grad_final_pred_proba_acc)
                    f1_grad[run*numfolds + fold] = mt.f1_score(Y_test, grad_final_pred_proba_f1, average='weighted')
                    auc_grad[run*numfolds + fold] = mt.roc_auc_score(Y_test, grad_final_pred_proba_auc)
                    
    #                if verbose == True:
    #                    print("Base acc(fold): %s" % accuracy_base[run*numfolds + fold])
    #                    print("Reg acc(fold): %s" % accuracy_reg[run*numfolds + fold])
    #                    print("PyBag acc(fold): %s" % accuracy_pybag[run*numfolds + fold])
    #                    print("Wag acc(fold): %s" % accuracy_wag[run*numfolds + fold])
    #                    print("Mix acc(fold): %s" % accuracy_mix[run*numfolds + fold])
    #                    print("Grad acc(fold): %s" % accuracy_grad[run*numfolds + fold])
                               
                    fold = fold + 1
                ''' End of fold for evaluation of test set '''
 
            run = run + 1
            ''' End of one run of k-fold evaluation '''
            
            end = time.time()
            #Print the avg of all the metrics
            outfile.write("\n\n\nFinal results:\n")
            if synthdata == True:
                outfile.write("Inst, Features: %d, %d" % (nsamples, nfeatures))
                outfile.write("\tClass balance:" + str(balance))
                outfile.write("\tClass sep:" + str(classsep))
                outfile.write("\tRand state:" + str(randomstate))
    
            else:  # Real data
                outfile.write("Dataset: %s" % dataset_name)
                outfile.write("\nDataset with " + str(X.shape[0]) + " instances and " + str(X.shape[1]) + " predictive features")
            
    #        if verbose == True:
    #            print("Base %s" % accuracy_base)
    #            print("Reg %s" % accuracy_reg)
    #            print("Wag %s" % accuracy_wag)
    #            print("Mix %s" % accuracy_mix)
    #            print("Grad %s" % accuracy_grad)
            
            # Final results for the whole dataset
            csvwriter.writerows([])
            csvwriter.writerows([accuracy_base])
            csvwriter.writerows([accuracy_reg])
            csvwriter.writerows([accuracy_pybag])
            csvwriter.writerows([accuracy_wag])
            csvwriter.writerows([accuracy_mix])
            csvwriter.writerows([accuracy_grad])
            
            outfile.write("\n\t\tBase\t\tRegular\t\tPybag\t\tWagging\t\tMixed (opt)\tGrad-Mixed (opt)")
            outfile.write("\nAccuracy\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f" % 
                          ( np.mean(accuracy_base), np.std(accuracy_base), 
                            np.mean(accuracy_reg),  np.std(accuracy_reg), 
                            np.mean(accuracy_pybag),  np.std(accuracy_pybag),
                            np.mean(accuracy_wag),  np.std(accuracy_wag), 
                            np.mean(accuracy_mix),  np.std(accuracy_mix), 
                            np.mean(accuracy_grad), np.std(accuracy_grad)))
            outfile.write("\nF1\t\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f" % 
                          ( np.mean(f1_base), np.std(f1_base), 
                            np.mean(f1_reg),  np.std(f1_reg), 
                            np.mean(f1_pybag),  np.std(f1_pybag),
                            np.mean(f1_wag),  np.std(f1_wag), 
                            np.mean(f1_mix),  np.std(f1_mix), 
                            np.mean(f1_grad), np.std(f1_grad)))
            outfile.write("\nAUC\t\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f\t%.3f, %.3f" % 
                          ( np.mean(auc_base), np.std(auc_base), 
                            np.mean(auc_reg),  np.std(auc_reg),
                            np.mean(auc_pybag),  np.std(auc_pybag),
                            np.mean(auc_wag),  np.std(auc_wag), 
                            np.mean(auc_mix),  np.std(auc_mix), 
                            np.mean(auc_grad), np.std(auc_grad)))
            
            
            outfile.write("\n\nTime taken: %.3f s" % (end - start))
            
            acc_basereg_sigtest = st.wilcoxon(accuracy_base, accuracy_reg)
            acc_basewag_sigtest = st.wilcoxon(accuracy_base, accuracy_wag)
            acc_basemix_sigtest = st.wilcoxon(accuracy_base, accuracy_mix)
            acc_basegrad_sigtest = st.wilcoxon(accuracy_base, accuracy_grad)
            acc_regmix_sigtest = st.wilcoxon(accuracy_reg, accuracy_mix)
            acc_regwag_sigtest = st.wilcoxon(accuracy_reg, accuracy_wag)
            acc_reggrad_sigtest = st.wilcoxon(accuracy_reg, accuracy_grad)
            acc_wagmix_sigtest = st.wilcoxon(accuracy_wag, accuracy_mix)
            acc_waggrad_sigtest = st.wilcoxon(accuracy_wag, accuracy_grad)
            acc_mixgrad_sigtest = st.wilcoxon(accuracy_mix, accuracy_grad)
            acc_pybase_sigtest = st.wilcoxon(accuracy_pybag, accuracy_base)
            acc_pyreg_sigtest = st.wilcoxon(accuracy_pybag, accuracy_reg)
            acc_pywag_sigtest = st.wilcoxon(accuracy_pybag, accuracy_wag)
            acc_pymix_sigtest = st.wilcoxon(accuracy_pybag, accuracy_mix)
            acc_pygrad_sigtest = st.wilcoxon(accuracy_pybag, accuracy_grad)
            
            
            f1_basereg_sigtest = st.wilcoxon(f1_base, f1_reg)
            f1_basewag_sigtest = st.wilcoxon(f1_base, f1_wag)
            f1_basemix_sigtest = st.wilcoxon(f1_base, f1_mix)
            f1_basegrad_sigtest = st.wilcoxon(f1_base, f1_grad)
            f1_regmix_sigtest = st.wilcoxon(f1_reg, f1_mix)
            f1_regwag_sigtest = st.wilcoxon(f1_reg, f1_wag)
            f1_reggrad_sigtest = st.wilcoxon(f1_reg, f1_grad)
            f1_wagmix_sigtest = st.wilcoxon(f1_wag, f1_mix)
            f1_waggrad_sigtest = st.wilcoxon(f1_wag, f1_grad)
            f1_mixgrad_sigtest = st.wilcoxon(f1_mix, f1_grad)
            f1_pybase_sigtest = st.wilcoxon(f1_pybag, f1_base)
            f1_pyreg_sigtest = st.wilcoxon(f1_pybag, f1_reg)
            f1_pywag_sigtest = st.wilcoxon(f1_pybag, f1_wag)
            f1_pymix_sigtest = st.wilcoxon(f1_pybag, f1_mix)
            f1_pygrad_sigtest = st.wilcoxon(f1_pybag, f1_grad)
            
            auc_basereg_sigtest = st.wilcoxon(auc_base, auc_reg)
            auc_basewag_sigtest = st.wilcoxon(auc_base, auc_wag)
            auc_basemix_sigtest = st.wilcoxon(auc_base, auc_mix)
            auc_basegrad_sigtest = st.wilcoxon(auc_base, auc_grad)
            auc_regmix_sigtest = st.wilcoxon(auc_reg, auc_mix)
            auc_regwag_sigtest = st.wilcoxon(auc_reg, auc_wag)
            auc_reggrad_sigtest = st.wilcoxon(auc_reg, auc_grad)
            auc_wagmix_sigtest = st.wilcoxon(auc_wag, auc_mix)
            auc_waggrad_sigtest = st.wilcoxon(auc_wag, auc_grad)
            auc_mixgrad_sigtest = st.wilcoxon(auc_mix, auc_grad)
            auc_pybase_sigtest = st.wilcoxon(auc_pybag, auc_base)
            auc_pyreg_sigtest = st.wilcoxon(auc_pybag, auc_reg)
            auc_pywag_sigtest = st.wilcoxon(auc_pybag, auc_wag)
            auc_pymix_sigtest = st.wilcoxon(auc_pybag, auc_mix)
            auc_pygrad_sigtest = st.wilcoxon(auc_pybag, auc_grad)
            
            outfile.write("\n\nSiginificance tests:")
            outfile.write("\nFor Accuracy:")
            outfile.write("\n\tBase\tReg_bag\tPy_bag\tWagging\tMix_bag\tGrad_mix")        # Divide p-value by 2 since we can use one-tailed test
            outfile.write("\nBase\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (acc_basereg_sigtest.pvalue, acc_pybase_sigtest.pvalue, acc_basewag_sigtest.pvalue, acc_basemix_sigtest.pvalue, acc_basegrad_sigtest.pvalue))
            outfile.write("\nReg_bag\t\t\t%.3f\t%.3f\t%.3f\t%.3f" % (acc_pyreg_sigtest.pvalue, acc_regwag_sigtest.pvalue, acc_regmix_sigtest.pvalue, acc_reggrad_sigtest.pvalue))
            outfile.write("\nPy_bag\t\t\t\t%.3f\t%.3f\t%.3f" % (acc_pywag_sigtest.pvalue, acc_pymix_sigtest.pvalue, acc_pygrad_sigtest.pvalue))
            outfile.write("\nWagging\t\t\t\t\t%.3f\t%.3f" % (acc_wagmix_sigtest.pvalue, acc_waggrad_sigtest.pvalue))
            outfile.write("\nMix_bag\t\t\t\t\t\t%.3f" % (acc_mixgrad_sigtest.pvalue))
            
            outfile.write("\nFor F1:")
            outfile.write("\n\tBase\tReg_bag\tPy_bag\tWagging\tMix_bag\tGrad_mix")        # Divide p-value by 2 since we can use one-tailed test
            outfile.write("\nBase\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (f1_basereg_sigtest.pvalue, f1_pybase_sigtest.pvalue, f1_basewag_sigtest.pvalue, f1_basemix_sigtest.pvalue, f1_basegrad_sigtest.pvalue))
            outfile.write("\nReg_bag\t\t\t%.3f\t%.3f\t%.3f\t%.3f" % (f1_pyreg_sigtest.pvalue, f1_regwag_sigtest.pvalue, f1_regmix_sigtest.pvalue, f1_reggrad_sigtest.pvalue))
            outfile.write("\nPy_bag\t\t\t\t%.3f\t%.3f\t%.3f" % (f1_pywag_sigtest.pvalue, f1_pymix_sigtest.pvalue, f1_pygrad_sigtest.pvalue))
            outfile.write("\nWagging\t\t\t\t\t%.3f\t%.3f" % (f1_wagmix_sigtest.pvalue, f1_waggrad_sigtest.pvalue))
            outfile.write("\nMix_bag\t\t\t\t\t\t%.3f" % (f1_mixgrad_sigtest.pvalue))
            
            outfile.write("\nFor AUC:")
            outfile.write("\n\tBase\tReg_bag\tPy_bag\tWagging\tMix_bag\tGrad_mix")        # Divide p-value by 2 since we can use one-tailed test
            outfile.write("\nBase\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (auc_basereg_sigtest.pvalue, auc_pybase_sigtest.pvalue, auc_basewag_sigtest.pvalue, auc_basemix_sigtest.pvalue, auc_basegrad_sigtest.pvalue))
            outfile.write("\nReg_bag\t\t\t%.3f\t%.3f\t%.3f\t%.3f" % (auc_pyreg_sigtest.pvalue, auc_regwag_sigtest.pvalue, auc_regmix_sigtest.pvalue, auc_reggrad_sigtest.pvalue))
            outfile.write("\nPy_bag\t\t\t\t%.3f\t%.3f\t%.3f" % (auc_pywag_sigtest.pvalue, auc_pymix_sigtest.pvalue, auc_pygrad_sigtest.pvalue))
            outfile.write("\nWagging\t\t\t\t\t%.3f\t%.3f" % (auc_wagmix_sigtest.pvalue, auc_waggrad_sigtest.pvalue))
            outfile.write("\nMix_bag\t\t\t\t\t\t%.3f" % (auc_mixgrad_sigtest.pvalue))
            
            # Finishing tasks at end of dataset
            winsound.Beep(300, 500)
        
            datasetcount = datasetcount + 1
            
            outfile.close()
            csvfile.close()
    """ End of loop for dataset """
    
    
    winsound.Beep(300, 2500)      # Ending notification sound