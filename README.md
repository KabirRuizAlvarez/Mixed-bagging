# Mixed-bagging
This repository contains the code and data for the experiments in our paper "Mixed Bagging based on Instance Hardness: A Novel Ensemble Learning Framework for Supervised Classification"

The code folder has the following programs:
  "Mixed bagging v12 (with Adaboost) code for ICDM 2018.py" - Where the proposed and traditional methods have been implemented and tested with cross-validation on the datasets,
  "Estimate Bias-Variance v3 (Kohavi-Wolpert-Boosting added).py" - Where the bias-variance estimation method (a slightly modified version of Kohavi-Wolpert) has been implemented
  "Mixed bag Full dataset v6 (visualizations).py" - Where the final bootsraps for the whole dataset can be built, and some visualization on the dataset is done,
  "Visualize datasets.py" - For visualization of decision boundary and lower dimensional projection,
  "ih_calc.py" - To calculate Instance Hardness (IH),
  "mix_bag_library.py" - Necessary functions that other programs can import.

The dataset folder contains the 47 datasets (after necessary preprocessing) used in the experiments 

