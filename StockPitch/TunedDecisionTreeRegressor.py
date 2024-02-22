#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024
This program sets up a class to run a decision tree algorithm to evaluate the best valuation predictors for stocks.

@author: Kobena Amoah
"""
import numpy as np
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz

class TunedDecisionTreeRegressor:
    """
    A class for performing hyperparameter tuning on DecisionTreeRegressor models.

    Attributes:
        max_depth (int): Maximum depth of the decision tree. Default is None.
        best_model (DecisionTreeRegressor): Best trained model after hyperparameter tuning.
        best_model_params (dict): Parameters of the best model found during tuning.
        best_score (float): Best score achieved during hyperparameter tuning.
        dt_regressor (DecisionTreeRegressor): Decision tree regressor instance.
    """

    def __init__(self):
        """
        Initializes the TunedDecisionTreeRegressor object.

        Parameters:
            max_depth (int): Maximum depth of the decision tree. Default is None.
        """
        self.min_samples_split = 20  # Minimum number of samples required to split an internal node
        self.min_samples_leaf = 20  # Minimum number of samples required to be at a leaf node
        self.best_model = None
        self.best_model_params = None
        self.best_score = None
        self.x_val = None
        self.dt_regressor = DecisionTreeRegressor(min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
        
        
    def fit_with_tuning(self, X, y, param_grid, cv=5, scoring='neg_mean_squared_error',n_jobs = -1, refit=True):
        """
        Fits the decision tree regressor with hyperparameter tuning using GridSearchCV.

        Parameters:
            X (array-like or sparse matrix): Features.
            y (array-like): Target variable.
            param_grid (dict): Dictionary with parameters names as keys and lists of parameters to try.
            cv (int or cross-validation generator): Determines the cross-validation splitting strategy. Default is 5.
            scoring (str): Scoring method. Default is 'neg_mean_squared_error'.
        """
        grid_search = GridSearchCV(self.dt_regressor, param_grid=param_grid, cv=cv, scoring=scoring,n_jobs = n_jobs,refit=refit)
        self.x_val = X.columns
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        
    def plot_tree_model(self):
        """
        Plots the decision tree of the best model found during hyperparameter tuning.

        Parameters:
            feature_names (list): Names of the features.
        """
        if self.best_model is None:
            print("No best model available. Please fit the model first.")
            return
        
        export_graphviz(self.best_model,out_file = "tree.dot",feature_names= self.x_val, filled=True, impurity=False )
        with open("tree.dot") as f:
            dot_graph = f.read()
        return graphviz.Source(dot_graph)
        
    
