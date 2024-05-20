#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024
The Program does the following 
1. Gets a list of all technology stocks currently in the Russell 1000.
2. Extracts financial information (operating and valuation statistics) for the obtained stocks.
3. Extracts period return data for the obtained stocks.
4. Fits a decision tree model using the valuation and operating statistics as features and the period returns as the target variable.
5. Plots the decision tree and selects the best group of stocks(highest return) based on the plot.
6. Runs Data Envelopment Analysis (DEA) to evaluate the efficiency of the selected stocks and returns the top 5 cheapest stocks based on their normalized efficiency score.

@author: Kobena Amoah
"""

from scipy.stats import poisson, uniform, norm
import datetime as dt
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import yfinance as yf
from yahoo_fin import stock_info as si
from RussellSectorComponents import *
from DataExtraction import *
from TunedDecisionTreeRegressor import *
from DEA import *

### 1. Get list of all Tech stocks currently in the Russell 1000
# Instantiate RussellSectorComponents to and get tech stocks in Russell 1000
sector = "Information Technology"
russell_1000 = RussellSectorComponents(sector=sector)

# Retrieve Russell 1000 components for the Information Technology sector
tickers_all = russell_1000.russell_components_with_sector()

#Randomly select stocks to avoid query limit
tickers = random.sample(list(tickers_all), min(len(tickers_all),100))

### 2. Extract Financial Info
# Define lists of operating and valuation stats names
operating_names = ['Beta','Operating Margin', 'Profit Margin', 'Revenue Per Share', 'Return on Assets','Return on Equity','Diluted EPS','Revenue Growth','Debt/Equity']
valuation_names = ['Trailing P/E', 'Forward P/E','Enterprise Value/Revenue','Enterprise Value/EBITDA','Price/Book','PEG','Price/Sales']
# Instantiate DataExtraction object for extracting financial info
data_extractor = DataExtraction(tickers, operating_names, valuation_names)
# Extract operating and valuation stats
data_extractor.extract_operating_stats()
data_extractor.extract_valuation_stats()
# Clean the extracted data
data_extractor.clean_data()
# Access the extracted operating statistics DataFrame
operating_stats = data_extraction.operating_stats
# Access the extracted valuation statistics DataFrame
valuation_stats = data_extraction.valuation_stats

### 3. Get Stock Return Data
# Instantiate DataExtraction for getting period returns
data_extractor = DataExtraction(tickers, [], [])
# Set end date as current date and start date as one year before
end_date = dt.datetime.now().strftime("%Y-%m-%d")
start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
# Get period returns
period_returns = data_extractor.get_period_returns(start_date, end_date)

### 4. Decision Tree Model
# Instantiate TunedDecisionTreeRegressor for tuning decision tree model
tuned_dt = TunedDecisionTreeRegressor(max_depth=None)
# Join operating and valuation stats and drop NaN values
stock_info = operating_stats.join(valuation_stats[['Trailing P/E','Enterprise Value/Revenue','Enterprise Value/EBITDA','Price/Book','Price/Sales']]).dropna()
# Combine the three dataframes as one
stock_info = pd.concat([period_returns,tickers, stock_info], axis=1)
# Replace Space with _ in Column names 
stock_info.rename(columns={'Operating Margin':'Operating_Margin',
                        'Profit Margin':'Profit_Margin', 'Revenue Per Share':'Revenue_Per_Share', 'Return on Assets':'Return_on_Assets','Return on Equity':'Return_on_Equity','Diluted EPS':'Diluted_EPS','Revenue Growth':'Revenue_Growth','Trailing P/E':'Trailing_P/E', 'Forward P/E': 'Forward_P/E','Enterprise Value/Revenue':'Enterprise_Value/Revenue','Enterprise Value/EBITDA':'Enterprise_Value/EBITDA'}, inplace=True)
# Fit decision tree with tuning
tuned_dt.fit_with_tuning(stock_info.iloc[:, 2:], stock_info.iloc[:, 0], param_distributions={'max_depth': poisson(mu=2, loc=2), 'max_leaf_nodes': poisson(mu=5, loc=5), 'min_samples_split': uniform(), 'min_samples_leaf': uniform()}, n_iter=10, cv=5, random_state=42)
# Get best model parameters
best_model_params = tuned_dt.get_best_model_params()
# Create new model with best parameters
best_dt_model = DecisionTreeRegressor(**best_model_params)
# Plot decision tree
feature_names = list(stock_info.columns) 
tuned_dt.plot_tree(feature_names)