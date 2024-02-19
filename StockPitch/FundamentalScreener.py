#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024
The Program creates a class that identifies cheap stocks using a combination of Decision Trees Regression and Data Envelopment Analysis. It does so by the following
1. Gets a list of all technology stocks currently in the Russell 1000.
2. Extracts financial information (operating and valuation statistics) for the obtained stocks.
3. Extracts period return data for the obtained stocks.
4. Fits a decision tree model using the valuation and operating statistics as features and the period returns as the target variable.
5. Plots the decision tree and selects the best group of stocks(highest return) based on the plot.
6. Runs Data Envelopment Analysis (DEA) to evaluate the efficiency of the selected stocks and returns the top 5 cheapest stocks based on their normalized efficiency score.

@author: Kobena Amoah
"""
import warnings
import datetime as dt
import pandas as pd
import numpy as np
import random
import yfinance as yf
from yahoo_fin import stock_info as si
from RussellSectorComponents import *
from DataExtraction import *
from TunedDecisionTreeRegressor import *
from DEA import *
from scipy.stats import poisson, uniform, norm

# Ignore warnings, mainly DeprecationWarnings
warnings.filterwarnings("ignore")

class FundamentalScreener:
    """
    A class for identifying cheap stocks using a combination of Decision Trees Regression and Data Envelopment Analysis.
    The Decision Trees is run to identify the structure of high performing stocks in a sector given certain characteristics. The DEA is used to identify the cheapness, so to speak, of each stock in the group on a scale of 0 - 1.

    Attributes:
        sector (str): The sector for which to analyze stocks.
        operating_names (list): List of operating statistic names.
        valuation_names (list): List of valuation statistic names.
    """

    def __init__(self, sector, operating_names, valuation_names):
        """
        Initializes the StockAnalysis object.

        Parameters:
            sector (str): The sector for which to analyze stocks.
            operating_names (list): List of operating statistic names.
            valuation_names (list): List of valuation statistic names.
        """
        self.sector = sector
        self.operating_names = operating_names
        self.valuation_names = valuation_names

    def get_sector_stocks(self):
        """
        Retrieves a list of tech stocks currently in the Russell 1000 for the specified sector.

        Returns:
            list: List of tech stocks.
        """
        russell_1000 = RussellSectorComponents(sector=self.sector)
        return russell_1000.russell_components_with_sector()

    def extract_financial_info(self, tickers,data_range):
        """
        Extracts operating and valuation statistics for the specified list of tickers.

        Parameters:
            tickers (list): List of stock tickers.

        Returns:
            DataFrame: DataFrame containing extracted operating statistics.
            DataFrame: DataFrame containing extracted valuation statistics.
        """
        data_extractor = DataExtraction(tickers, self.operating_names, self.valuation_names)
        data_extractor.extract_operating_stats(data_range)
        data_extractor.extract_valuation_stats(data_range)
        return data_extractor.operating_stats, data_extractor.valuation_stats

    def get_period_returns(self, tickers):
        """
        Retrieves period returns for the specified list of tickers.

        Parameters:
            tickers (list): List of stock tickers.

        Returns:
            dict: Dictionary containing period returns for each stock.
        """
        data_extractor = DataExtraction(tickers, [], [])
        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
        return data_extractor.get_period_returns(start_date, end_date)

    def fit_decision_tree_model(self, operating_stats, valuation_stats, period_returns):
        """
        Fits a decision tree model to the extracted financial data.

        Parameters:
            operating_stats (DataFrame): DataFrame containing operating statistics.
            valuation_stats (DataFrame): DataFrame containing valuation statistics.
            period_returns (DataFrame or Series): DataFrame or Series containing period returns for each stock.
        """
        stock_info = pd.concat([operating_stats, valuation_stats], axis=1)
        period_returns = period_returns.iloc[:,0]
        # Convert period_returns to DataFrame if it's a Series
        if isinstance(period_returns, pd.Series):
            period_returns = pd.DataFrame(period_returns, columns=['Period_Return'])
        
        # Instantiate TunedDecisionTreeRegressor
        tuner = TunedDecisionTreeRegressor()

        # Define the parameter grid
        param_grid = [{
            'max_depth': [3, 5, 7, 9,11,13],  # Depth of the tree
            'min_samples_split': [2, 5, 7,10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4, 6],  # Minimum number of samples required to be at a leaf node
            'max_features': [None, 'sqrt', 'log2']  # Number of features to consider when looking for the best split
        }]

        # Fit the model with hyperparameter tuning
        tuner.fit_with_tuning(stock_info, period_returns, param_grid) 
        tuner.plot_tree_model()
        
    def select_best_stocks(self, operating_stats, valuation_stats, query_criteria):
        """
        Selects the best group of stocks based on the specified query criteria.

        Parameters:
            stock_info (DataFrame): DataFrame containing stock information.
            query_criteria (str): Criteria for selecting the best stocks using a query string.

        Returns:
            DataFrame: DataFrame containing the selected best stocks.
        """
        stock_info = pd.concat([operating_stats, valuation_stats], axis=1)
        stock_info.rename(columns=lambda x: x.replace(' ', '_').replace('/', '_'),inplace=True)
        print(stock_info.columns)
        print(stock_info)
        best_stocks = stock_info.query(query_criteria)
        return best_stocks


    def run_dea_and_return_cheapest_stocks(self, best_stocks, cheapest_number):
        """
        Runs DEA and returns the cheapest stocks in the group.

        Parameters:
            best_stocks (DataFrame): DataFrame containing the selected best stocks.

        Returns:
            DataFrame: DataFrame containing the cheapest stocks.
        """
        inputs = best_stocks.iloc[:, 1:8]
        outputs = best_stocks.iloc[:, 8:]
        screen = DEA(inputs=inputs, outputs=outputs)
        status, weights, efficiency = screen.solve()

        def rinfunc(ds, column):
            ds_rank = ds[column].rank()
            numerator = ds_rank - 0.5
            par = numerator / len(ds)
            result = norm.ppf(par)
            return result

        efficiency["Normalized_Efficiency"] = rinfunc(efficiency, 'Efficiency')
        lower_bound = efficiency['Normalized_Efficiency'].quantile(0.05)
        upper_bound = efficiency['Normalized_Efficiency'].quantile(0.95)
        efficiency['Normalized_Efficiency'] = efficiency['Normalized_Efficiency'].clip(lower=lower_bound, upper=upper_bound)
        cheapest_number = cheapest_number
        temp = efficiency.sort_values(by=['Normalized_Efficiency'], ascending=True)[:cheapest_number]

        res = pd.DataFrame(data=np.nan, index=temp.index, columns=['Efficiency', 'Market Cap'])
        res['Efficiency'] = temp.iloc[:, 0]
        for i in range(len(res.index)):
            val = si.get_stats_valuation(res.index[i])
            res.iloc[i, 1] = val.loc[val.iloc[:, 0].str.contains('Market Cap')].iloc[0, 1]

        return res
    
if __name__ == "__main__":
    # Define the sector, operating names, and valuation names for analysis
    sector = "Information Technology"
    operating_names = ['Beta','Operating Margin', 'Profit Margin', 'Revenue Per Share', 'Return on Assets','Return on Equity','Diluted EPS']
    valuation_names = ['Trailing P/E', 'Forward P/E','Enterprise Value/Revenue','Enterprise Value/EBITDA','Price/Book','PEG Ratio','Price/Sales']
    
    # Initialize an instance of FundamentalScreener with the specified sector and financial metric names
    analyzer = FundamentalScreener(sector, operating_names, valuation_names)
    
    # Retrieve a list of technology stocks in the Russell 1000 for the specified sector
    tickers = analyzer.get_sector_stocks()
    
    # Extract operating and valuation statistics for the retrieved list of tickers
    operating_stats, valuation_stats = analyzer.extract_financial_info(tickers,list(range(1)) + [-1])
    
    # Retrieve period returns for the specified list of tickers
    period_returns = analyzer.get_period_returns(tickers) 
    
    # Drop the index
    operating_stats_all = operating_stats.reset_index(drop=True)
    valuation_stats_all = valuation_stats.reset_index(drop=True)
    period_returns = period_returns.reset_index(drop=True)  
    
    # Fit a decision tree model using the operating and valuation statistics as features and period returns as the target variable
    analyzer.fit_decision_tree_model(operating_stats_all, valuation_stats_all, period_returns)
    
    # Define the query criteria for selecting the best stocks based on financial metrics
    query_criteria = 'Diluted_EPS < 7.4  and Price_Sales < 2.63 and Enterprise_Value_EBITDA < 246.635'
    
    # Extract operating and valuation statistics again for further analysis
    operating_stats, valuation_stats = analyzer.extract_financial_info(tickers,list(range(2)))
    
    # Select the best group of stocks based on the specified query criteria
    best_stocks = analyzer.select_best_stocks(operating_stats, valuation_stats, query_criteria)
    
     # Specify the number of cheapest stocks to return
    cheapest_number = 10 
    
    # Run DEA and return the top cheapest stocks based on their normalized efficiency score
    result = analyzer.run_dea_and_return_cheapest_stocks(best_stocks, cheapest_number)
    
    # Print the result
    print(result)