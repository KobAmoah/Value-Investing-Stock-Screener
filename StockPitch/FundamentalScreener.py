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
6. Runs Data Envelopment Analysis (DEA) to evaluate the efficiency of the selected stocks and returns the top 10 cheapest stocks based on their normalized efficiency score.

@author: Kobena Amoah
"""
import warnings
import datetime as dt
import pandas as pd
import numpy as np
import random
import yfinance as yf
from YahooFinanceScraper import *
from RussellSectorComponents import *
from DataExtraction import *
from TunedDecisionTreeRegressor import *
from DEA import *
from scipy.stats import poisson, uniform, norm
from pandas.tseries.offsets import DateOffset, YearBegin

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

    def extract_financial_info(self, tickers,data_range,ip_list):
        """
        Extracts operating and valuation statistics for the specified list of tickers.

        Parameters:
            tickers (list): List of stock tickers.

        Returns:
            DataFrame: DataFrame containing extracted operating statistics.
            DataFrame: DataFrame containing extracted valuation statistics.
        """
        data_extractor = DataExtraction(tickers, self.operating_names, self.valuation_names)
        data_extractor.extract_operating_stats(data_range,ip_list)
        data_extractor.extract_valuation_stats(data_range,ip_list)
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
        today = dt.datetime.now()
        end_date = today.strftime("%Y-%m-%d")
        # Get 1 year cumulative returns
        start_date = (today -  DateOffset(months=12)).strftime("%Y-%m-%d")
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
        # Convert period_returns to DataFrame if it's a Series
        if isinstance(period_returns, pd.Series):
            period_returns = pd.DataFrame(period_returns, columns=['Period_Return'])
        
        # Instantiate TunedDecisionTreeRegressor
        tuner = TunedDecisionTreeRegressor()

        # Define the parameter grid
        param_grid = [{
            'max_depth':  list(range(3, stock_info.shape[1])),  # Depth of the tree
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
        best_stocks = stock_info.query(query_criteria)
        return best_stocks


    def run_dea_and_return_cheapest_stocks(self, best_stocks, cheapest_number,ip_list,data_range):
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
        cheapest_number = cheapest_number
        temp = efficiency.sort_values(by=['Efficiency'], ascending=True)[:cheapest_number]

        res = pd.DataFrame(data=np.nan, index=temp.index, columns=['Efficiency', 'Market Cap'])
        res['Efficiency'] = temp.iloc[:, 0]
        val = DataExtraction(list(res.index), self.operating_names, self.valuation_names)
        val.extract_valuation_stats(data_range,ip_list,all_data = True)
        val = val.valuation_stats
        market_cap = val['Market Cap']
        market_cap = market_cap / 1_000_000_000  # Convert to billions
        res['Market Cap (Billions)'] = market_cap 
            
        # Obtain Stock list from Russell 1000 (This enables us to get company name)
        url = f"http://en.wikipedia.org/wiki/Russell_1000_Index"
        russell = pd.read_html(url)[2]
        russell["Ticker"] = russell["Ticker"].str.replace(".", "-")
        russell = russell.iloc[:,:2]
        russell = pd.DataFrame(russell['Company'].values, index=russell['Ticker'], columns=['Company'])
        # Join Russell and res on common indices
        res = russell.join(res, how='inner').sort_values('Efficiency')
        res = res.drop('Market Cap', axis=1)
        
        return res