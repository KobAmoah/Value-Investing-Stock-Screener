#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat April  13 2024

Program Description:
    This program defines a class named SectorAnalyzer which identifies cheap stocks using a combination of Decision Trees Regression and Data Envelopment Analysis (DEA).

@author: Kobena Amoah
"""

import pandas as pd
import time
from pprint import pprint
from FundamentalScreener import *
from YahooFinanceScraper import *

class SectorAnalyzer:
    """
    A class for analyzing sectors to identify cheap stocks using a combination of Decision Trees Regression and Data Envelopment Analysis (DEA).

    Attributes:
        sector (str): The sector to analyze.
        operating_names (list): List of operating financial metric names.
        valuation_names (list): List of valuation financial metric names.
        analyzer (FundamentalScreener): An instance of the FundamentalScreener class for performing financial analysis.
    """

    def __init__(self, sector, operating_names, valuation_names):
        """
        Initializes the SectorAnalyzer object.

        Parameters:
            sector (str): The sector to analyze.
            operating_names (list): List of operating financial metric names.
            valuation_names (list): List of valuation financial metric names.
        """
        self.sector = sector
        self.operating_names = operating_names
        self.valuation_names = valuation_names
        self.analyzer = FundamentalScreener(sector, operating_names, valuation_names)

    def analyze_sector(self, cheapest_number, query_criteria):
        """
        Analyzes the sector to identify cheap stocks based on the provided criteria.

        Parameters:
            cheapest_number (int): The number of cheapest stocks to return.
            query_criteria (str): The criteria for selecting the best stocks based on financial metrics.

        Returns:
            pd.DataFrame: DataFrame containing information about the cheapest stocks identified.
        """
        print("Thank you for using the Fundamental Stock Screener.")
        print("This program will take roughly 6 hours to complete.\n ")
        # Retrieve a list of stocks in the specified sector
        t0 = time.time()
        
        tickers = self.analyzer.get_sector_stocks()

        # Retrieve period returns for the specified list of tickers
        period_returns = self.analyzer.get_period_returns(tickers)
        
        data = YahooFinanceScraper()
        ip_list = data.get_iplist()

        # Extract operating and valuation statistics for the retrieved list of tickers
        operating_stats, valuation_stats = self.analyzer.extract_financial_info(tickers, list(range(1)) + [-1],ip_list)

        # Combine period returns, operating, and valuation statistics into a single DataFrame
        combined_df = pd.concat([period_returns.reset_index(drop=True),
                                 operating_stats.reset_index(drop=True),
                                 valuation_stats.reset_index(drop=True)], axis=1)

        # Remove rows with NaNs
        combined_df = combined_df.fillna(0)


        # Split combined DataFrame into period returns, operating statistics, and valuation statistics
        period_returns_all = combined_df.iloc[:, 0]
        operating_stats_all = combined_df[self.operating_names]
        valuation_stats_all = combined_df[self.valuation_names]

        # Fit a decision tree model using the operating and valuation statistics as features and period returns as the target variable
        print("Fitting decision tree model")
        self.analyzer.fit_decision_tree_model(operating_stats_all, valuation_stats_all, period_returns_all)
        
        # Extract operating and valuation statistics again for further analysis
        ip_list = data.get_iplist() #Obtain updated ip_list
        operating_stats, valuation_stats = self.analyzer.extract_financial_info(tickers, list(range(2)),ip_list)
        
        # Combine operating, and valuation statistics into a single DataFrame
        combined_df = pd.concat([operating_stats,valuation_stats], axis=1)

        # Remove rows with NaNs
        combined_df = combined_df.fillna(0)
        
        # Split combined DataFrame into, operating statistics, and valuation statistics
        operating_stats = combined_df[self.operating_names]
        valuation_stats = combined_df[self.valuation_names]                        

        # Select the best group of stocks based on the specified query criteria
        best_stocks = self.analyzer.select_best_stocks(operating_stats, valuation_stats, query_criteria)

        # Run DEA and return the top cheapest stocks based on their normalized efficiency score
        print("Fitting DEA model")
        ip_list = data.get_iplist() #Obtain updated ip_list
        result = self.analyzer.run_dea_and_return_cheapest_stocks(best_stocks, cheapest_number,ip_list,list(range(2)))
        
        t1 = time.time()
        total = t1-t0
        
        total = time.strftime("%H:%M:%S",time.gmtime(total))
        print(f"The program took {total} to run\n")

        return result

    

if __name__ == "__main__":
    # Define the sector, operating names, and valuation names for analysis
    sector = "Information Technology"
    operating_names = ['Beta', 'Operating Margin', 'Profit Margin', 'Revenue Per Share', 'Return on Assets',
                       'Return on Equity', 'Diluted EPS']
    valuation_names = ['Trailing P/E', 'Forward P/E', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA',
                       'Price/Book', 'PEG Ratio', 'Price/Sales']

    # Define the query criteria for selecting the best stocks based on financial metrics
    query_criteria = "Return_on_Assets < 42.81 and Diluted_EPS > 8.345"

    # Specify the number of cheapest stocks to return
    cheapest_number = 10    
    
    # Create an instance of SectorAnalyzer
    analyzer = SectorAnalyzer(sector, operating_names, valuation_names)

    # Call the method to perform the analysis
    result = analyzer.analyze_sector(cheapest_number, query_criteria)
    if result is not None:
        author = "Kobena Amoah"
        print(f"Author : {author}")
        print(f"Selected Sector : {sector}\n")
        # Print the result
        pprint(result)
    else:
        print("Wait for a few Minutes and Re-run the Code")