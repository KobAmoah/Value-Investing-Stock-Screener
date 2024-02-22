#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024
# The basic idea is captured in the code written by Hoagie T available at
# https://github.com/HoagieT/Stock-Screening-Model-Based-On-Data-Envelopment-Analysis/blob/main/stock_screen.py

I make significant changes to Hoagie's code to make it more compatible with the OOP paradigm

This program sets up a class to extract valuation information for a given list of stocks from Yahoo Finance

@author: Kobena Amoah
"""    
import numpy as np
import pandas as pd
import yfinance as yf
import re
from yahoo_fin import stock_info as si

class DataExtraction:
    """
    A class for extracting financial data from Yahoo Finance for a given list of stocks.
    """

    def __init__(self, stock_list, operating_names, valuation_names):
        """
        Initializes the DataExtraction object.

        Parameters:
            stock_list (list): List of stock tickers.
            operating_names (list): List of operating statistic names.
            valuation_names (list): List of valuation statistic names.
        """
        self.stock_list = stock_list
        self.operating_names = operating_names
        self.valuation_names = valuation_names
        self.operating_stats = pd.DataFrame(index=stock_list, columns=operating_names)
        self.valuation_stats = pd.DataFrame(index=stock_list, columns=valuation_names)

    def extract_operating_stats(self,data_range =list(range(1)) + [-1] ):
        """
        Extracts operating statistics for each stock in the stock_list and populates the operating_stats DataFrame.
        """
        for ticker in self.stock_list:
            temp = si.get_stats(ticker)
            temp = temp.iloc[:, data_range]
            temp.columns = ["Attribute", "Recent"]
            # Include only strings in operating_names
            strings_to_include = self.operating_names
            temp = temp[temp.apply(lambda row: any(s in str(cell) for cell in row for s in strings_to_include), axis=1)]
            temp["Recent"] = temp["Recent"].apply(lambda x: self.parse_value(x))
            temp = temp.fillna(0).pivot_table(values='Recent', columns='Attribute').reset_index(drop=True)
            
            # Remove anything within parentheses using regular expressions
            column_names = temp.columns
            column_names_cleaned = [re.sub(r'\([^()]*\)', '', column) for column in column_names]
            # Strip leading and trailing whitespaces
            column_names_cleaned = [column.strip() for column in column_names_cleaned]
            temp.columns = column_names_cleaned       
            # Find common columns between self.operating_stats and temp
            common_columns = self.operating_stats.columns.intersection(temp.columns)
            # Subset columns present in temp from self.operating_stats 
            temp = temp[common_columns] 
            self.operating_stats.loc[ticker, :] = temp.values[0]
            
                  
    def extract_valuation_stats(self,data_range =list(range(1)) + [-1] ):
        """
        Extracts valuation statistics for each stock in the stock_list and populates the valuation_stats DataFrame.
        """
        for ticker in self.stock_list:
            temp = si.get_stats_valuation(ticker)
            temp = temp.iloc[:, data_range]
            temp.columns = ["Attribute", "Recent"]
            # Include only strings in valuation_names
            strings_to_include = self.valuation_names
            temp = temp[temp.apply(lambda row: any(s in str(cell) for cell in row for s in strings_to_include), axis=1)]
            temp["Recent"] = temp["Recent"].apply(lambda x: self.parse_value(x))
            temp = temp.fillna(0).pivot_table(values='Recent', columns='Attribute').reset_index(drop=True)
            # Remove anything within parentheses using regular expressions
            column_names = temp.columns
            column_names_cleaned = [re.sub(r'\([^()]*\)', '', column) for column in column_names]
            # Strip leading and trailing whitespaces
            column_names_cleaned = [column.strip() for column in column_names_cleaned]
            temp.columns = column_names_cleaned
            # Find common columns between self.valuation_stats and temp
            common_columns = self.valuation_stats.columns.intersection(temp.columns)
            # Subset columns present in  temp from self.valuation_stats 
            temp = temp[common_columns] 
            self.valuation_stats.loc[ticker, :] = temp.values
            
    def parse_value(self, value_str):
        """
        Function to parse string representations of values with magnitude suffixes (e.g., T for trillion, B for billion).

        Parameters:
            value_str (str): String representation of the value.

        Returns:
            float: Numerical value.
        """
        if isinstance(value_str, float):
            return value_str
        else:
            value_str = str(value_str)  # Convert to string if not already
            value_str = value_str.replace(',', '')  # Remove commas

            if 'T' in value_str:
                return float(value_str.replace('T', '')) * 1e12
            elif 'B' in value_str:
                return float(value_str.replace('B', '')) * 1e9
            elif 'M' in value_str:
                return float(value_str.replace('M', '')) * 1e6
            elif 'K' in value_str:
                return float(value_str.replace('K', '')) * 1e3
            elif 'k' in value_str:
                return float(value_str.replace('k', '')) * 1e3
            elif '%' in value_str:
                return float(value_str.replace('%', ''))
            else:
                return float(value_str)
      

    def get_period_returns(self, start_date, end_date):
        """
        Computes the period returns for each stock in the stock_list.

        Parameters:
            start_date (str): Start date of the period in the format 'YYYY-MM-DD'.
            end_date (str): End date of the period in the format 'YYYY-MM-DD'.

        Returns:
            DataFrame: A DataFrame containing period returns for each stock.
        """
        returns_data = []

        for ticker in self.stock_list:
            # Retrieve historical price data
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress= False ).dropna()
            
            # Check if the stock data has history starting from the start_date
            if len(stock_data) > 0:
                # Compute period returns
                log_return = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift())
                log_return = log_return[~np.isnan(log_return)]
                period_return = np.cumprod(1 + np.array(log_return)) - 1
                period_return = np.multiply(period_return,100)
                period_return = period_return.tolist()
                returns_data.append({'Period_Return': period_return[-1], 'Ticker': ticker})
    
        returns_df = pd.DataFrame(returns_data)
        return returns_df
