#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 2024

This program scrapes information from the statistics tab on Yahoo Finance for an input ticker.

@author: Kobena Amoah
"""
import re
import requests
import pandas as pd
import random
import yfinance as yf
import urllib3
from random import randint
from fake_useragent import UserAgent
from fp.fp import FreeProxy
from lxml.etree import ParserError

class YahooFinanceScraper:
    """
    A class for scraping financial information from Yahoo Finance.

    Attributes:
        sector (str): The sector to use for data retrieval. Default is "Technology".
        headers (dict): Headers for HTTP request. Default is Mozilla user-agent.
    """

    def __init__(self):
        """
        Initializes the YahooFinanceScraper object.

        Parameters:
            headers (dict): Headers for HTTP request. Default is Mozilla user-agent.
        """ 
        self.ua = UserAgent()
         
    def get_iplist(self):
        """
        Generates an ip_list to prevent blocking from webscraping.

        Returns:
            list : List containing ip addresses.
        """
        ip_list = []
        for i in range(300):  
            proxy = FreeProxy(country_id=['US', 'GB',"CA","NL","FR","DE","JP","KR","SG","AU"],timeout=1,https=True).get()
            ip_list.append(proxy)
        ip_list = list(set(ip_list))
        return ip_list
    
    def get_stats(self,ticker,ip_list):
        """
        Scrapes information from the statistics tab on Yahoo Finance for an input ticker.

        Args:
            ticker (str): Ticker symbol for the company.
            headers (dict): Headers for HTTP request. Default is Mozilla user-agent.

        Returns:
            pd.DataFrame: DataFrame containing statistics information.
        """

        url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
        header_value = self.ua.random
        random_header = header_value
        # Create a dictionary with the 'User-Agent' header set to random_header
        headers = {'User-Agent': random_header}
        
        # Shuffle the proxy list to ensure random order
        random.shuffle(ip_list)
        # Cyclic generator to yield proxies without repetition
        def proxy_generator():
            for proxy in ip_list:
                yield {"http": proxy, "https": proxy}
        
        proxy_gen = proxy_generator()
        proxy = next(proxy_gen)
        
        max_retries = 5  # Maximum number of retries before returning None
        retry_count = 0
        # Check for successful connection. Otherwise try again
        while retry_count < max_retries:
            try:
                # Make the request with the current proxy
                response = requests.get(url, headers=headers, proxies=proxy)

                # Check the response status code
                if response.status_code == 200:
                    tables = pd.read_html(response.text)
                    tables = [table for table in tables[1:] if table.shape[1] == 2]

                    if tables:
                        # Concatenate all tables into a single DataFrame
                        table = pd.concat(tables)
                        table.columns = ["Attribute", "Value"]
                        return table.reset_index(drop=True)
                    else:
                        return None  # No tables found

                elif retry_count == max_retries - 1:
                    # If it's the last retry and the response is still not successful, return None
                    return None

            except (requests.exceptions.RequestException, requests.exceptions.HTTPError,
                 requests.exceptions.ConnectionError, ValueError, requests.exceptions.ProxyError, urllib3.exceptions.NewConnectionError, urllib3.exceptions.MaxRetryError,ParserError) as e:
                # Handle request exceptions by rotating proxy and retrying
                proxy = next(proxy_gen)
                retry_count += 1

        # If max retries exceeded and no valid response was obtained, return None
        return None
    
    def get_stats_valuation(self,ticker,ip_list):
        """
        Scrapes Valuation Measures table from the statistics tab on Yahoo Finance for an input ticker.

        Args:
            ticker (str): Ticker symbol for the company.
            headers (dict): Headers for HTTP request. Default is Mozilla user-agent.

        Returns:
            pd.DataFrame: DataFrame containing valuation measures information.
        """

        url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
        header_value = self.ua.random
        random_header = header_value
        # Create a dictionary with the 'User-Agent' header set to random_header
        headers = {'User-Agent': random_header}
        
        # Shuffle the proxy list to ensure random order
        random.shuffle(ip_list)
        # Cyclic generator to yield proxies without repetition
        def proxy_generator():
            for proxy in ip_list:
                yield {"http": proxy, "https": proxy}
        
        proxy_gen = proxy_generator()
        proxy = next(proxy_gen)
        
        max_retries = 5  # Maximum number of retries before returning None
        retry_count = 0

        # Check for successful connection. Otherwise try again
        while retry_count < max_retries:
            try:
                # Make the request with the current proxy
                response = requests.get(url, headers=headers, proxies=proxy)

                # Check the response status code
                if response.status_code == 200:
                    tables = pd.read_html(response.text)
                    tables = [table for table in tables if "Trailing P/E" in table.iloc[:, 0].tolist()]
                    table = tables[0].reset_index(drop=True)
                    return table
                else:
                    # If the response status code is not 200 and it's the last retry, return None
                    if retry_count == max_retries - 1:
                        return None

                    # Rotate to the next proxy and retry
                    proxy = next(proxy_gen)
                    retry_count += 1
                    
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError, 
                    requests.exceptions.ConnectionError, ValueError, requests.exceptions.ProxyError, urllib3.exceptions.NewConnectionError, urllib3.exceptions.MaxRetryError,ParserError) as e:
             
                # Handle request exceptions by rotating proxy and retrying
                proxy = next(proxy_gen)
                retry_count += 1

        # If max retries exceeded and no valid response was obtained, return None
        return None