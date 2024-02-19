#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024

This program sets up a class to download a list stocks on an index from Yahoo Finance along with their sector.
@author: Kobena Amoah
"""
import pandas as pd

class RussellSectorComponents:
    """
    A class for retrieving sector information for Russell 1000 Sector components.

    Attributes:
        sector (str): The sector to use for data retrieval. Default is "Technology".
    """

    def __init__(self, sector = "Technology"):
        """
        Initializes the RussellSectorComponents object.

        Parameters:
            sector (str): Sector name to assign to the retrieved components.
            server (str): The Yahoo Finance server to use for data retrieval. Default is "download".
            sep (str): Separator used for parsing the response data. Default is ",".
        """
        self.sector = sector

    def russell_components_with_sector(self):
        """
        Retrieves Russell 1000 components along with their corresponding sectors.

        Returns:
            pd.DataFrame: A DataFrame containing components and their corresponding sectors.
        """
        # Compose URL
        url = f"http://en.wikipedia.org/wiki/Russell_1000_Index"
        
        russell = pd.read_html(url)[2]
        russell["Ticker"] = russell["Ticker"].str.replace(".", "-")
        russell.rename(columns={"GICS Sector": "Sector"}, inplace=True)
        russell = russell[russell.Sector == self.sector]
            
        return russell.Ticker