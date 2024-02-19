#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  10 2024
This program was originally created by Hogan Tong to identify cheap stocks. I added documentation to his code.

His code is available at
https://github.com/HoagieT/Stock-Screening-Model-Based-On-Data-Envelopment-Analysis/blob/main/stock_screen.py

@author: Hogan
@user: Kobena Amoah
"""

import pulp as p
import pandas as pd
import numpy as np
  
class DEA:
    """
    A class for solving Data Envelopment Analysis (DEA) problems.

    Attributes:
        inputs (DataFrame): Input data for the DEA model.
        outputs (DataFrame): Output data for the DEA model.
        units_number (int): Number of decision-making units (DMUs) in the DEA model.
        inputs_number (int): Number of input variables in the DEA model.
        outputs_number (int): Number of output variables in the DEA model.
        _i (range): Range object for DMU indices.
        _j (range): Range object for input variable indices.
        _k (range): Range object for output variable indices.
        DMU (dict): Dictionary containing LP problems for each DMU.
    """

    def __init__(self, inputs, outputs):
        """
        Initializes the DEA object.

        Parameters:
            inputs (DataFrame): Input data for the DEA model.
            outputs (DataFrame): Output data for the DEA model.
        """
        if len(inputs.index) != len(outputs.index):
            return print("Error: Number of units inconsistent")

        self.inputs = inputs
        self.outputs = outputs
        
        self.units_number = len(inputs.index)
        self.inputs_number = len(inputs.columns)
        self.outputs_number = len(outputs.columns)
        
        self._i = range(self.units_number)
        self._j = range(self.inputs_number)
        self._k = range(self.outputs_number)
        
        self.DMU = self.create_problems()

    def create_problems(self):
        """
        Creates LP problems for each DMU.

        Returns:
            dict: Dictionary containing LP problems for each DMU.
        """
        DMU = {}
        for i in self._i:
            DMU[i] = self.create_dmu(i)
        return DMU
    
    def create_dmu(self, j0):
        """
        Creates an LP problem for a specific DMU.

        Parameters:
            j0 (int): Index of the DMU.

        Returns:
            LpProblem: LP problem for the specified DMU.
        """
        problem = p.LpProblem('DMU_'+str(j0), p.LpMaximize)
        self.input_weights = p.LpVariable.dicts("input_weights", ((i, j) for i in self._i for j in self._j), lowBound=0, cat='Continuous')
        self.output_weights = p.LpVariable.dicts("output_weights", ((i, k) for i in self._i for k in self._k), lowBound=0, cat='Continuous')
        
        # Objective function
        problem += p.LpAffineExpression([(self.output_weights[(j0,k)], self.outputs.values[(j0,k)]) for k in self._k])
        
        # Constraints
        problem += p.LpAffineExpression([(self.input_weights[(j0,i)],self.inputs.values[(j0,i)]) for i in self._j]) == 1, "Norm Constraint"
        
        for j1 in self._i:
            problem += self.dmu_constraint(j0, j1)  <= 0, "DMU_constraint_"+str(j1)
        
        return problem
    
    def dmu_constraint(self, j0, j1):
        """
        Creates a constraint for the specified DMU pair.

        Parameters:
            j0 (int): Index of the first DMU.
            j1 (int): Index of the second DMU.

        Returns:
            LpAffineExpression: LP affine expression representing the constraint.
        """
        Out = p.LpAffineExpression([(self.output_weights[(j0,k)], self.outputs.values[(j1,k)]) for k in self._k])
        In = p.LpAffineExpression([(self.input_weights[(j0,i)], self.inputs.values[(j1,i)]) for i in self._j])
        
        return Out - In
        
    def solve(self):
        """
        Solves the DEA model.

        Returns:
            dict: Dictionary containing the status of each DMU's optimization problem.
            dict: Dictionary containing the weights of each DMU's input and output variables.
            DataFrame: Efficiency scores for each DMU.
        """
        status = {}
        weights = {}
        efficiency = pd.DataFrame(data=np.nan, index=self.inputs.index, columns=['Efficiency'])
        
        for i, problem in list(self.DMU.items()):
            problem.solve()
            status[i] = p.LpStatus[problem.status]
            weights[i] = {}
            
            for j in problem.variables():
                weights[i][j.name] = j.varValue
            efficiency.iloc[i,0] = p.value(problem.objective)
        
        return status, weights, efficiency
