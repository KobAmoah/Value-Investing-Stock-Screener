![GitHub all releases](https://img.shields.io/github/downloads/KobAmoah/Fundamental-Stock-Screener/total)
![GitHub language count](https://img.shields.io/github/languages/count/KobAmoah/Fundamental-Stock-Screener) 
![GitHub top language](https://img.shields.io/github/languages/top/KobAmoah/Fundamental-Stock-Screener?color=yellow) 
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/KobAmoah/Fundamental-Stock-Screener)
![GitHub forks](https://img.shields.io/github/forks/KobAmoah/Fundamental-Stock-Screener?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/KobAmoah/Fundamental-Stock-Screener?style=social)


# Enhanced-Value-Investing-Stock-Screener
The Enhanced Value Investing Stock Screening Model integrates Decision Tree Regression and Data Envelopment Analysis to systematically identify potentially undervalued stocks based on their financial performance and market valuation.

## Introduction:
The Enhanced Stock Screening Model integrates Decision Tree Regression (DTR) and Data Envelopment Analysis (DEA) to identify potentially undervalued stocks. DTR assesses stocks based on their returns and financial metrics, while DEA evaluates their efficiency relative to market valuation. This approach offers a systematic method for investors to uncover investment opportunities in the stock market.

** The application provided in the repository is on the Information Technology sector. To apply the algorithm to another sector, modify the inputs in the main file FundamentalScreener.py. Make sure to refresh the jupyter notebook as well to get the newly produced DTR.

## Usage:

The following provides instructions on setting up your project locally. To get a local copy up and running, follow these steps.
1. Clone the repo:

````git clone https://github.com/KobAmoah/Fundamental-Stock-Screener.git````

2. Compile and run the program on the terminal :

````python FundamentalScreener.py````

### Methodology:
#### Decision Tree Regression (DTR):
DTR involves segmenting predictor variables into distinct regions to predict stock returns. It starts with all observations in a single region and iteratively splits data into smaller regions to minimize the Residual Sum of Squares (RSS). The tree is pruned to prevent overfitting, using a Grid Search Cross-Validation to optimize hyperparameters.

#### Data Envelopment Analysis (DEA):
DEA calculates the efficiency of each stock by comparing its financial metrics to market valuation. It formulates an optimization problem to find the best weights for inputs and outputs, maximizing efficiency subject to constraints. Linear Dynamic Programming is used to solve this optimization problem.

### Inputs and Outputs:
DTR Inputs: Beta, operating margin, profit margin, revenue per share, return on assets, return on equity, EPS, revenue growth, leverage ratio, Trailing P/E, forward P/E, EV/Sales, EV/EBIT, P/BV, PEG, P/sales.

DTR Outputs: Total Returns

DEA Inputs: Beta, operating margin, profit margin, revenue per share, return on assets, return on equity, EPS, revenue growth, leverage ratio.

DEA Outputs: Trailing P/E, forward P/E, EV/Sales, EV/EBIT, P/BV, PEG, P/sales.

## Application:
- The model is applied to all stocks in the market.
- Stocks with the lowest efficiency according to DEA are considered potentially undervalued.
- Further analysis is conducted on these stocks to assess their fundamental factors and investment potential.

## Conclusion:
By combining DTR and DEA, the model provides investors with a powerful tool to identify undervalued stocks systematically. This approach enhances decision-making in fundamental investing by leveraging quantitative techniques to uncover investment opportunities that may have been overlooked.

## Credits
I am incredibly grateful to Hogan Tong,CFA . His initial application of DEA as a stock screening tool inspired this project.

