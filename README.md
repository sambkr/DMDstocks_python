# DMDstocks_python
Using dynamic mode decomposition (DMD) to predict stock prices and execute a daily trading algorithm.

## Description
The daily trading algorithm works as follows:
1. Take some initial capital and distribute it equally among the companies in the sector proxy that we are trading with.
2. For each day, build a DMD model that takes the previous seven days of stock close prices as inputs, and yields predictions for the stock prices on the following eighth day as outputs.
3. Report the best- and worst-performing stocks based on the next-day predictions.
4. Sell the predicted worst-performing stocks up to a user-specified value of the portfolio, and re-invest the cash into the predicted best-performing stocks.
5. Repeat from step 2 until the finish time is reached.

## Assumptions
* Eight companies have been chosen to represent a proxy of the retail sector, and all trading is conducted between these companies.
* There are no withdrawals or new injections of cash, all money generated is re-invested.
* No transaction costs have been accounted for.
* Stocks are treated as continuous quanitites.
* The number of past days (the variable mp) used to predict one future day was chosen to be 7, based on the heatmaps presented in Mann and Kutz (2016). This paramter has yet to be optimised and applied to other sector types.

## Getting Started
data? 

## Contributor
Samuel Baker
samuel.baker@balliol.ox.ac.uk

## Acknowledgments and references
* The code for the DMD calculation is modified from Kutz and Brunton, available here: https://github.com/dynamicslab/databook_python/blob/master/CH07/CH07_SEC02_DMD_Cylinder.ipynb
* The idea for this application of DMD was inspired by the work from Mann and Kutz (2016), available here: https://doi.org/10.1080/14697688.2016.1170194
* The historical stock price data was accessed by courtesy of Evan Hallmark, available here: https://www.kaggle.com/datasets/ehallmar/daily-historical-stock-prices-1970-2018?select=historical_stock_prices.csv
* The data for the S&P retail index was downloaded from the S&P website, available here: https://www.spglobal.com/spdji/en/indices/equity/sp-retail-select-industry-index/#overview
