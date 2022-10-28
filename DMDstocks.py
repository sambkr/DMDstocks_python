#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Load data
data = pd.read_csv('historical_stock_prices.csv')

#%% Set parameters and reduce table size
# Choose some dates
start_date = '2014-03-18'
end_date = '2015-03-18'

after_start_date = data['date'] >= start_date
before_end_date = data['date'] <= end_date
between_two_dates = after_start_date & before_end_date

tabledates = data.loc[between_two_dates]

# Choose tickers
s1 = 'AEO'
s2 = 'ANF'
s3 = 'FL'
s4 = 'GPS'
s5 = 'SCVL'
s6 = 'RL'
s7 = 'URBN'
s8 = 'ROST'

tickers = [s1,s2,s3,s4,s5,s6,s7,s8]
portfolio_size = len(tickers)

# Get retail_table in the specified date range
reduced_table = []
for i in tickers:
    ticker_loc = tabledates['ticker'] == i
    reduced_table.append(tabledates.loc[ticker_loc])

retail_table = pd.concat(reduced_table)

# Form the big data matrix.
# For each ticker, get all the close prices and store.
days = len(retail_table[retail_table['ticker'] == tickers[0]])
bigX = np.zeros(shape=(portfolio_size,days))
for i in range(0,portfolio_size):
    temp = retail_table[retail_table['ticker'] == tickers[i]]
    temp_price_vector =  temp['close'].values.tolist()
    bigX[i,:] = temp_price_vector
    
#%% Initialise the trading
# Number of past days to build the DMD model on
mp = 7
# Number of future days to predict with DMD
mf = 1

# Percentage of portfolio to sell off each day
sell_perc = 0.25

# Initialise at day 7, as DMD uses data on the previous 7 days to predict
# the price on the following day
current_day = 7

# Initialise capital and date
init_cap = 1e6
init_each = 1e6/portfolio_size
init_day = datetime.datetime.strptime(start_date,'%Y-%m-%d') + datetime.timedelta(days = (mp-1))

day_close = GetPrices(portfolio_size, bigX, current_day)

# Evenly distribute stock
stock_amounts = np.zeros(shape=(portfolio_size,1))
for i in range(0,portfolio_size):
    stock_amounts[i,0] = init_each/day_close[i]
    
#%% The trading
# Initialise portfolio value over time
valuet = np.zeros(shape=(1,days))

# Trade
for i in range(0,days-mp-1):
    stock_amounts, day_close, current_day = Trade(current_day, mp, mf, bigX, portfolio_size, stock_amounts, day_close, sell_perc);

    # Calculate value of portfolio and store in valuet
    value = np.sum(stock_amounts*day_close)
    valuet[0,i] = value
    
#%% Load S&P data
SP = pd.read_csv('S&Pretail_reduced.csv')

#%% Average returns
returnDMD = valuet[0,0:days-(mp+mf)] - 1e6
avreturnDMD = np.mean(returnDMD)
returnSP = SP['close'][0:days-(mp+mf)] - 1e6
avreturnSP = np.mean(returnSP)
DMDperformance = avreturnDMD/avreturnSP

#%% Functions

def GetPrices(portfolio_size, bigX, current_day):
    # Find prices on a given day
    day_close = np.zeros(shape=(portfolio_size,1))
    for i in range(0,portfolio_size):
        day_close[i,0] = bigX[i,current_day-1]
    
    return day_close

def Trade(current_day, mp, mf, bigX, portfolio_size, stock_amounts, day_close, sell_perc):
    
    first_day = current_day - (mp-1)
    
    # Time vector spans mp+mf, DMD will extrapolate to make a prediction about mf
    t = list(range(first_day,mp+first_day+1))

    # Form the DMD matrices
    X1 = bigX[:,(first_day-1):(current_day-1)]
    X2 = bigX[:,(first_day):current_day]

    # Snapshots separated by 1 trading day
    dt = 1

    # Conduct DMD
    Phi, b, omega = DMD(X1, X2, dt)

    # DMD reconstruction to predict price on current_day + 1
    price_predictions = DMDreconstruct(X1, t, b, omega, Phi, mp, mf)

    # Calculate increases in price between current_day and the following day
    price_increases = np.zeros(shape=(portfolio_size,1))
    for i in range(0,portfolio_size):
        price_increases[i,0] = (price_predictions[i] - bigX[i,current_day-1])/bigX[i,current_day-1]

    # Calculate current portfolio value
    portfolio_value = np.zeros(shape=(portfolio_size,1))
    for i in range(0,portfolio_size):
        portfolio_value[i,0] = stock_amounts[i,0]*day_close[i,0]

    # Sell bottom 25% of portfolio
    cash, stock_amounts = Sell(portfolio_value, sell_perc, price_increases, portfolio_size, stock_amounts, day_close)

    # Buy best performing shares with cash from sales.
    stock_amounts = Buy(price_increases, cash, day_close, stock_amounts)

    # Increment day
    current_day += 1
    
    # Get new day_close prices
    day_close = GetPrices(portfolio_size, bigX, current_day)
    
    return stock_amounts, day_close, current_day

def Sell(portfolio_value, sell_perc, price_increases, portfolio_size, stock_amounts, day_close):
    sell_value = np.sum(portfolio_value)*sell_perc
    cash = 0
    lowest = np.sort(price_increases,axis=None)
    for i in range(0,portfolio_size):
    # For each ticker, find location of lowest price in price_increases 
        lowest_value = stock_amounts[price_increases == lowest[i]]*day_close[price_increases == lowest[i]]
        temp_cash = cash + lowest_value
        if temp_cash < sell_value:
            stock_amounts[price_increases == lowest[i]] = 0
            cash = temp_cash
        elif temp_cash == sell_value:
            stock_amounts[price_increases == lowest[i]] = 0
            cash = temp_cash
            break
        else:
            number_sold = (sell_value-cash)/day_close[price_increases == lowest[i]]
            stock_amounts[price_increases == lowest[i]] = stock_amounts[price_increases == lowest[i]] - number_sold
            new_cash = number_sold*day_close[price_increases == lowest[i]]
            cash = new_cash + cash
            break
        
    return cash, stock_amounts

def Buy(price_increases, cash, day_close, stock_amounts):
    
    best = np.sort(price_increases,axis=None)[::-1]
    
    number_bought1 = 0.5*cash/day_close[price_increases == best[0]]
    number_bought2 = 0.5*cash/day_close[price_increases == best[1]]    
    
    stock_amounts[price_increases == best[0]] = stock_amounts[price_increases == best[0]] + number_bought1
    stock_amounts[price_increases == best[1]] = stock_amounts[price_increases == best[1]] + number_bought2
    
    return stock_amounts

def DMD(X1, X2, dt):
    # SVD on X1
    U,S,V = np.linalg.svd(X1,full_matrices=0)
    Sigmar = np.diag(S)

    # Calculate Atilde
    Atilde = np.linalg.solve(Sigmar.T,(U.T @ X2 @ V.T).T).T

    # Eigendecomp of Atilde
    Lambda, W = np.linalg.eig(Atilde)
    L = np.diag(Lambda)

    # DMD modes
    Phi = X2 @ np.linalg.solve(Sigmar.T,V).T @ W

    # DMD amplitudes
    alpha1 = Sigmar @ V[:,0]
    b = np.linalg.solve(W @ L,alpha1)

    # Frequency
    omega = np.log(Lambda)/dt
    
    return Phi, b, omega

def DMDreconstruct(X1, t, b, omega, Phi, mp, mf):
    time_dynamics = np.zeros(shape=(X1.shape[1],len(t)),dtype=np.complex128)

    for i in range(0,len(t)):
        time_dynamics[:,i] = np.multiply(b,np.exp(omega*t[i]))

    X_dmd = Phi @ time_dynamics
    price_predictions = np.real(X_dmd[:,(mp)])
    
    return price_predictions


