# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 08:23:24 2019

@author: Pavan
"""

import numpy as np
import pandas as pd 
from hypothesisEngine.fitness.expression import *
import math as mth
import os

start_date = '2004-01-01'
end_date = '2018-11-15'

abs_path = os.getcwd()
datasets_address = abs_path+"\\datasets\\Engine_Datasets\\"

tickers_file_name = datasets_address+'Final ticker list.xlsx'
tickers_column_name ='Ticker'
variables_file_name = datasets_address+'Variable_list.xlsx'
variables_column_name ='Variable'
delay = 1

portfolio_all = True
other_assets = ['benchmark','rf_rate','Fama_French']

dollar_neutral = True
long_leverage = 0.5
short_leverage = 0.5

starting_value = 20E6
costs_threshold = 0


def excel_list(file_name,column_name,sheet=0):
    List = pd.read_excel(file_name,sheet_name=sheet)
    List = List[column_name].tolist()
    return List

def shift(array, n,fill):
    shifted_array = np.empty_like(array)
    if n >= 0:
        shifted_array[:n,:] = fill
        shifted_array[n:,:] = array[:-n,:]
    else:
        shifted_array[n:,:] = fill
        shifted_array[:n,:] = array[-n:,:]
    return shifted_array

def shift_array(array,n,fill):
    shifted_array = np.empty_like(array)
    if n >= 0:
        shifted_array[:n] = fill
        shifted_array[n:] = array[:-n]
    else:
        shifted_array[n:] = fill
        shifted_array[:n] = array[-n:]
    return shifted_array

variable_list    =  excel_list(variables_file_name,variables_column_name)

address=datasets_address

def get_clean_index(address,start_date,end_date):
    sample_address = address+"sample.csv"
    sample = pd.read_csv(sample_address,index_col=0)
    sample.index = pd.to_datetime(sample.index)
    sample = sample[sample.index.isin(pd.date_range(start=start_date,end=end_date))]
    sample = sample.dropna(axis=0,how='all')
    
    diff_dates = pd.DataFrame(index=sample.index)
    diff_dates['index']= diff_dates.index.astype('datetime64[ns]') 
    diff_dates['Diff'] = (diff_dates['index'] - diff_dates['index'].shift())
    tickers_universe = list(sample.columns)
    return sample.index, diff_dates['Diff'].values, tickers_universe

clean_index,diff_dates,tickers_universe = get_clean_index(datasets_address,start_date,end_date)
tickers = tickers_universe[50:75]

def get_portfolio(portfolio_all,tickers_universe,tickers):
    if portfolio_all:
        return tickers_universe
    else:
        return tickers

portfolio = get_portfolio(portfolio_all,tickers_universe,tickers)
    

def load_datasets(address,other_assets,variable_list,clean_index,portfolio,tickers_universe):
    datasets_dict = {}
    for var in variable_list:
        var_address=address+var+".csv"
        var_df=pd.read_csv(var_address,index_col=0)
        if var not in other_assets:
#            print(var)
            var_df.columns = tickers_universe #white spaces issues; hence this
            var_df = var_df[portfolio]
        var_df.index = pd.to_datetime(var_df.index)
        var_df = var_df[var_df.index.isin(clean_index)]
        var_df.fillna(method='ffill',axis=0,inplace=True)
        var_array=var_df.values
        if var in other_assets and var !='Fama_French':
            var_array = var_array[:,6]
        datasets_dict[var]=var_array
    return datasets_dict
        
datasets_dict = load_datasets(datasets_address,other_assets,variable_list,clean_index,portfolio,tickers_universe)

Open_for_ret = shift(datasets_dict['Open'],-1*delay,np.nan)
Close_for_ret = shift(datasets_dict['Close'],-1*delay,np.nan) 
Volume_for_ret = shift(datasets_dict['Volume'],-1*delay,np.nan)
High_for_ret = shift(datasets_dict['High'],-1*delay,np.nan)
Low_for_ret = shift(datasets_dict['Low'],-1*delay,np.nan)



for key, val in datasets_dict.items():
    vars()[key] = val