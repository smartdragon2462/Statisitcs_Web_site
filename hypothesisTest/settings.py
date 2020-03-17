# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 02:22:57 2019

@author: Pavan
"""

# start_date                      = '2004-01-01'
# end_date                        = '2018-11-15'
# datasets_address                = "Datasets\\"
# tickers_file_name               = 'Final ticker list.xlsx'
# tickers_column_name             = 'Ticker'
# variables_file_name             = 'Variable_list.xlsx'
# variables_column_name           = 'Variable'
# delay                           = 1
# portfolio_all                   = True
# other_assets                    = ['benchmark','rf_rate','Fama_French']
# dollar_neutral                  = True
# long_leverage                   = 0.5
# short_leverage                  = 0.5
# starting_value                  = 20E6
# costs_threshold                 = 0
# strategy_expression             = '-rank(Volume)*(gauss_filter(High,5)-gauss_filter(Open,5))'

from .performance import metrics

global start_date
global end_date
global datasets_address
global tickers_file_name
global tickers_column_name
global variables_file_name
global variables_column_name
global delay
global portfolio_all
global other_assets
global dollar_neutral
global long_leverage
global short_leverage
global starting_value
global costs_threshold
global strategy_expression

datasets_address = "datasets\\Test_Datasets\\"
tickers_file_name = 'data\\Final ticker list.xlsx'
tickers_column_name = 'Ticker'
variables_file_name = 'data\\Variable_list.xlsx'
variables_column_name = 'Variable'
portfolio_all                   = True
other_assets                    = ['benchmark','rf_rate','Fama_French']
delay                           = 1

# if __name__ =="__main__":

# def run_hypothesisTest():
#     res = metrics()
#     for key,value in res.items():
#         if key not in ["CLASSIFICATION_DATA","FACTOR_RES"]:
#             print("{:<40}{:^5}{:<20}".format(key," :\t", value))
#
#     print("Classification Metrics : \n")
#     print(res['CLASSIFICATION_DATA'],"\n")
#     print("Factor Analysis : \n")
#     print(res['FACTOR_RES'])
#
#     return res



