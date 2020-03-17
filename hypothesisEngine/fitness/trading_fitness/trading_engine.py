from hypothesisEngine.fitness.base_ff_classes.base_ff import base_ff
from hypothesisEngine.fitness.expression import *
from hypothesisEngine.fitness.fitness_helper import *
import numpy as np
import math as mth
import textwrap
import hypothesisEngine.print_setting as ps

#from utiities.representation import Tee
#import logging
#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s',
#                    filename='logs_file',
#                    filemode='w')
## Until here logs only to file: 'logs_file'
#
## define a new Handler to log to console as well
#console = logging.StreamHandler()
## optional, set the logging level
#console.setLevel(logging.INFO)
## set a format which is the same for console use
#formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
## tell the handler to use this format
#console.setFormatter(formatter)
## add the handler to the root logger
#logging.getLogger('').addHandler(console)

class trading_engine(base_ff):
    """
    Basic fitness function template for writing new fitness functions. This
    basic template inherits from the base fitness function class, which
    contains various checks and balances.
    
    Note that all fitness functions must be implemented as a class.
    
    Note that the class name must be the same as the file name.
    
    Important points to note about base fitness function class from which
    this template inherits:
    
      - Default Fitness values (can be referenced as "self.default_fitness")
        are set to NaN in the base class. While this can be over-written,
        PonyGE2 works best when it can filter solutions by NaN values.
    
      - The standard fitness objective of the base fitness function class is
        to minimise fitness. If the objective is to maximise fitness,
        this can be over-written by setting the flag "maximise = True".
    
    """

    # The base fitness function class is set up to minimise fitness.
    # However, if you wish to maximise fitness values, you only need to
    # change the "maximise" attribute here to True rather than False.
    # Note that if fitness is being minimised, it is not necessary to
    # re-define/overwrite the maximise attribute here, as it already exists
    # in the base fitness function class.
    maximise = True

    def __init__(self):
        """
        All fitness functions which inherit from the bass fitness function
        class must initialise the base class during their own initialisation.
        """
#        log = logging.getLogger(__name__)
        # Initialise base fitness function class.
        super().__init__()
        
    @staticmethod
    def fitness_exp(strategy,params):
        dollar_neutral = params['dollar_neutral']
        long_leverage = params['long_leverage']
        short_leverage = params['short_leverage']
        
        
        def weights(strategy,dollar_neutral=False,long_leverage=0.5,short_leverage=0.5):
            if dollar_neutral:
                exp_ls = strategy-np.nanmean(strategy,axis=1)[:,np.newaxis]
            else:
                exp_ls = strategy
            exp_norm = (exp_ls/np.nansum(abs(exp_ls),axis=1)[:,np.newaxis])
            exp_norm = np.where(exp_norm>=0,2*long_leverage*exp_norm,2*short_leverage*exp_norm)
            return exp_norm
        
        strategy_weights = weights(strategy,dollar_neutral,long_leverage,short_leverage)
        
        empty = np.full_like(strategy_weights,np.nan)    
        check=((strategy_weights == empty) | (np.isnan(strategy_weights) & np.isnan(empty))).all() 
        if check:
            strategy_weights = np.full_like(strategy_weights,0)   
            
        non_nan_frac = np.count_nonzero(~np.isnan(strategy_weights))/np.size(strategy_weights)
        if non_nan_frac<=0.7:
            strategy_weights = np.full_like(strategy_weights,0) 
            
        
        
        def get_valid_index(strategy_weights,delay):
            valid_index = ~np.isnan(strategy_weights).all(axis=1)
            valid_index[-1*delay]=False
            return valid_index
        delay = 1
        clean_values_from_weights = get_valid_index(strategy_weights,delay)
    #    cleaned_index_weights = (clean_index.values)[clean_values_from_weights]
        
    
        
        
        def bets_to_pnl(starting_value,strategy_weights,clean_values,o,h,l,c,long_lev, short_lev):
            
        
            cleaned_weights       = strategy_weights[clean_values]
            if cleaned_weights.shape[0]==0:
                cleaned_weights = strategy_weights
                O = o
                C = c
                H = h
                L = l
        
            O  = (o)[clean_values]
            C = (c)[clean_values]
        #   V   = (V)[clean_values]
            H  = (h)[clean_values]
            L   = (l)[clean_values]
            
            
            dollars_at_open  = np.zeros(cleaned_weights.shape)
            dollars_at_close = np.zeros(cleaned_weights.shape)
            purchased_shares = np.zeros(cleaned_weights.shape)
            costs            = np.zeros(cleaned_weights.shape)
            value_at_open    = np.zeros(cleaned_weights.shape[0])
            value_at_open[0] = starting_value
            value_at_close   = np.zeros(cleaned_weights.shape[0])
            pnl              = np.zeros(cleaned_weights.shape)
            daily_pnl        = np.zeros(cleaned_weights.shape[0])
            long_pnl        = np.zeros(cleaned_weights.shape[0])
            short_pnl        = np.zeros(cleaned_weights.shape[0])
            
        
            for i in range(dollars_at_open.shape[0]):
                dollars_at_open[i,:] = value_at_open[i]*cleaned_weights[i,:]
                purchased_shares[i,:]= dollars_at_open[i,:]/O[i,:]
                dollars_at_close[i,:]= C[i,:]*purchased_shares[i,:]
                costs[i,:]           = np.abs(purchased_shares[i,:])*(H[i,:]-L[i,:])*costs_threshold
                pnl[i,:]             = (dollars_at_close[i,:]-dollars_at_open[i,:]-costs[i,:])
                daily_pnl[i]         = np.nansum(pnl[i,:])
                long_pnl[i]         = np.nansum(pnl[i,:][cleaned_weights[i,:]>0])
                short_pnl[i]        = np.nansum(pnl[i,:][cleaned_weights[i,:]<0])
                value_at_close[i]    = np.nansum(np.abs(dollars_at_open[i,:])/(long_lev+short_lev))+np.nansum(pnl[i])
                if i != dollars_at_open.shape[0]-1:
                    value_at_open[i+1]=value_at_close[i]
            
            strategy_daily_returns = np.nan_to_num(daily_pnl/value_at_open)
            long_contribution = np.nan_to_num(long_pnl/value_at_open)
            short_contribution = np.nan_to_num(short_pnl/value_at_open)
        
            return strategy_daily_returns, long_contribution, short_contribution, costs, purchased_shares,dollars_at_open, dollars_at_close, value_at_open, value_at_close, pnl, daily_pnl
            
        strategy_daily_returns, long_contribution, short_contribution, costs, purchased_shares, dollars_at_open, dollars_at_close, value_at_open, value_at_close, pnl, daily_pnl = bets_to_pnl(starting_value,strategy_weights,clean_values_from_weights,Open_for_ret,High_for_ret,Low_for_ret,Close_for_ret, long_leverage, short_leverage)
        
        
    #  
    #    benchmark_returns = datasets_dict['benchmark'][clean_values_from_weights]
        if np.sum(clean_values_from_weights)!=0:
            rf_returns   = datasets_dict['rf_rate'][clean_values_from_weights]
        else:
            rf_returns   = datasets_dict['rf_rate']
        
    #    cleaned_index = cleaned_index_weights
        
        if np.sum(strategy_daily_returns)==0:
            return 0
        else:
            excess_returns = strategy_daily_returns-rf_returns
            fitness = mth.sqrt(252)*np.mean(excess_returns)/np.std(excess_returns)
            return fitness
    
    def evaluate(self, ind, **kwargs):
        """
        Default fitness execution call for all fitness functions. When
        implementing a new fitness function, this is where code should be added
        to evaluate target phenotypes.
        
        There is no need to implement a __call__() method for new fitness
        functions which inherit from the base class; the "evaluate()" function
        provided here allows for this. Implementing a __call__() method for new
        fitness functions will over-write the __call__() method in the base
        class, removing much of the functionality and use of the base class.
                
        :param ind: An individual to be evaluated.
        :param kwargs: Optional extra arguments.
        :return: The fitness of the evaluated individual.
        """
        
        
#        strategy =eval(ind.phenotype)
        
        params_dict = dict()
        params_dict['dollar_neutral']=True
        params_dict['short_leverage']=0.5
        params_dict['long_leverage']=0.5
        
        # Evaluate the fitness of the phenotype
#        print(ind.phenotype)
        strategy = eval(ind.phenotype)
        nan_frac = np.count_nonzero(~np.isnan(strategy))/np.size(strategy)
        if nan_frac <=0.6:
            fitness = 0
        else:

            string = 'self.fitness_exp(strategy,params_dict)'
    
            fitness = eval(string)
            if fitness == np.inf or fitness == -np.inf:
                fitness = 0
#        logging.basicConfig(format='%(process)d-%(message)s')
#        logging.info("%50s : %10s"%(ind.phenotype,fitness))
                
                



        f = str(round(fitness,3))
        prefix = f + "\t : "
        preferredWidth = 70
        wrapper = textwrap.TextWrapper(initial_indent=prefix, width=preferredWidth,
                                       subsequent_indent=' '*len(prefix))
        message = ind.phenotype

        print(wrapper.fill(message))
        ps.print_list.append(wrapper.fill(message))
        print('\n')

        return fitness
