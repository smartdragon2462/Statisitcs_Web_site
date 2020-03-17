# #from utilities.algorithm.general import check_python_version
# #
# #check_python_version()
#
# from hypothesisEngine.stats.stats import get_stats
# from hypothesisEngine.algorithm.parameters import params, set_params
# import sys
# #sys.stdout=open("test.txt","a")
# import warnings
# warnings.filterwarnings("ignore")
#
# class Transcript(object):
#
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.logfile = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.logfile.write(message)
#
#     def flush(self):
#         pass
#
# def start(filename):
#     """Start transcript, appending print output to given filename"""
#     sys.stdout = Transcript(filename)
#
# def stop():
#     """Stop transcript and return print functionality to normal"""
#     sys.stdout.logfile.close()
#     sys.stdout = sys.stdout.terminal
#
# from datetime import datetime
# # datetime object containing current date and time
# now = datetime.now()
# dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
#
# start('log_'+dt_string+'.txt')
#
#
# params['GRAMMAR_FILE'] = "trading_grammar/trading_engine.bnf"
# params['FITNESS_FUNCTION']="trading_fitness.trading_engine"
#
# params['MAX_INIT_TREE_DEPTH'] = 10
# params['MIN_INIT_TREE_DEPTH'] = 3
#
# params['INIT_GENOME_LENGTH']=200
# params['INTERACTION_PROBABILITY'] = 0.5
#
# params['MAX_TREE_DEPTH']=10
# params['POPULATION_SIZE']=10
#
# params['SELECTION_PROPORTION']=0.25
#
# params['GENERATIONS']=5
# params['GENERATION_SIZE']=3
# params['ELITE_SIZE']=1
# params['CROSSOVER_PROBABILITY']=0.4
#
# params['MUTATION_EVENTS'] = 1
# params['MUTATION_PROBABILITY']=0.1
#
#
#
# params['TOURNAMENT_SIZE']=2
#
#
#
# set_params(sys.argv[1:])
#
# individuals = params['SEARCH_LOOP']()
#
# # Print final review
# get_stats(individuals, end=True)
#
# stop()