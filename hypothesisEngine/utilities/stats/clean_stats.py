from hypothesisEngine.stats.stats import stats
from hypothesisEngine.algorithm.parameters import params


def clean_stats():
    """
    Removes certain unnecessary stats from the stats.stats.stats dictionary
    to clean up the current run.
    
    :return: Nothing.
    """
    if not stats.keys().__contains__('www'):
        stats.pop('www')

    if not params['CACHE'] and stats.keys().__contains__('unique_inds') and stats.keys().__contains__('unused_search'):
        stats.pop('unique_inds')
        stats.pop('unused_search')
    
    if not params['MUTATE_DUPLICATES'] and stats.keys().__contains__('unique_inds'):
        stats.pop('regens')
