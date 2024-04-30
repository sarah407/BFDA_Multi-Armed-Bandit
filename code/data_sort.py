#%% 
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import pickle
import os 
import re
import random

from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline


#%% Main


def sort_softmax(data_shocks, horizons):
    '''
    Parameters
    ----------
    data_shocks: dict
        simulated data
    free_trials: int
        number of free choices after observation
    

    Returns
    -------
    data_shocks: dict
        simulated data, sorted
    '''
    for tau in data_shocks['softmax'].keys():
        data_shocks['all_tau_{}'.format(tau)] = dict()
        
        for h in range(len(horizons)):
            data_shocks['all_tau_{}'.format(tau)]['softmax_h{}'.format(horizons[h])] = []
        
        for k in data_shocks['softmax'][tau].keys():
            horizon =  re.search('h:(.*)',k).group(1)
            
            for h in range(len(horizons)):
                if horizon == str(horizons[h]):
                    data_shocks['all_softmax_h{}'.format(horizons[h])] = np.append(data_shocks['all_softmax_h{}'.format(horizons[h])], data_shocks['softmax'][tau][k]['shocks'])
                    data_shocks['all_tau_{}'.format(tau)]['softmax_h{}'.format(horizons[h])] = np.append(data_shocks['all_tau_{}'.format(tau)]['softmax_h{}'.format(horizons[h])], data_shocks['softmax'][tau][k]['shocks'])
    return(data_shocks)
