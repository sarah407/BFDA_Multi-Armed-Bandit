
import numpy as np

from data_simulate import *
from data_sort import *

#%% Functions

def sample(n_e, number_data_points, prob_all, observed, trials_per_game, obs_information, model, horizons, tau_mean, alpha = 0.1):
    
    '''
    Parameters
    ----------
    n_e: int
        sample size to be simulated
    number_data_points: int
        number of games per set of probabilities
    prob_all: array of shape (x, 2)  
        combinations of outcome probabilities for each arm
    observed: int
        number of observed outcomes prior to first free choice
    trials_per_game: list
        number of trials per game including observed and free choices
    obs_information: list
        distribution of observed trials in unequal condition, e.g.[.9, .1] -> one side 90% of time the other 10%
    model : array
        contains the model used for behavioural data
        'softmax' for softmax model
    horizons: array
        number of free choice trials
    tau_mean: float
        population mean
    alpha: float
        learning rate
    

    Returns
    -------
    data_shocks: dict
        simulated data
    list_combinations: array
        list of simulated combinations of outcome probabilities, equal condition and horizon
    real_mean: float
        mean of all individual taus
    '''
    
    reward_condition = np.array(0) # 0: loss condition, outcomes will be negative
    
    data_shocks = dict()
    data_shocks[model[0]] = dict()
    
    for h in range(len(horizons)):
        data_shocks['all_{}_h{}'.format(model[0], horizons[h])] = []

        
    if (model == 'softmax'):
        
        # Generate individual tau parameters from population
        tau_all = np.random.normal(loc = tau_mean, scale = 0.02, size = n_e)
        while np.any(tau_all <= 0):
            tau_all[np.where(tau_all <= 0)] = np.random.normal(loc = tau_mean, scale = 0.02, size = np.sum(tau_all <= 0))
         
        
        for tau in tau_all:
            data_shocks['softmax']['{}'.format(tau)] = dict()  

        data_shocks = sim_softmax(tau_all, data_shocks, prob_all, trials_per_game, observed, number_data_points, obs_information, reward_condition, alpha, number_arms = 2)
        data_shocks = sort_softmax(data_shocks, horizons)
        
        
        real_mean = np.mean(tau_all)
        
    
    v = list(data_shocks[model[0]].keys())

    list_combinations = [['p_a', 'p_b', 'e_c', 'h', 'combination']]
    for combination in list(data_shocks[model[0]][v[0]]):
        p = re.search('(?<=p:\[).*(?=\])', combination)[0]
        p_a = p.split()[0]
        p_b = p.split()[1]
        e_c = re.search('(?<=e_c:).*(?=,)', combination)[0]
        h = re.search('(?<=h:).*', combination)[0]
        list_combinations = np.append(list_combinations, [[p_a, p_b, e_c, h, combination]], axis = 0)
    
    return (data_shocks, list_combinations, real_mean)
    