import numpy as np
import itertools as it
import pickle
import random
from scipy.stats import beta


# %% Functions

class model_softmax:

    def __init__(self, Q, tau, alpha, number_arms):
        self.Q = Q
        self.alpha = alpha
        self.number_arms = number_arms

        self.information = []
        self.tau = tau

        self.p = [np.exp(self.Q)/(np.sum(np.exp(self.Q)))]

    def update(self, choice, R):
        # updates Q values
        self.Q[choice] = self.Q[choice] * (1 - self.alpha) + self.alpha * R
        
    def choice(self):
        # choose arm
        dummy = self.Q/self.tau
        dummy = dummy - max(dummy)
        
        p = np.exp(dummy)/(np.sum(np.exp(dummy)))
        self.p = np.append(self.p, [p], axis=0)

        choice = random.choices(range(0, self.number_arms), weights=p)
        return(choice)

    def save_outcomes(self, choice, outcome):
        dummy = np.full([1, self.number_arms], np.nan)
        dummy[0, choice] = outcome
        self.information = np.append(self.information, dummy)

 

def simulation(strategy, probabilities, observed, obs_information, equal_condition, reward_condition, free_trials, tau, alpha, number_arms=2):    
    '''
    Parameters
    ----------
    strategy : array
        contains the model used for behavioural data
        'softmax' for softmax model
    probabilities: array
        outcome probabilities of each arm
    observed: int
        number of observed outcomes prior to first free choice
    obs_information: list
        distribution of observed trials in unequal condition, e.g.[.9, .1] -> one side 90% of time the other 10%
    equal_condition: boolean
        false: unequal condition in observation
    reward_condition: array
        0 for loss condition
    free_trials: int
        number of free choices after observation 
    tau: float
        individual tau value
    alpha: float
        learning rate 
    number_arms: int
        number of arms of the multi-armed bandit


    Returns
    -------
    model: object
        model used for behavioural data
        includes outcomes & choices
    '''
    
    Q = np.zeros((number_arms))

    if strategy == 'softmax':
        model = model_softmax(Q, tau, alpha, number_arms)

        
    # simulate forced/observed trials
    if equal_condition:
        for i in range(int(observed/number_arms)):
            r = np.random.rand(number_arms)
            outcome = (r < probabilities) * (-1 + reward_condition * 2)

            for ii in range(number_arms):
                model.update(ii, outcome[ii])
                model.save_outcomes(ii, outcome[ii])

    elif not(equal_condition):
        for j in range(number_arms):
            for i in range(int(observed * obs_information[j])):
                
                # simulate observed outcome
                r = np.random.rand()
                outcome = (r < probabilities[j]) * (-1 + reward_condition * 2)
                
                # update Q values
                model.update(j, outcome)
                model.save_outcomes(j, outcome)
                
    # free choices
    for trial in range(free_trials):
        choice = model.choice() # simulate choice
        
        # simulate outcome
        r = np.random.rand() 
        outcome = (r < probabilities[choice]) * (-1 + reward_condition * 2)
        
        # update Q values
        model.update(choice, outcome)
        model.save_outcomes(choice, outcome)

    model.information = model.information.reshape(-1, number_arms)

    return (model)


def sim_softmax(tau_all, data_shocks, prob_all, trials_per_game, observed, number_data_points, obs_information, reward_condition, alpha, number_arms):
    '''
    Parameters
    ----------
    strategy : array
        contains the model used for behavioural data
        'softmax' for softmax model
    tau_all: array
        individual tau parameters
    data_shocks: dict
        for saving data
    prob_all: array of shape (x, 2)  
        combinations of outcome probabilities for each arm    
    trials_per_game: list
        number of trials per game including observed and free choices 
    observed: int
        number of observed outcomes prior to first free choice
    number_data_points: int
        number of games per set of probabilities
    obs_information: list
        distribution of observed trials in unequal condition, e.g.[.9, .1] -> one side 90% of time the other 10%
    reward_condition: array
        0 for loss condition
    alpha: float
        learning rate 
    number_arms: int
        number of arms of the multi-armed bandit
    

    Returns
    -------
    data_shocks: dict
        simulated data
    '''
    
    for prob in prob_all:
        for trials in trials_per_game:
            
            free_trials = trials - observed
            equal_condition = False # unequal condition in observation
            
            for tau in tau_all:
                data_shocks_soft = []
                data_steps_soft = []

                data_shocks['softmax']['{}'.format(tau)]['p:{}, e_c:{}, h:{}'.format(
                    prob, equal_condition, free_trials)] = dict()

                for i in range(number_data_points):
                    
                    # simulate task
                    sim = simulation('softmax', prob, observed, obs_information, equal_condition,
                                     reward_condition, free_trials, tau, alpha, number_arms=number_arms)
                    
                    # save number of shocks and steps
                    data_shocks_soft = np.append(data_shocks_soft, np.nansum(
                        sim.information[-free_trials:, :]) * -1)
                    data_steps_soft = np.append(
                        data_steps_soft, sim.information[-free_trials:, :])
                
                # save number of shocks and steps
                data_shocks['softmax']['{}'.format(tau)]['p:{}, e_c:{}, h:{}'.format(
                    prob, equal_condition, free_trials)]['shocks'] = data_shocks_soft
                data_shocks['softmax']['{}'.format(tau)]['p:{}, e_c:{}, h:{}'.format(
                    prob, equal_condition, free_trials)]['steps'] = data_steps_soft.reshape(number_data_points, free_trials, -1)


    return data_shocks
