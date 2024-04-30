
import numpy as np
import pickle
import itertools as it
from time import time

from likelihood_shocks import *
from data_sample import *
from data_simulate import *
from data_sort import *

start = time()

#%% Main Calculations

def main(number_simulations, mean, n_e_all, like, model, trials_per_game, n_indiv, prob_all, observed, obs_information, theta, prior):
    '''

    Parameters
    ----------
    number_simulations: int
        number of simulations for this mean
    mean: float
        group mean for the group
    n_e_all: array of ints
        all sample sizes to be simulated, sample sizes are per group not total
    like: array
        likelihoods as a function of tau, outcome probabilities, and number of shocks
    model : array
        contains the model used for behavioural data
        'softmax' for softmax model
    trials_per_game: list
        number of trials per game including observed and free choices
    n_indiv: int
        number of games per simulated participant
    prob_all: array of shape (x, 2)  
        combinations of outcome probabilities for each arm
    observed: int
        number of observed outcomes prior to first free choice
    observed_information: list
        distribution of observed trials in unequal condition, e.g.[.9, .1] -> one side 90% of time the other 10%
    theta: array 
        values of tau for which the likelihood of receiving a certain number of shocks is calculated
    prior: array
        probability of prior as function of theta

    Returns
    -------
    estimations: array
        estimation of population mean
        
    '''
    estimations = np.zeros((len(n_e_all), number_simulations))
    
    for index, n_e in enumerate(n_e_all):  
        for i_sim in range(number_simulations):
            
            horizon= np.asarray(trials_per_game - observed * np.ones(np.shape(trials_per_game)), dtype = 'int')
            
            # simulate data
            data_shocks, list_combinations, real_mean = sample(n_e, int(n_indiv/len(prob_all)), prob_all, observed, trials_per_game, obs_information, model, horizon, mean)
            
            #________________________Calculating the likelihood for simulated data__________________________________________
            tot_likelihood = np.zeros((len(data_shocks[model[0]].keys()) * n_indiv, np.shape(theta)[0]))
    
            counter = 0
            
            for i_comb, comb in enumerate(list_combinations[1::]):
    
                # Extract parameters
                if bool(comb[2]):
                    #equal condition
                    I = np.array((int(observed/2), int(observed/2)))
                else:
                    I = np.array((int(observed * obs_information[0]), int(observed * obs_information[1])))
    
                p = np.array((float(comb[0]), float(comb[1])))
        
                horizon = int(comb[3])
    
                for i, epsilon in enumerate(data_shocks[model[0]].keys()):
                    number_shocks = data_shocks[model[0]][epsilon][comb[4]]['shocks'].astype(int)
                    for i_indiv, shock in enumerate(number_shocks):
                        tot_likelihood[counter, :] = like[i_comb, shock,:]
                        counter = counter + 1
    
            tot_likelihood = np.sum(np.log10(tot_likelihood), axis = 0)
            tot_likelihood = tot_likelihood + 290 - np.max(tot_likelihood)
            
            #________________________Calculating the estimation for simulated data__________________________________________
            
            Posterior = tot_likelihood + np.log10(prior)
            Posterior = Posterior  + 290 - np.nanmax(Posterior)
            Posterior = 10 ** Posterior            
            Posterior = Posterior/np.nansum(Posterior)
            
            # Calculate estimate
            estimate = np.sum(Posterior * theta)
            estimations[index, i_sim] = estimate
    return(estimations)

#%% Setup

all_trials_per_game = [[10]]
prob_all = [0.1, 0.3, 0.9]
prob_all = np.array(list(it.combinations(np.unique(prob_all), 2)))
prob_all = np.append(prob_all, prob_all[:,::-1]).reshape(-1,2)

models = np.array((['softmax']))

observed = 4
obs_information = [0.25, 0.75] # distribution of observed trials in unequal condition, e.g.[.9, .1] -> one side 90% of time the other 10%
learning_rate = 0.1

I = np.array((int(observed * obs_information[0]), int(observed * obs_information[1])))

n_e_all = np.array(range(10, 60, 2))

trials_per_game = all_trials_per_game[0]
free_choices = trials_per_game[0] - observed


theta = np.linspace(0.01,3,300)
prior =  np.ones(np.shape(theta))/np.shape(theta)

Q_1 = np.array((0,0))

# Calculating the likelihood for all possible outcomes
likelihood = likelihood_shocks(models, I, prob_all, learning_rate, free_choices, theta, Q_1)

print('finished likelihood shocks')

#%% estimations as function of sample size & means

n_indiv = 150
number_simulations = 250
means_all = np.linspace(0.05, 0.1, 20) 

# Simulations

estimations_all = np.zeros((len(means_all), len(n_e_all), int(number_simulations)))

for i_mean, mean in enumerate(means_all):
    start_diff = time()
    print('----------------- \n', mean)    
    estimations = main(number_simulations, mean, n_e_all, likelihood, models,trials_per_game, n_indiv, prob_all, observed, obs_information, theta, prior)     
    estimations_all[i_mean, :, :] = estimations
    end_diff = time()
    print('time in minutes: ', round((end_diff - start_diff)/60, 2), 'in hours:', round((end_diff - start_diff)/3600, 1)) 
end = time()

#%% Save Data
print('=========================\ntotal time in minutes: ', round(( end - start)/60, 2), 'in hours:', round((end - start)/3600, 1))

            
var = dict()
var['n_e_all'] = n_e_all
var['number_simulations'] = number_simulations
var['means'] = means_all
var['trials_per_game'] = trials_per_game
var['horizon'] = free_choices
var['observed'] = observed
var['obs_information'] = obs_information
var['theta'] = theta
var['n_indiv'] = n_indiv
var['models'] = models



with open('../data/data_within_group.pckl', 'wb') as f:
    pickle.dump([estimations_all, means_all, var], f)
print('data saved')
