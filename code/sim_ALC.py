import numpy as np
import pickle
import itertools as it
from time import time

from likelihood_shocks import *
from data_sample import *
from data_simulate import *
from data_sort import *

start = time()

#%%
def main(number_simulations, group_means, n_e_all, like, model, trials_per_game, n_indiv, prob_all, observed, obs_information, theta, prior_diff):
    '''
    Parameters
    ----------
    number_simulations: int
        number of simulations for this set of means
    group_means: array
        group means for the two groups
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
    prior_diff: array
        prior of difference in means

    Returns
    -------
    BF: array
        Bayes factor testing tau =\= 0 against tau = 0
    estimations: array
        estimation of difference in mean between the two groups
    pred_posteriors: array
        predictive posteriors divided by 10**factors
    factors:array
        predictive posteriors were divided by  10 ** factors
    lengths: array
        length of HPDI
    '''
    BF = np.zeros((len(n_e_all), number_simulations))
    estimations = np.zeros((len(n_e_all), number_simulations))
    lengths = np.zeros((len(n_e_all), number_simulations))
    pred_posteriors = np.zeros((len(n_e_all), number_simulations))
    factors = np.zeros((len(n_e_all), number_simulations))
    
    for index, n_e in enumerate(n_e_all):       
        for i_sim in range(number_simulations):
            
            both_Likelihoods = np.zeros((2, np.shape(theta)[0]))
            both_predictive = np.zeros(2)
            both_factors = np.zeros(2)

            for i_mean, mean in enumerate(group_means):
                horizon= np.asarray(trials_per_game - observed * np.ones(np.shape(trials_per_game)), dtype = 'int')
                
                # simulate data
                data_shocks, list_combinations, real_mean = sample(n_e, int(n_indiv/len(prob_all)), prob_all, observed, trials_per_game, obs_information, model, horizon, mean)

                #________________________Calculating the likelihood for simulated data__________________________________________
                
                tot_likelihood = np.zeros((len(data_shocks[model[0]].keys()) * n_indiv, np.shape(theta)[0]))
                counter = 0
                
                for i_comb, comb in enumerate(list_combinations[1::]):
                    horizon = int(comb[3])

                    for i, epsilon in enumerate(data_shocks[model[0]].keys()):
                        number_shocks = data_shocks[model[0]][epsilon][comb[4]]['shocks'].astype(int)
                        for i_indiv, shock in enumerate(number_shocks):
                            tot_likelihood[counter, :] = like[i_comb, shock,:]
                            counter = counter + 1

                tot_likelihood = np.sum(np.log10(tot_likelihood), axis = 0)

                
                # Calculating the individual predictive posteriors
                pred_posterior = tot_likelihood + np.log10(prior)
                factor = 290 - np.max(pred_posterior)
                pred_posterior = pred_posterior  + factor
                pred_posterior = 10 ** pred_posterior
                pred_posterior = np.nansum(pred_posterior)                
                
                # Calculating the individual likelihoods
                tot_likelihood = tot_likelihood + 290 - np.max(tot_likelihood)
                tot_likelihood = 10 ** tot_likelihood            
                tot_likelihood = tot_likelihood/np.nansum(tot_likelihood)

                # Saving
                both_predictive[i_mean] = pred_posterior
                both_factors[i_mean] = factor
                both_Likelihoods[i_mean] = tot_likelihood
            
            # Convolution of Likelihoods
            Z = np.convolve(both_Likelihoods[0,:][::-1], both_Likelihoods[1,:])
            Z = Z/np.sum(Z)
            new_theta = np.unique(np.round(np.sum(np.meshgrid(theta, -theta), axis = 0),4)) # adjust rounding depending on theta intervals
            
            # Calculate Bayes Factor
            m1 = np.sum(Z * prior_diff)
            m0 = Z[np.where(new_theta == 0)]
            BF[index, i_sim] =  m1/m0

            # Calculate predictive posteriors
            f0 = np.floor(np.log10(both_predictive[0]))
            f1 = np.floor(np.log10(both_predictive[1]))
            pred_posteriors[index, i_sim] = both_predictive[0]/(10**f0) * both_predictive[1]/(10**f1)
            factors[index, i_sim] = both_factors[0] + both_factors[1] - f0 - f1
            
            # Calculate the estimate
            Posterior = Z * prior_diff
            Posterior = Posterior/np.sum(Posterior)
            estimate = np.sum(new_theta * Posterior)  
            estimations[index, i_sim] = estimate
            
            #_____________________________HPDI_____________________________________________
            # Calculating the highest probability density interval of convoluted Posteriors
            
            perc = 0.95
            
            def f1(x, perc):                
                return(abs(perc - np.sum(Z[Z>x])))
            
            f2 = np.vectorize(f1)
            cut_off = Z[np.argmin(f2(Z, perc))]
            HPDI = np.array((new_theta[np.where(Z>cut_off)[0][0] -1], new_theta[np.where(Z>cut_off)[0][-1] + 1]))
            length = HPDI[1] - HPDI[0]
            lengths[index, i_sim] = length
            #plt.plot(new_theta, Z)
            #plt.fill_between(new_theta[np.where(Z>cut_off)], Z[np.where(Z>cut_off)])
        
            
    return(estimations, pred_posteriors, lengths, factors)

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

n_e_all =  np.array(range(10, 60, 2))

trials_per_game = all_trials_per_game[0]
free_choices = trials_per_game[0] - observed

theta = np.linspace(0.003,3, 1000)
prior =  np.ones(np.shape(theta))
prior_diff = np.ones((np.shape(theta)[0] * 2 - 1))/np.ones((np.shape(theta)[0] * 2 - 1))

Q_1 = np.array((0,0))

likelihood = likelihood_shocks(models, I, prob_all, learning_rate, free_choices, theta, Q_1)

print('finished likelihood shocks')

#%%
n_indiv = np.array((150, 210, 270))
number_simulations = 500

# Generating pseudo-random means for a set of differences in means
diff_mean =  np.array((0.3))
means_all = np.array(
    np.meshgrid(
        diff_mean, np.random.uniform(0, 1, number_simulations)
        )).T.reshape(-1,2)

while any(np.logical_or(np.sum(means_all, axis = 1) < 0, np.sum(means_all, axis = 1) > 1)):
    i = np.where(np.logical_or(np.sum(means_all, axis = 1) < 0, np.sum(means_all, axis = 1) > 1))
    means_all[i,1] = np.random.uniform(0, 1, np.shape(i)[1])

means_all = [means_all[:,1] , np.sum(means_all, axis = 1)]
means_all = np.array(means_all).T

# Simulations of Bayes Factors
BF_all          = np.zeros((len(n_indiv), len(n_e_all), number_simulations))
estimations_all = np.zeros((len(n_indiv), len(n_e_all), number_simulations))
lengths_all     = np.zeros((len(n_indiv), len(n_e_all), number_simulations))    
factors_all     = np.zeros((len(n_indiv), len(n_e_all), number_simulations))    
pred_posteriors_all = np.zeros((len(n_indiv), len(n_e_all), number_simulations))

start = time()
for i_indiv, n in enumerate(n_indiv):
    print('-----n_indiv: ', n)
    start_indiv = time()
    for i_means, means in enumerate(means_all):
        BF, estimations, pred_posteriors, lengths, factors = main(1, means, n_e_all, likelihood, models,trials_per_game, n, prob_all, observed, obs_information, theta,prior_diff)
        BF_all[i_indiv, :, i_means] = BF[:,0]
        
        estimations_all[i_indiv, :, i_means] = estimations[:,0]
        pred_posteriors_all[i_indiv, :, i_means] = pred_posteriors[:,0]
        lengths_all[i_indiv, :, i_means] = lengths[:,0]
        factors_all[i_indiv, :, i_means] = factors[:,0]
    end_indiv = time()
    print('time for {} individuals: {} minutes bzw. {} hours'.format(n, round((end_indiv - start_indiv)/60, 2), round((end_indiv - start_indiv)/3600, 2)))
end = time()

#%% Save Data
print('=========================\ntotal time in minutes: ', round(( end - start)/60, 2), 'in hours:', round((end - start)/3600, 1))
 
var = dict()
var['n_e_all'] = n_e_all
var['number_simulations'] = number_simulations
var['diff_mean'] = diff_mean
var['trials_per_game'] = trials_per_game
var['horizon'] = free_choices
var['observed'] = observed
var['obs_information'] = obs_information
var['theta'] = theta
var['n_indiv'] = n_indiv
var['models'] = models



with open('../data/data_ALC.pckl', 'wb') as f:
    pickle.dump([estimations_all, means_all, pred_posteriors_all, lengths_all, factors_all, var], f)

print('data saved')