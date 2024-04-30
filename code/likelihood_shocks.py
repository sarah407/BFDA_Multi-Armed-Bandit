
import numpy as np
import itertools as it
import pandas as pd

#%%


def P_matr_soft(Q, tau):
    
    '''

    Parameters
    ----------
    Q : array
        Q-values for each arm
    tau: 1-dim array 
        values of exploration parameter tau 
            

    Returns
    -------
    p: array
         probability of choosing the left arm
        
    '''
    
    p = np.exp(Q/tau)/np.expand_dims(np.sum(np.exp(Q/tau), axis = 2), axis = 2)  
    
    return p[:,:,0]


def likelihood_shocks(model, I_0, prob_all, learning_rate, free_choices, tau_all, Q_1 = np.array((0,0))):
    
    '''

    Parameters
    ----------
    model : array
        contains the model used for behavioural data
        'softmax' for softmax model
    I_0 : array of shape (2,)
        distribution of observed trials, e.g.[5, 1] -> one side 5 observed the other 1
    prob_all: array of shape (x, 2)  
        combinations of outcome probabilities for each arm
    learning_rate: float
        learning rate of behavioural model
    free_choices: int
        number of free choices the agent makes
    tau_all: 1-dim array 
        values of tau for which the likelihood of receiving a certain number of shocks is calculated
    Q_1: 1-dim array
        initital Q values
            

    Returns
    -------
    likelihood: array
        likelihoods calculated as a function of tau_all
        
    '''
    
    if model == 'softmax':
        print('softmax')
        def P_matr(Q, tau, tot_trials, current_trial, N):
            return(P_matr_soft(Q, tau))

    # Initialize Likelihood
    likelihood = np.zeros((np.shape(prob_all)[0], free_choices + 1, np.shape(tau_all)[0]))
    
    
    # Calculating possible Q-values at the start of free choices (after observing 4 outcomes)
    poss_observed_a = np.array(list(it.product([0, -1], repeat = I_0[0])))
    poss_observed_b = np.array(list(it.product([0, -1], repeat = I_0[1])))
    Q_a = Q_1[0] * (1-learning_rate)**I_0[0] + np.sum(learning_rate * (1-learning_rate)**(I_0[0]-np.array(range(1, I_0[0]+ 1))) * poss_observed_a[:,:], axis = 1)
    Q_b = Q_1[1] * (1-learning_rate)**I_0[1] + np.sum(learning_rate * (1-learning_rate)**(I_0[1]-np.array(range(1, I_0[1]+ 1))) * poss_observed_b[:,:], axis = 1)
    Q_end_observation = np.array(list(it.product(Q_a, Q_b)))
    
    

    for i_p, p in enumerate(prob_all):
        print(p)
        
        # Calculating the probabilities of the possible Q values after observation
        probability_a  = (1 - p[0])**(np.sum(poss_observed_a + 1, axis = 1)) * p[0]**(np.sum(poss_observed_a, axis = 1) * -1)
        probability_b  = (1 - p[1])**(np.sum(poss_observed_b + 1, axis = 1)) * p[1]**(np.sum(poss_observed_b, axis = 1) * -1)
        probability_end_observation = np.prod(np.array(list(it.product(probability_a, probability_b))), axis = 1)
        
        
        number_possible_paths = 4**free_choices
        
        
        # Creating dataframe with all possible combinations of choice and outcomes for free choice trials
        outcomes = np.array(list(it.product([0, -1], repeat = free_choices)))   # all possible outcomes for the free choices
        choices = np.array(list(it.product([0, 1], repeat = free_choices)))     # all possible free choices
        columns1 = ['choices'] * free_choices + ['outcomes'] * free_choices 
        columns2 = np.tile(np.array(range(1, free_choices + 1)), 2).astype('str')
        data = np.array(list(it.product(choices, outcomes))).reshape(-1, 2 * free_choices)
        df = pd.DataFrame(data, columns  = [columns1, columns2]) 
        
        
        for i_tau, tau in enumerate(tau_all):
            
            # Calculating the probability of each path (combination of choices & outcomes) for each tau
            
            Q = np.tile(Q_end_observation,(number_possible_paths,1,1))
            P_path = np.ones((1,1))
            P_path = P_path * probability_end_observation   # probability of path for all possible Q values after observation
            N = np.tile(I_0, (number_possible_paths,1))     # number of observations

            for step in range(1, free_choices + 1):
                
                P_choice_left = P_matr(Q, tau, free_choices + np.sum(I_0), step - 1, N)
                
                choice_this_trial = df['choices'][str(step)]
                P_choice_this_trial = np.array(choice_this_trial).reshape(-1,1) * (1-P_choice_left) +  np.array(1 - choice_this_trial).reshape(-1,1) * (P_choice_left)
                
                outcome_this_trial = df['outcomes'][str(step)]
                P_outcome_this_trial = (1 + np.array(outcome_this_trial).astype('int')) * (1-p[np.array(choice_this_trial).astype('int')]) - np.array(outcome_this_trial).astype('int') * p[np.array(choice_this_trial).astype('int')]
                
                P_path = P_path * P_choice_this_trial * np.tile(P_outcome_this_trial, (np.shape(Q)[1],1)).T
                
                
                # Updating Q & N
                R = np.tile(np.array(outcome_this_trial), (np.shape(Q)[1],1)) # Reward
                i_choice = np.array(choice_this_trial).astype('int')
                Q[np.array(range(0, np.shape(Q)[0])), :,i_choice] = Q[np.array(range(0, np.shape(Q)[0])), :,i_choice] + learning_rate * (R.T - Q[np.array(range(0, np.shape(Q)[0])), :,i_choice]) 
                N[np.arange(number_possible_paths), choice_this_trial] = N[np.arange(number_possible_paths), choice_this_trial] + 1
            
            # Add path porobability & outcomes to data frame
            df['probability path'] = np.sum(P_path, axis = 1)
            df['total shocks'] = np.sum(df['outcomes'], axis = 1) * (-1)
            
            
            # Calculating the likelihood of receiving a number of shocks as a function of tau and the outcome probabilities of both arms
            for number_shocks in range(free_choices + 1):
                likelihood[i_p, number_shocks, i_tau] = np.sum(df['probability path'][df['total shocks'] == number_shocks])
                
    return likelihood
