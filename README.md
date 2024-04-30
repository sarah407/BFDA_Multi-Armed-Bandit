# Enhancing Experimental Design through Bayes Factor Design Analysis: Insights from an Example Estimating Random Exploration within a Multi-Armed Bandit Task

## Introduction
This is the code used for the simulations and analysis in the paper 'Enhancing Experimental Design through Bayes Factor Design Analysis: Insights from an Example Estimating Random Exploration within a Multi-Armed Bandit Task' (submitted, currently in review). 
The pipeline simulates behavioural data from a two-armed bandit task, after which estimates of the exploration parameter and differences in exploration between groups are analysed.
A design analysis using Bayes Factor was conducted to examine the correlation between sample size and the strength of evidence for a difference in exploration between two groups. This analysis considers the likelihood of substantial evidence for an incorrect hypothesis and the likelihood of insufficient evidence for the correct hypothesis.

## Analysis
The data resulting form the simulations and analysis is stored in the 'data' folder and includes the files:
- data_within_group.pckl
    - data from estimating the exploration parameter within one group
    - includes the variables estimations, means, var 
        - estimations: array
            - estimation of population means
        - means: array
            - tested population mean
        - var: dict
            - summary of used variables
- data_ng-ss.pckl
    - data from estimating the difference in exploration parameter between two groups, as well as calculating the Bayes factor in favour of a difference in means.
        - as a function of number of games per participant and sample size
    - includes the variables BF, estimations, means_all, var
    - true difference in exploration is 0.3

        - BF: array
            - Bayes factor testing tau =\= 0 against tau = 0
        - estimations: array
            - estimation of difference in population means
        - means_all: array
            - tested population means for both groups
        - var: dict
            - summary of used variables
- data_ng-ss_0.pckl
    - data from estimating the difference in exploration parameter between two groups, as well as calculating the Bayes factor in favour of a difference in means.
        - as a function of number of games per participant and sample size
        - true difference in exploration is 0
    - includes the variables BF, estimations, means_all, var
        - BF: array
            - Bayes factor testing tau =\= 0 against tau = 0
        - estimations: array
            - estimation of difference in population means
        - means_all: array
            - tested population means for both groups
        - var: dict
            - summary of used variables
- data_mu-ss.pckl
    - data from estimating the difference in exploration parameter between two groups, as well as calculating the Bayes factor in favour of a difference in means.
        - as a function of the effect size and sample size
    - includes the variables BF, estimations, means_all, var
        - BF: array
            - Bayes factor testing tau =\= 0 against tau = 0
        - estimations: array
            - estimation of difference in population means
        - means_all: array
            - tested population means for both groups
        - var: dict
            - summary of used variables
- data_ALC.pckl
    - data from estimating the average length of the Posterior's 95% HPDI 
    - includes the variables: estimations, means_all, pred_posteriors, lengths, factors, var
        - estimations: array
            - stimation of difference in population means
        - means_all: array
            - tested population means for both groups
        - pred_posteriors: array
            - predictive posteriors for each simulation (multiplied by 10^factors)
        - lengths: array
            - length of HPDI
        - factors: array
            - predictive posteriors were multiplied by  10^factors
        - var: dict
            - summary of used variables

## License
Enhancing Experimental Design through Bayes Factor Design Analysis: Insights from Multi-Armed Bandit Tasks © 2024 is licensed under CC BY 4.0 

This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes.

More information can be found [here](LICENSE.txt)
