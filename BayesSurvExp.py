import numpy as np
import pymc as pm
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pickle

from BayesSurv import *
from BayesSurvPlots import *

if __name__ == "__main__":

    # load data
    file = "PBMonoOS_TRRW_start.csv"
    dataDF = pd.read_csv(file)

    # plot KM
    fig, ax = plt.subplots()
    color = ['b', 'r']
    kind = ['RW', 'TR']
    plotKaplanMeier(dataDF, color, ax, cols=kind, Ts=[30, 30])
    #plt.show()

    # model fitting
    ndraw = 5000
    num_steps = 6
    PosteriorsT = []
    ts = np.linspace(1,30,num_steps)
    beta_prior = (0, 0.5)
    nu_prior = (-2, 0.5)
    load = True

    if load:
        with open('PosteriorExponentialPembro.pk', 'rb') as f:
            PosteriorsT = pickle.load(f)
    else:
        for T in ts:
            data, t, censoredsAtT = resetDataToT(dataDF, T)
            model = LifeTimesFull_Exp(data, 
                                    t,
                                    censoredsAtT, 
                                    beta_params=beta_prior,
                                    nu_params=nu_prior)
            with model:
                result = pm.sample(draws=ndraw, 
                                tune=4000, 
                                target_accept=0.999, 
                                random_seed=42)
            PosteriorsT.append([T, result.posterior])

        # save runs
        with open('PosteriorExponentialPembroX.pk', 'wb') as handle:
            pickle.dump(PosteriorsT, handle)

    # plot Exp Posteriors
    num_shows=6
    _, ax1 = plt.subplots(1, 2, figsize=(10,4))
    ax1 = plotPosteriors(PosteriorsT, 'nu', ax1, num_shows, beta_prior, nu_prior)
    #plt.show()

    # plot KMs and median fits
    _, ax2 = plt.subplots(num_shows, 1, figsize=(5,11))
    surv = survExp
    ax2 = plotKMPosteriorFits(PosteriorsT, dataDF, surv, ax2, num_shows)
    #plt.show()

    # plot CIs
    HRs = [1, 1.1, 1.2, 1.5, 2, 2.5]
    _, ax3 = plt.subplots(1,2, figsize=(10,4))
    ax3 = plotPosteriorBetaCI(PosteriorsT, HRs, ax3)
    #plt.show()

    # plot BayesFactor
    _, ax4 = plt.subplots(1,1, figsize=(10,4))
    ax4 = plotPosteriorBetaBayesFactors(PosteriorsT, beta_prior, HRs, ax4)
    plt.show()
