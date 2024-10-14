import numpy as np
import pymc as pm
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime

from BayesSurv import *
from BayesSurvPlots import *

SEED = 42

if __name__ == "__main__":

    now = datetime.now() # current date and time
    date = now.strftime("%Y%m%d") # edit for loading

    #### load data ####
    save_dir = f"plots_pembro_{date}/"
    file = "PBMonoOS_TRRW_start.csv"
    dataDF = pd.read_csv(file)
    dataDF.rename(columns={'RW' : 'Test', 'TR' : 'Reference'}, inplace=True)

    #### model fitting parameters ####
    surv = survWB
    ndraw = 5000
    num_steps = 63
    PosteriorsT = []
    ts = np.linspace(1,32,num_steps)
    num_steps = len(ts)
    beta_prior = (0, 0.5)
    b_prior = (-3, 1)
    k_prior = (0.5, 1.5)

    #### run model ####
    for T in ts:
        data = resetDataToT(dataDF, T)
        model = LifeTimesFull_WB(
            data,
            beta_params=beta_prior,
            b_params=b_prior,
            k_params=k_prior
        )
        with model:
            # prior
            prior_pred = pm.sample_prior_predictive(
                draws=ndraw*10,
                var_names = ["b", "k", "beta"],
                random_seed=SEED,
            )
            # posterior
            result = pm.sample(
                draws=ndraw, 
                tune=4000, 
                target_accept=0.999, 
                random_seed=SEED,
                return_inferencedata=True
            )
            # mcmc convergence
            ess = az.ess(result)
            rhat = az.rhat(result)

        # print and store    
        print(ess)
        print(rhat)
        PosteriorsT.append([T, result.posterior, 
                            prior_pred.prior, 
                            (ess, rhat)])

    # save runs
    with open(Path(save_dir, f"PosteriorWeibullpembro_{date}.pk"), 'wb') as handle:
        pickle.dump(PosteriorsT, handle)


    #### PLOTTING ####

    # mcmc check
    az.plot_trace(PosteriorsT[50][1])
    plt.show()

    # plot KMs and median fits
    _, ax2 = plt.subplots(1, 1, figsize=(11,3))
    ax2 = plotKMPosteriorFits(
        PosteriorsT[30][0], 
        PosteriorsT[-1][0], 
        PosteriorsT[30][1], 
        dataDF, 
        surv, 
        ax2,
        plot_ref_intervals=False,
    )
    plt.show()

    # plot posterior
    _, ax1 = plt.subplots(1, 3, figsize=(17,5))
    ax1 = plotPosteriors(PosteriorsT, ax1, [2,22,42,62], ['b','k'], xlims_baseline=[[0, 0.1], [0.5,1.5]])
    plt.show()

    # plot CIs
    HRs = [0.5, 1.0, 2]
    _, ax3 = plt.subplots(1,1, figsize=(10,4))
    ax3 = plotPosteriorBetaCI(PosteriorsT, HRs, ax3)
    ax3.set_ylim([0, 3])
    plt.show()

    # plot BayesFactor    
    HRLs = [0.50, 0.67, 0.83, 1.00]
    HRHs = [1.00, 1.20, 1.50, 2.00]
    _, ax4 = plt.subplots(1,1, figsize=(10,4))
    ax4 = plotPosteriorBetaBayesFactors(PosteriorsT, beta_prior, HRLs, HRHs, ax4)
    ax4.set_ylim([1/100, 100])
    plt.show()
