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
from BayesSurvSplitT import *
from BayesSurvPlots import *

SEED = 42

if __name__ == "__main__":

    now = datetime.now() # current date and time
    date = now.strftime("%Y%m%d") # edit for loading

    #### load data ####
    save_dir = f"results/plots_chemo_{date}_splitT/"
    file = "data/PBChemoOS_TRRW_start.csv"
    dataDF = pd.read_csv(file)
    dataDF.rename(columns={'RW' : 'Test', 'TR' : 'Reference'}, inplace=True)

    #### model fitting parameters ####
    surv = survWB_splitT
    ndraw = 5000
    num_steps = 1
    PosteriorsT = []
    ts = np.linspace(1,50,num_steps)
    num_steps = len(ts)
    beta1_prior = (0.5, 0.2)
    beta2_prior = (0, 0.2)
    b_prior = (-3, 1)
    k_prior = (0.5, 1.5)
    T_cut_prior = (.2)

    params = dict()
    params["beta1"] = 2
    params["beta2"] = 0
    params["k"] = 1
    params["b"] = 0.015
    params["n1"] = 100
    params["n2"] = 300
    x = np.linspace(0,50,100)
    y = survWB_splitT(params, x)
    plt.plot(x,y)
    plt.show()

    #### run model ####
    for T in ts:
        data = resetDataToT(dataDF, T)
        model = LifeTimesSplitT_WB(
            data,
            beta1_params=beta1_prior,
            beta2_params=beta2_prior,
            T_cut_param=T_cut_prior,
            b_params=b_prior,
            k_params=k_prior
        )
        with model:
            # prior
            prior_pred = pm.sample_prior_predictive(
                draws=ndraw*10,
                var_names = ["b", "k", "beta1", "beta2", "T_cut"],
                random_seed=SEED,
            )
            # posterior
            result = pm.sample(
                draws=ndraw, 
                tune=4000, 
                target_accept=0.9, 
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
    #with open(Path(save_dir, f"PosteriorWeibullchemo_{date}.pk"), 'wb') as handle:
    #    pickle.dump(PosteriorsT, handle)


    #### PLOTTING ####

    # mcmc check
    az.plot_trace(PosteriorsT[0][1])
    plt.show()

    # plot KMs and median fits
    _, ax2 = plt.subplots(1, 1, figsize=(11,3))
    ax2 = plotKMPosteriorFits(
        PosteriorsT[0][0], 
        PosteriorsT[-1][0], 
        PosteriorsT[0][1], 
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
