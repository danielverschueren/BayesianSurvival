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
    date = "20241024"

    #### load data ####
    save_dir = f"results/plots_chemo_{date}_splitT/"
    file = "data/PBChemoOS_TRRW_start.csv"
    dataDF = pd.read_csv(file)
    dataDF.rename(columns={'RW' : 'Test', 'TR' : 'Reference'}, inplace=True)
    #dataDF['StartTime'][dataDF['Test'] == 1] 

    #### model fitting parameters ####
    surv = survWB_splitT
    ndraw = 5000
    num_steps = 20
    PosteriorsT = []
    ts = np.linspace(1,50,num_steps)
    #ts = [50]
    num_steps = len(ts)
    beta1_prior = (0, 0.5)
    beta2_prior = (0, 0.5)
    b_prior = (-3, 1)
    k_prior = (0, 3)
    #T_cuts = np.arange(6,30)
    T_cuts = [4]

    params = dict()
    params["beta1"] = 1.7
    params["beta2"] = 0.15
    params["k"] = 1
    params["b"] = 0.03
    params["n1"] = 100
    params["n2"] = 300
    x = np.linspace(0,50,100)
    y = survWB_splitT(params, x)
    _, ax = plt.subplots(1,1)
    plotKaplanMeier(dataDF, ['Test', 'Reference'], ['r', 'b'], ax)
    ax.plot(x,y)
    plt.savefig("best_double_exp_fit.png")
    plt.close()
    #plt.show()

    #### run model ####
    for T in ts:
        for T_cut in T_cuts:
            data = resetDataToT(dataDF, T)
            model = LifeTimesSplitT_WB(
                data,
                beta1_params=beta1_prior,
                beta2_params=beta2_prior,
                T_cut_param=T_cut,
                b_params=b_prior,
                k_params=k_prior
            )
            with model:
                # prior
                prior_pred = pm.sample_prior_predictive(
                    draws=ndraw*10,
                    var_names = ["b", "k", "beta1", "beta2"],
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
    with open(Path(save_dir, f"PosteriorTcut{T_cut}mchemo_{date}.pk"), 'wb') as handle:
        pickle.dump(PosteriorsT, handle)


    #### PLOTTING ####

    # mcmc check
    for i in range(len(ts)):
        az.plot_trace(PosteriorsT[i][1])
        plt.savefig(Path(save_dir, f"azdiagTcut{T_cut}_T{ts[i]}.png"))
        plt.close()

        # plot KMs and median fits
        _, ax2 = plt.subplots(1, 1, figsize=(11,3))
        ax2 = plotKMPosteriorFits(
            50, 
            50, 
            PosteriorsT[i][1], 
            dataDF, 
            surv, 
            ax2,
            plot_ref_intervals=False,
        )
        plt.savefig(Path(save_dir, f"KMposteriorsTcut{T_cut}_T{ts[i]}_t.png"))
        plt.close()

    # plot posterior
    _, ax1 = plt.subplots(1, 2, figsize=(12,5))
    ax1 = plotPosteriors(PosteriorsT, ax1, [i for i in range(len(ts))], ['beta2','beta1'], xlims_baseline=[[-2,2], [-2,2]])
    plt.savefig(Path(save_dir, f"PosteriorsTcut{T_cut}_T.png"))
    plt.close()

    # plot CIs
    HRs = [0.5, 1.0, 2]
    _, ax3 = plt.subplots(1,1, figsize=(10,4))
    ax3 = plotPosteriorBetaCI(PosteriorsT, HRs, ax3, param_name='beta1')
    ax3.set_ylim([0, 3])
    plt.savefig(Path(save_dir, f"CI_beta1Tcut{T_cut}_T.png"))
    plt.close()

    HRs = [0.5, 1.0, 2]
    _, ax3 = plt.subplots(1,1, figsize=(10,4))
    ax3 = plotPosteriorBetaCI(PosteriorsT, HRs, ax3, param_name='beta2')
    ax3.set_ylim([0, 3])
    plt.savefig(Path(save_dir, f"CI_beta2Tcut{T_cut}_T.png"))
    plt.close()

