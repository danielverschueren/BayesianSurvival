import numpy as np
import pymc as pm
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import os
from string import ascii_uppercase as auc
from pathlib import Path
from datetime import datetime

from BayesSurv import *
from BayesSurvPlots import *

SEED = 42

if __name__ == "__main__":

    #### load data ####
    file = "synthData_beta0.5_lamsurv0.05.csv" 
    dataDF = pd.read_csv(file)

    #### model fitting parameters ####
    surv = survExp
    ndraw = 5000
    beta_prior = (0, 0.5)
    nu_prior = (-3, 1)
    
    #### run model ####
    data = resetDataToT(dataDF, 1000)
    model = LifeTimesFull_Exp(
        data,
        beta_params=beta_prior,
        nu_params=nu_prior
    )
    with model:
        # prior
        prior_pred = pm.sample_prior_predictive(
            draws=ndraw*10,
            var_names = ["nu", "beta"],
            random_seed=SEED,
        )
        # posterior
        result = pm.sample(
            draws=ndraw, 
            tune=4000, 
            target_accept=0.99, 
            random_seed=SEED,
            return_inferencedata=True
        )
        # mcmc convergence
        ess = az.ess(result)
        rhat = az.rhat(result)

    # print metrics   
    print(ess)
    print(rhat)

    #### PLOTTING ####
    now = datetime.now() # current date and time
    date = now.strftime("%Y%m%d")
    show_js = [0]
    plotContainer = [[100, result.posterior, prior_pred.prior, (ess, rhat)]]
    save_dir = f"results/plots_synth_{date}/"
    try:
        os.makedirs(save_dir)
    except:
        pass

    # mcmc check
    az.plot_trace(plotContainer[0][1])
    plt.savefig(Path(
        save_dir, f"synth_traceplot_{date}.png"), bbox_inches="tight"
    )
    plt.close()

    # plot posterior
    _, ax1 = plt.subplots(1, 2, figsize=(17,5))
    ax1 = plotPosteriors(
        plotContainer, 
        ax1, 
        [0], 
        ['beta','nu'], 
        xlims_baseline=[[-1, 1], [0.01, 0.1]]
    )
    ax1[0].axvline(0.5, linestyle='-.')
    ax1[1].axvline(0.05, linestyle='-.')
    plt.savefig(
        Path(save_dir, f"synth_posteriors_{date}.png"), bbox_inches="tight"
    )
    plt.close()

