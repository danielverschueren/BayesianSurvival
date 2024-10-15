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
    #date = "20241013"

    #### load data ####
    save_dir = f"results/plots_chemo_{date}/"
    file = "data/PBChemoOS_TRRW_start.csv"
    dataDF = pd.read_csv(file)
    dataDF.rename(columns={'RW' : 'Test', 'TR' : 'Reference'}, inplace=True)

    #### model fitting parameters ####
    surv = survWB
    ndraw = 5000
    num_steps = 99
    PosteriorsT = []
    ts = np.linspace(1,50,num_steps)
    num_steps = len(ts)
    beta_prior = (0, 0.5)
    b_prior = (-3, 1)
    k_prior = (0.5, 1.5)
    load = True

    if load:
        with open(Path(save_dir, f"PosteriorWeibullchemo_{date}.pk"), 'rb') as f:
            PosteriorsT = pickle.load(f)
    else:
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
        with open(Path(save_dir, f"PosteriorWeibullchemo_{date}.pk"), 'wb') as handle:
            pickle.dump(PosteriorsT, handle)


    #### PLOTTING ####
    # settings
    show_js = [8, 18, 28, 38, 48, 58, 68, 78, 88, 98]

    # simple stats
    dataRW = dataDF[dataDF['Test'] == 1]
    outfileStats = pd.DataFrame(index=ts, columns=['subjects', 'censored', 'event', 'person_months'])
    for ti in ts:
        dataT = dataRW[ti > dataRW['StartTime'] ]
        t = (dataT['EndTime'] + dataT['StartTime']).astype(float).to_numpy()
        t_start = dataT['StartTime'].astype(float).to_numpy()
        x = pd.Series(index=['subjects', 'censored', 'event', 'person_months'])
        x['subjects'] = len(dataT)
        x['event'] = sum([(dataT['Event'] == 1) & (ti > t)]).sum()
        x['censored'] = sum([(dataT['Event'] == 0) & (ti > t)]).sum()
        person_months = sum(t[ti > t] - t_start[ti > t]) + \
            sum((ti*(ti <= t).astype(int) - t_start*(ti <= t).astype(int)))
        x['person_months'] = person_months
        outfileStats.loc[ti] = x
    outfileStats.to_csv(Path(save_dir, f"chemo_stats_{date}.csv"))

    # mcmc check
    az.plot_trace(PosteriorsT[50][1])
    plt.savefig(Path(save_dir, f"chemo_traceplot_t{PosteriorsT[50][0]}_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_traceplot_t{PosteriorsT[50][0]}_{date}.svg"), bbox_inches="tight")
    plt.close()

    # plot priors
    data = resetDataToT(dataDF, ts[-1])
    model = LifeTimesFull_WB(
        data,
        beta_params=beta_prior,
        b_params=b_prior,
        k_params=k_prior
    )
    with model:
        _, axa = plt.subplots(1, 3, figsize=(17,5))
        plot_pymc(pm.Normal('beta_', mu=beta_prior[0], sigma=beta_prior[1]), ax=axa[0], xlims=[-1.5, 1.5])
        plot_pymc(pm.LogNormal('b_', mu=b_prior[0], sigma=b_prior[1]), ax=axa[1], xlims=[0,0.3])
        plot_pymc(pm.Uniform('k_', lower=k_prior[0], upper=k_prior[1]), ax=axa[2], xlims=[0.5, 1.5])
        plt.savefig(Path(save_dir, f"chemo_priors_{date}.png"), bbox_inches="tight")
        plt.savefig(Path(save_dir, f"chemo_priors_{date}.svg"), bbox_inches="tight")
        plt.close()

    # plot prior check
    _, ax0 = plt.subplots(1, 1, figsize=(11,3))
    ax0 = plotKMPosteriorFits(
        PosteriorsT[-1][0], 
        PosteriorsT[-1][0], 
        PosteriorsT[-1][2], 
        dataDF, 
        surv, 
        ax0,
        plot_ref_intervals=True,
        prior=True
    )
    plt.savefig(Path(save_dir, f"chemo_kms_priorcheck_t{PosteriorsT[-1][0]}_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_kms_priorcheck_t{PosteriorsT[-1][0]}_{date}.svg"), bbox_inches="tight")
    plt.close()

    # plot Exp Posteriors
    _, ax1 = plt.subplots(1, 3, figsize=(17,5))
    ax1 = plotPosteriors(PosteriorsT, ax1, show_js, ['b','k'], xlims_baseline=[[0, 0.1], [0.5,1.5]])
    plt.savefig(Path(save_dir, f"chemo_posteriors_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_posteriors_{date}.svg"), bbox_inches="tight")
    plt.close()

    # plot KMs and median fits
    for j in show_js:
        _, ax2 = plt.subplots(1, 1, figsize=(11,3))
        ax2 = plotKMPosteriorFits(
            PosteriorsT[j][0], 
            PosteriorsT[-1][0], 
            PosteriorsT[j][1], 
            dataDF, 
            surv, 
            ax2,
            plot_ref_intervals=False,
        )
        plt.savefig(Path(save_dir, f"chemo_kms_posteriorcheck_t{PosteriorsT[j][0]}_{date}.png"), bbox_inches="tight")
        plt.savefig(Path(save_dir, f"chemo_kms_posteriorcheck_t{PosteriorsT[j][0]}_{date}.svg"), bbox_inches="tight")
        plt.close()

    # plot CIs
    HRs = [0.5, 1.0, 2]
    _, ax3 = plt.subplots(1,1, figsize=(10,4))
    ax3 = plotPosteriorBetaCI(PosteriorsT, HRs, ax3)
    ax3.set_ylim([0, 3])
    plt.savefig(Path(save_dir, f"chemo_cis_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_cis_{date}.svg"), bbox_inches="tight")
    plt.close()

    # plot BayesFactor    
    HRLs = [0.33, 0.5, 0.67, 1.00]
    HRHs = [1.00, 1.50, 2.00, 3.00]
    _, ax4 = plt.subplots(1,1, figsize=(10,4))
    ax4 = plotPosteriorBetaBayesFactors(PosteriorsT, beta_prior, HRLs, HRHs, ax4)
    ax4.set_ylim([1/100, 100])
    plt.savefig(Path(save_dir, f"chemo_bfs_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_bfs_{date}.svg"), bbox_inches="tight")
    plt.close()

    # plot BayesFactor inBetween
    HRs = [1.1, 1.2, 1.5]
    _, ax5 = plt.subplots(1,1, figsize=(10,4))
    ax5 = plotPosteriorBetaBayesFactors_inBetween(PosteriorsT, beta_prior, HRs, ax5)
    ax5.set_ylim([1/100, 100])
    plt.savefig(Path(save_dir, f"chemo_bfs_inbetween_{date}.png"), bbox_inches="tight")
    plt.savefig(Path(save_dir, f"chemo_bfs_inbetween_{date}.svg"), bbox_inches="tight")
    plt.close()


    a, b, c = betaBayesFactors(PosteriorsT, beta_prior, HRs,)
    a.to_csv(Path(save_dir, f"chemo_bfs_{date}.csv"))
    b.to_csv(Path(save_dir, f"chemo_cis_{date}.csv"))
    c.to_csv(Path(save_dir, f"chemo_cfs_{date}.csv"))

