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
    num_steps = 3
    PosteriorsT = []
    ts = np.linspace(1,50,num_steps)
    num_steps = len(ts)
    beta1_prior = (0, 0.5)
    beta2_prior = (0, 0.5)
    b_prior = (-3, 1)
    k_prior = (0.5, 1.5)
    T_cut_prior = (5.)

    with open(Path(save_dir, f"PosteriorWeibullchemo_{date}.pk"), 'rb') as f:
        PosteriorsT = pickle.load(f)

    # plot KMs and median fits
    _, ax2 = plt.subplots(1, 1, figsize=(11,3))
    ax2 = plotKMPosteriorFits(
        PosteriorsT[2][0], 
        PosteriorsT[-1][0], 
        PosteriorsT[2][1], 
        dataDF, 
        surv, 
        ax2,
        plot_ref_intervals=False,
    )
    plt.show()

    # plot posterior
    _, ax1 = plt.subplots(1, 3, figsize=(17,5))
    ax1 = plotPosteriors(PosteriorsT, ax1, [0,1,2], ['b','k'], xlims_baseline=[[0, 0.1], [0.5,1.5]])
    plt.show()
