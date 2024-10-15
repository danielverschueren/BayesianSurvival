import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BayesSurvPlots import plotKaplanMeier

SEED = 42
lam_surv = 0.1
lam_start = 0.2
lam_cens = 0.1
beta = 0.5

with pm.Model() as model:
    tref = pm.Exponential("t_ref", lam=lam_surv)
    tstart = pm.Exponential("t_start", lam=lam_start)
    tend = pm.Exponential("t_end", lam=lam_surv*np.exp(beta))
    tcens = pm.Exponential("t_cens", lam=lam_cens)

with model:
    x = pm.sample_prior_predictive(draws=100, random_seed=SEED)

t_ref = x.prior.t_ref.values.flatten()
t_start = x.prior.t_start.values.flatten()
t_end = x.prior.t_end.values.flatten()
t_cens = x.prior.t_cens.values.flatten()

# construct data
data_test = pd.DataFrame(
    np.zeros((100,5)), 
    columns=['EndTime', 'StartTime', 'Event', 'Test', 'Reference']
)
data_ref = pd.DataFrame(
    np.zeros((100,5)), 
    columns=['EndTime', 'StartTime', 'Event', 'Test', 'Reference']
)
# ref
data_ref['EndTime'] = t_ref
data_ref['Event'] = 1
data_ref['StartTime'] = 0
data_ref['Reference'] = 1
data_ref['Test'] = 0

# test
data_test['EndTime'] = np.min([t_cens,t_end], axis=0)
data_test['Event'] = np.argmin([t_cens,t_end], axis=0)
data_test['StartTime'] = t_start
data_test['Reference'] = 0
data_test['Test'] = 1

data = pd.concat([data_ref, data_test], axis=0)

data.to_csv("data_exp_example.csv")

_, ax = plt.subplots(1,1)
plotKaplanMeier(
    data, 
    colors=['b', 'r'], 
    ax=ax, 
    cols=['Test','Reference'],
    Ts=[-1.,-1.]
)
plt.show()



