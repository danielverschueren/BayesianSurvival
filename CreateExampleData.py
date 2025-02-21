import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BayesSurvPlots import plotKaplanMeier

SEED = 400
lam_surv = 0.05
lam_start = 0.05
beta = 0.5
lam_cens = 0.1

with pm.Model() as model:
    tref = pm.Exponential("t_ref", lam=lam_surv)
    tstart = pm.Exponential("t_start", lam=lam_start)
    tend = pm.Exponential("t_end", lam=lam_surv*np.exp(beta))
    tcenstest = pm.Exponential("t_cens_test", lam=lam_cens)

with model:
    x = pm.sample_prior_predictive(draws=100, random_seed=SEED)

t_ref = x.prior.t_ref.values.flatten()
t_start = x.prior.t_start.values.flatten()
t_end = x.prior.t_end.values.flatten()
t_cens_test = x.prior.t_cens_test.values.flatten()

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
data_test['EndTime'] = np.min([t_cens_test,t_end], axis=0)
data_test['Event'] = np.argmin([t_cens_test,t_end], axis=0)
data_test['StartTime'] = t_start
data_test['Reference'] = 0
data_test['Test'] = 1

# join
data = pd.concat([data_ref, data_test], axis=0)

# save
data.to_csv(f"synthData_beta{beta:.1f}_lamsurv{lam_surv:.2f}.csv" )

print(f"cens prob ref: {sum(data_ref['Event'] == 0)/len(data_ref):.3f} ")
print(f"cens prob test: {sum(data_test['Event'] == 0)/len(data_test):.3f} ")

# verify
_, ax = plt.subplots(1,1)
plotKaplanMeier(
    data, 
    colors=['b', 'r'], 
    ax=ax, 
    cols=['Test','Reference'],
    Ts=[-1.,-1.]
)
plt.show()



