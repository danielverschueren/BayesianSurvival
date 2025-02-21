import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

eps = 1e-10
SEED = 42

def loglp(t, lam, delta):
    pll = pm.math.sum(
            (delta*(pm.math.log(lam + eps) - (lam*t)) # event
            - (1-delta)*lam*t)  # survived
    ) 
    return pll

def survs_model(t, deltas, nu_params=(-2.,0.5)):

    with pm.Model() as model:

        # parse data
        delta = pm.Data("events_id", deltas)
        t = pm.Data("t", t, dims="obs_id")

        # prior
        lam = pm.LogNormal('lam', mu=nu_params[0], sigma=nu_params[1])

        # ll
        pm.CustomDist(
             "likeli", lam, delta, logp=loglp, observed=t, dims='obs_id'
        )

    return model

if __name__ == "__main__":

    # set up generative model
    with pm.Model() as model:
        T = pm.Exponential("T", lam=0.1) # mean lifetime 10
        C = pm.Uniform("C", lower=0, upper=30)
    
    # sample
    with model:
        x = pm.sample_prior_predictive(draws=100, random_seed=SEED)
    T = x.prior.T.values.flatten()
    C = x.prior.C.values.flatten()

    t = np.min([C,T], axis=0)
    deltas = np.argmin([C,T], axis=0)
    
    # infer
    ndraw = 5000
    model = survs_model(t, deltas)

    with model:
            # prior
            prior_pred = pm.sample_prior_predictive(
                draws=ndraw*10,
                var_names = ["lam"],
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

    # az.plot_trace(prior_pred.prior)
    az.plot_trace(result.posterior)
    plt.show()


