import numpy as np
import pymc as pm
import pandas as pd

def resetDataToT(dataX, T):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # get data
    data = dataX[dataX['StartTime'] < T]
    t = data['EndTime'].astype(float).values
    t_start = data['StartTime'].astype(float).values
    t_abs = t_start + t
    real = data['Test'].astype(int).values

    # update t for active uncensored subjects
    censoredsAtT = data['Censored'].astype(int).values
    censoredsAtT[(t_abs > T) & (real == 1)] = 0
    t[(t_abs > T) & (censoredsAtT == 0) & (real == 1)] = \
                    T - t_start[(t_abs > T) & (censoredsAtT == 0) & (real == 1)]

    # check if all t's are valid (>= 0)
    assert (t > 0).all(), "Some lifetimes t <= 0."

    # print info
    print("+=============================================================+")
    print('T = {}'.format(T))
    print("+=============================================================+")
    print('Current subjects: \t\t{}'.format(len(t[real == 1])))
    print('Censored RW at T: \t\t{}'.format(np.sum(censoredsAtT[real == 1])))

    return data, t, censoredsAtT
    
def loglp(t, 
          beta, 
          nu, 
          test_id,
          reference_id,
          censoreds):
    """
    +==========================================================================+

    +==========================================================================+
    """
    eps = 1e-10
    pll = 0

    # baseline
    expB = pm.math.exp(beta)

    # baseline Hazard exp
    # trial
    pll = pm.math.sum(reference_id*(censoreds*(pm.math.log(nu + eps) - nu*t) # true deads               
                      - (1-censoreds)*nu*t))  # alive

    # real
    pll += pm.math.sum(test_id*(censoreds*(pm.math.log(nu + eps) + beta - (nu*t)*expB) # true deads                 
                      - (1-censoreds)*nu*t*expB)) # alive 
    
    return pll

def LifeTimesFull_Exp(data, 
                      t, 
                      censoreds,
                      beta_params=(0,0.5), 
                      nu_params=(-2,0.5)):
    """
    +==========================================================================+
    
    +==========================================================================+
    """
    with pm.Model() as model:

        # get IDs, order matters!
        test_id = pm.Data("test_id", data['Test'].to_numpy())
        reference_id = pm.Data("reference_id", data['Reference'].to_numpy())
        censoreds = pm.Data("censoreds_id", censoreds)

        # set up priors
        beta = pm.Normal('beta', mu=beta_params[0], sigma=beta_params[1]) #23
        nu = pm.LogNormal('nu', mu=nu_params[0], sigma=nu_params[1])

        # parse data
        t = pm.Data("t", t, dims="obs_id")

        # set up custom pll
        pm.CustomDist("likeli", beta, nu,
                      test_id, reference_id, censoreds, logp=loglp, 
                      observed=t, dims='obs_id')

    return model

def loglwb(t, 
           beta, 
           b, 
           k, 
           test_id,
           reference_id,
           censoreds):
    """
    +==========================================================================+

    +==========================================================================+
    """
    eps = 1e-10
    pll = 0

    # baseline
    expB = pm.math.exp(beta)

    # baseline Hazard Weibull
    # trial
    pll = pm.math.sum(reference_id*(censoreds*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) - b*t**k) # true deads                 
                      - (1-censoreds)*b*t**k))  # censoreds

    # real
    pll += pm.math.sum(test_id*(censoreds*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) + beta - expB*b*t**k) # true deads                 
                      - (1-censoreds)*expB*b*t**k))  # censoreds
    
    return pll

def LifeTimesFull_WB(data, 
                     t, 
                     censoreds, 
                     beta_params=(0,2), 
                     b_params=(-2,3), 
                     k_params=(0.9,1.1)):
    """
    +==========================================================================+
    
    +==========================================================================+
    """
    with pm.Model() as model:

        # get IDs, order matters!
        test_id = pm.Data("test_id", data['Test'].to_numpy())
        reference_id = pm.Data("reference_id", data['Reference'].to_numpy())
        censoreds = pm.Data("censoreds_id", censoreds)

        # set up priors
        beta = pm.Normal('beta', mu=beta_params[0], sigma=beta_params[1]) #23
        b = pm.LogNormal('b', mu=b_params[0], sigma=b_params[1])
        k = pm.Uniform('k', lower=k_params[0], upper=k_params[1])

        # parse data
        t = pm.Data("t", t, dims="obs_id")

        # set up custom pll
        pm.DensityDist("likeli", beta, b, k, 
                       test_id, reference_id, censoreds, 
                       logp=loglwb, observed=t, dims='obs_id')

    return model

def survExp(baseline, x, beta):
    """
    +==========================================================================+

    +==========================================================================+
    """
    if 'nu' in baseline.keys():
        nu = baseline['nu']
    else:
        raise Exception('Exponential: param "nu" not in parameter keys...')

    return np.exp(-nu*x*np.exp(beta))

def survWB(baseline, x, beta):
    """
    +==========================================================================+

    +==========================================================================+
    """
    if 'b' in baseline.keys() and 'k' in baseline.keys():
        b = baseline['b']
        k = baseline['k']
    else:
        raise Exception('Weibull: param "b" and "k" not in parameter keys...')
    
    return np.exp(-np.exp(beta)*b*x**k)

def BayesFactorLowerThanHR(posterior_samples, prior_prob, HR):
    """
    +==========================================================================+

    +==========================================================================+
    """
    numSamples = posterior_samples.size
    post_prob = (posterior_samples < np.log(HR)).sum()/numSamples + 0.5/numSamples
    # add numerical error 
    return post_prob/prior_prob

if __name__ == "__main__":

    pass
