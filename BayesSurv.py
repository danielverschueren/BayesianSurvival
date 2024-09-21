import numpy as np
import pymc as pm
import pandas as pd

def resetDataToT(
        dataX: pd.DataFrame, 
        T: float,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    +==========================================================================+
    Reset historic data to time point T, so simulate real time updates.

    The input dataframe needs a specific structure, with subjects on the 
    rows, and columns:
        'StartTime' for the time at which subject had started,
        'EndTime' for the time at which the subject was censored/dropped out, 
        'Test' to indicate if subject belongs to the test or trial data
        'Censored' to indicate if subject was censored or not

    Args:
        dataX (pd.DataFrame) :  data frame with data 
        T (float) : time to reset endtime of subjects to

    Returns:
        pd.DataFrame with subjects that started before T, 
        np.ndarray of updated endtimes, 
        np.ndarray of censoring indiciation
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
    
def loglp(
        t: pm.CallableTensor, 
        beta: pm.CallableTensor, 
        nu: pm.CallableTensor, 
        test_id: pm.CallableTensor,
        reference_id: pm.CallableTensor,
        censoreds: pm.CallableTensor,
    ) -> pm.CallableTensor:
    """
    +==========================================================================+
    Survival Log likelihood for an exponential distribution of lifetimes between
    two populations identified with test_id or reference_id and constant hazard 
    beta difference between the populations

    Args:
        t (pm.CallableTensor) : (N,) vector of drop/censoring times
        beta (pm.CallableTensor) : (1,) hazard rate
        nu (pm.CallableTensor) : (1,) exponential rate mean (distribution param)
        test_id (pm.CallableTensor) : (N,) subject in test set (1) or not (0)
        reference_id (pm.CallableTensor) : (N,) subject in reference set (1)
                                            or not (0)
        censoreds (pm.CallableTensor) : (N,) subject censored (1) or not (0)
    
    Returns:
        pm.CallableTensor Survival Log Likelihood of data given parameters.
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

def LifeTimesFull_Exp(
        data: np.ndarray, 
        t: np.ndarray, 
        censoreds: np.ndarray,
        beta_params: tuple[float, float]=(0.,0.5), 
        nu_params: tuple[float, float]=(-2.,0.5),
    ) -> pm.Model:
    """
    +==========================================================================+
    Function to instantiate a pymc model consturcted using an exponential 
    lifetimes distribution, constant hazard rate, observed lifetimes t, data_id 
    to determine where subject's group.

    Args:
        data (np.ndarray) : (N,2) data array with the following columns
                            col 0: "test_id", in test set (1) or not (0)
                            col 1: "reference_id", in reference set (1) or not 
                            (0)
        t (np.ndarray) : (N,) observed lifetimes
        censoreds (np.ndarray) : (N,) censored (1) or not (0)
        beta_params tuple(float, float): beta prior parameters: Norm(mu, sigma)
        nu_params tuple(float, float): nu prior parameters: LogNorm(mu, sigma)
    Returns:
        pm.Model pymc model to sample from            
    +==========================================================================+
    """
    with pm.Model() as model:

        # get IDs, order matters!
        test_id = pm.Data("test_id", data[:,0])
        reference_id = pm.Data("reference_id", data[:,1])
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

def loglwb(
        t: pm.CallableTensor, 
        beta: pm.CallableTensor, 
        b: pm.CallableTensor, 
        k: pm.CallableTensor,
        test_id: pm.CallableTensor,
        reference_id: pm.CallableTensor,
        censoreds: pm.CallableTensor,
    ) -> pm.CallableTensor:
    """
    +==========================================================================+
    Survival Log likelihood for an Weibull distribution of lifetimes between
    two populations identified with test_id or reference_id and constant hazard 
    beta difference between the populations

    Args:
        t (pm.CallableTensor) : (N,) vector of drop/censoring times
        beta (pm.CallableTensor) : (1,) hazard rate
        b (pm.CallableTensor) : (1,) Weibull b param (distribution param)
        k (pm.CallableTensor) : (1,) Weibull k param (distribution param)
        test_id (pm.CallableTensor) : (N,) subject in test set (1) or not (0)
        reference_id (pm.CallableTensor) : (N,) subject in reference set (1)
                                            or not (0)
        censoreds (pm.CallableTensor) : (N,) subject censored (1) or not (0)
    
    Returns:
        pm.CallableTensor Survival Log Likelihood of data given parameters.
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

def LifeTimesFull_WB(
        data: np.ndarray, 
        t: np.ndarray, 
        censoreds: np.ndarray,
        beta_params: tuple[float, float]=(0.,2.), 
        b_params: tuple[float, float]=(-2.,3.), 
        k_params: tuple[float, float]=(0.9,1.1)
    ) -> pm.Model:
    """
    +==========================================================================+
    Function to instantiate a pymc model consturcted using a Weibull
    lifetimes distribution, constant hazard rate, observed lifetimes t, data_id 
    to determine where subject's group.

    Args:
        data (np.ndarray) : (N,2) data array with the following columns
                            col 0: "test_id", in test set (1) or not (0)
                            col 1: "reference_id", in reference set (1) or not 
                            (0)
        t (np.ndarray) : (N,) observed lifetimes
        censoreds (np.ndarray) : (N,) censored (1) or not (0)
        beta_params tuple(float, float): beta prior parameters: Norm(mu, sigma)
        b_params tuple(float, float): b prior parameters: LogNorm(mu, sigma)
        k_params tuple(float, float): k prior parameters: Unif(low, high)
    Returns:
        pm.Model pymc model to sample from     
    +==========================================================================+
    """
    with pm.Model() as model:

        # get IDs, order matters!
        test_id = pm.Data("test_id", data[:,0])
        reference_id = pm.Data("reference_id", data[:,1])
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

def survExp(
        baseline: dict, 
        x: np.ndarray, 
        beta: float
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate exponential survival probability for parameter 'nu' in 
    baseline and log hazard rate beta.

    p(x) = exp(-nu*x*exp(beta))

    Args:
        baseline (dict) : dictionary with parameter 'nu'
        x (np.ndarray) : (n,) array of survival times to evaluate
        beta (float) : log hazard rate
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    if 'nu' in baseline.keys():
        nu = baseline['nu']
    else:
        raise Exception('Exponential: param "nu" not in parameter keys...')

    return np.exp(-nu*x*np.exp(beta))

def survWB( 
        baseline: dict, 
        x: np.ndarray, 
        beta: float
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate Weibull survival probability for parameter 'k' and 'b' 
    in baseline and log hazard rate beta.

    p(x) = exp(-b*x**k*exp(beta))

    Args:
        baseline (dict) : dictionary with parameter 'k' and 'b'
        x (np.ndarray) : (n,) array of survival times to evaluate
        beta (float) : log hazard rate
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    if 'b' in baseline.keys() and 'k' in baseline.keys():
        b = baseline['b']
        k = baseline['k']
    else:
        raise Exception('Weibull: param "b" and "k" not in parameter keys...')
    
    return np.exp(-np.exp(beta)*b*x**k)

def BayesFactorLowerThanHR(
        posterior_samples: np.ndarray, 
        prior_cdf: float, 
        HR: float
    ) -> float:
    """
    +==========================================================================+
    Determination of BayesFactor for hazard rate lower than HR by counting 
    samples below cut-off HR from posterior distibution to determine an 
    empirical CDF and comparing it to the prior CDF.

    Args:
        posterior_samples (np.ndarray) : (N,) array with posterior samples
        prior_prob (float) : prior_prob cdf until cutoff
        HR (float) : hazard rate
    Returns:
        float : BayesFactor for hazard rate lower then HR
    +==========================================================================+
    """
    numSamples = posterior_samples.size
    post_prob = (posterior_samples < np.log(HR)).sum()/numSamples + 0.5/numSamples
    # add numerical error 
    return post_prob/prior_cdf
        

if __name__ == "__main__":

    pass
