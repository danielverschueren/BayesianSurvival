import numpy as np
import pymc as pm
import pandas as pd
import arviz as az

# turn of df copy warning
pd.options.mode.chained_assignment = None 

def resetDataToT(
        dataX: pd.DataFrame, 
        T: float,
    ) -> pd.DataFrame:
    """
    +==========================================================================+
    Reset historic data to time point T, so simulate real time updates.

    The input dataframe needs a specific structure, with subjects on the 
    rows, and columns:
        'StartTime' for the time at which subject had started,
        'EndTime' for the time at which the subject was event/dropped out, 
        'Test' to indicate if subject belongs to the test or trial data
        'Event' to indicate if subject had an event or not

    Args:
        dataX (pd.DataFrame) :  data frame with data 
        T (float) : time to reset endtime of subjects to

    Returns:
        pd.DataFrame with subjects that started before T and life times
    +==========================================================================+
    """
    # get data
    data = dataX[dataX['StartTime'] < T]
    t = data['EndTime'].astype(float).values
    t_start = data['StartTime'].astype(float).values
    t_abs = t_start + t
    real = data['Test'].astype(int).values

    # update t for active unevent subjects
    EventOrCensoredAtT = data['Event'].astype(int).values
    EventOrCensoredAtT[(t_abs > T) & (real == 1)] = 0 # update times for these
    t[(t_abs > T) & (EventOrCensoredAtT == 0) & (real == 1)] = \
            T - t_start[(t_abs > T) & (EventOrCensoredAtT == 0) & (real == 1)]

    # check if all t's are valid (>= 0)
    assert (t > 0).all(), "Some lifetimes t <= 0."

    # put all in dataFrame
    data['t'] = pd.Series(t, index=data.index)
    data['EventOrCensoredAtT'] = pd.Series(EventOrCensoredAtT, index=data.index)

    # print info
    print("+=============================================================+")
    print('T = {}'.format(T))
    print("+=============================================================+")
    print('Current subjects: \t\t\t{}'.format(len(t[real == 1])))
    print('Event RW prior to T: \t\t\t{}'.format(
        np.sum(EventOrCensoredAtT[real == 1])
    ))

    return data
    
def loglp(
        t: pm.CallableTensor, 
        beta: pm.CallableTensor, 
        nu: pm.CallableTensor, 
        test_id: pm.CallableTensor,
        reference_id: pm.CallableTensor,
        events: pm.CallableTensor,
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
        events (pm.CallableTensor) : (N,) subject event (1) or not (0)
    
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
    pll = pm.math.sum(
        reference_id*(events*(pm.math.log(nu + eps) - nu*t) # true deads               
        - (1-events)*nu*t) # alive
    )  

    # real
    pll += pm.math.sum(
        test_id*(events*(pm.math.log(nu + eps) + beta - (nu*t)*expB) # true deads                 
        - (1-events)*nu*t*expB)# alive 
    ) 
    
    return pll

def LifeTimesFull_Exp(
        data: pd.DataFrame, 
        beta_params: tuple[float, float]=(0.,0.5), 
        nu_params: tuple[float, float]=(-2.,0.5),
    ) -> pm.Model:
    """
    +==========================================================================+
    Function to instantiate a pymc model consturcted using an exponential 
    lifetimes distribution, constant hazard rate, observed lifetimes t, data_id 
    to determine where subject's group.

    Args:
        data (pd.DataFrame) : (N,m) data array with the following columns:
                                'Test': "test_id", in test set (1) or not (0)
                                'Reference': "reference_id", in reference set 
                                             (1) or not (0)
                                'EventOrCensoredAtT: "events_id", wether sample  
                                               is event (1) or not (0)
                                't': "t", observed lifetimes 
                              Additional columns are allowed, not used.
        beta_params tuple(float, float): beta prior parameters: Norm(mu, sigma)
        nu_params tuple(float, float): nu prior parameters: LogNorm(mu, sigma)
    Returns:
        pm.Model pymc model to sample from            
    +==========================================================================+
    """
    with pm.Model() as model:

        # protected names: 'EventOrCensoredAtT', 'Test', 'Reference', 't'
        assert ("EventOrCensoredAtT" in data.columns),  "'EventOrCensoredAtT' not in columns"
        assert ("Test" in data.columns),         "'Test' not in columns"
        assert ("Reference" in data.columns),    "'Reference' not in columns"
        assert ("t" in data.columns),            "'t' not in columns"

        test_id = pm.Data("test_id", data["Test"])
        reference_id = pm.Data("reference_id", data["Reference"])
        events = pm.Data("events_id", data["EventOrCensoredAtT"])
        # parse data
        t = pm.Data("t", data["t"], dims="obs_id")

        # set up priors
        beta = pm.Normal('beta', mu=beta_params[0], sigma=beta_params[1]) #23
        nu = pm.LogNormal('nu', mu=nu_params[0], sigma=nu_params[1])

        # set up custom pll
        pm.CustomDist("likeli", beta, nu,
                      test_id, reference_id, events, logp=loglp, 
                      observed=t, dims='obs_id')

    return model

def loglwb(
        t: pm.CallableTensor, 
        beta: pm.CallableTensor, 
        b: pm.CallableTensor, 
        k: pm.CallableTensor,
        test_id: pm.CallableTensor,
        reference_id: pm.CallableTensor,
        events: pm.CallableTensor,
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
        events (pm.CallableTensor) : (N,) subject event (1) or not (0)
    
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
    pll = pm.math.sum(
        reference_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) - b*t**k) # true deads                 
        - (1-events)*b*t**k) # alive
    )  

    # real
    pll += pm.math.sum(
        test_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) + beta - expB*b*t**k) # true deads                 
        - (1-events)*expB*b*t**k) # alive
    ) 
    
    return pll

def LifeTimesFull_WB(
        data: pd.DataFrame, 
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
        data (pd.DataFrame) : (N,m) data array with the following columns:
                                'Test': "test_id", in test set (1) or not (0)
                                'Reference': "reference_id", in reference set 
                                             (1) or not (0)
                                'EventOrCensoredAtT: "events_id", wether sample  
                                             is event (1) or not (0)
                                't': "t", observed lifetimes 
                              Additional columns are allowed, not used.
        beta_params tuple(float, float): beta prior parameters: Norm(mu, sigma)
        b_params tuple(float, float): b prior parameters: LogNorm(mu, sigma)
        k_params tuple(float, float): k prior parameters: Unif(low, high)
    Returns:
        pm.Model pymc model to sample from     
    +==========================================================================+
    """
    with pm.Model() as model:

        # protected names: 'EventOrCensoredAtT', 'Test', 'Reference', 't'
        assert ("EventOrCensoredAtT" in data.columns), "'EventOrCensoredAtT' not in columns"
        assert ("Test" in data.columns),         "'Test' not in columns"
        assert ("Reference" in data.columns),    "'Reference' not in columns"
        assert ("t" in data.columns),            "'t' not in columns"

        test_id = pm.Data("test_id", data["Test"])
        reference_id = pm.Data("reference_id", data["Reference"])
        events = pm.Data("events_id", data["EventOrCensoredAtT"])
        # parse data
        t = pm.Data("t", data["t"], dims="obs_id")

        # set up priors
        beta = pm.Normal('beta', mu=beta_params[0], sigma=beta_params[1]) #23
        b = pm.LogNormal('b', mu=b_params[0], sigma=b_params[1])
        k = pm.Uniform('k', lower=k_params[0], upper=k_params[1])

        # set up custom pll
        pm.DensityDist("likeli", beta, b, k, 
                       test_id, reference_id, events, 
                       logp=loglwb, observed=t, dims='obs_id')

    return model

def loglgomp(
        t: pm.CallableTensor, 
        beta: pm.CallableTensor, 
        b: pm.CallableTensor, 
        eta: pm.CallableTensor,
        test_id: pm.CallableTensor,
        reference_id: pm.CallableTensor,
        events: pm.CallableTensor,
    ) -> pm.CallableTensor:
    """
    +==========================================================================+
    Survival Log likelihood for an Gompertz distribution of lifetimes between
    two populations identified with test_id or reference_id and constant hazard 
    beta difference between the populations

    Args:
        t (pm.CallableTensor) : (N,) vector of drop/censoring times
        beta (pm.CallableTensor) : (1,) hazard rate
        b (pm.CallableTensor) : (1,) Gompertz b param (distribution param)
        eta (pm.CallableTensor) : (1,) Gompertz eta param (distribution param)
        test_id (pm.CallableTensor) : (N,) subject in test set (1) or not (0)
        reference_id (pm.CallableTensor) : (N,) subject in reference set (1)
                                            or not (0)
        events (pm.CallableTensor) : (N,) subject event (1) or not (0)
    
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
    pll = pm.math.sum(
        reference_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) - eta*(pm.exp(b*t)-1)) # true deads                 
        - (1-events)*eta*(pm.exp(b*t)-1)) # alive
    )  

    # real
    pll += pm.math.sum(
        test_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) + beta - expB*eta*(pm.exp(b*t)-1)) # true deads                 
        - (1-events)*expB*eta*(pm.exp(b*t)-1)) # alive
    ) 
    
    return pll

def survExp(
        params: dict, 
        x: np.ndarray, 
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate exponential survival probability for parameter 'nu' and 
    'beta' in exponential distribution

    p(x) = exp(-nu*x*exp(beta))

    Args:
        parmas (dict) : dictionary with parameters
        x (np.ndarray) : (n,) array of survival times to evaluate
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    nu = params["nu"]
    beta = params["beta"]
    return np.exp(-nu*x*np.exp(beta))

def survWB( 
        params: dict, 
        x: np.ndarray, 
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate Weibull survival probability for parameters 'k' and 'b' 
    and 'beta' in Weibull distribution

    p(x) = exp(-b*x**k*exp(beta))

    Args:
        params (dict) : dictionary with parameter 'k' and 'b' and 'beta'
        x (np.ndarray) : (n,) array of survival times to evaluate
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    beta = params["beta"]
    b = params["b"]
    k = params["k"]
    return np.exp(-np.exp(beta)*b*x**k)

def survGomp( 
        params: dict, 
        x: np.ndarray, 
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate Gompertz survival probability for parameters 'eta' and 
    'b' and 'beta' in Gompertz distribution

    p(x) = exp(-eta*(exp(b*x)-1)*exp(beta))

    Args:
        params (dict) : dictionary with parameter 'eta' and 'b' and 'beta'
        x (np.ndarray) : (n,) array of survival times to evaluate
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    beta = params["beta"]
    b = params["b"]
    eta = params["eta"]
    return np.exp(-np.exp(beta)*eta*(np.exp(b*x)-1))

def BayesFactorLowerThanHR(
        posterior_samples: np.ndarray, 
        prior_cdf: float, 
        HR: float
    ) -> float:
    """
    +==========================================================================+
    Determination of BayesFactor for hazard rate lower than HR by counting 
    samples below cut-off HR from posterior distibution to determine an 
    empirical CDF and comparing it to the prior CDF. The evaluation is based on 
    the Savage-Dickey method of nested hypotheses.
        BF01 = p(H0|x)/p(H1|x) = BF0e/BF1e =
        BF01 = p(beta < c| x)/p(beta < c)/(p(beta >= c| x)/p(beta >= c))

    Args:
        posterior_samples (np.ndarray) : (N,) array with posterior samples
        prior_prob (float) : prior_prob cdf until cutoff
        HR (float) : hazard rate
    Returns:
        float : BayesFactor for hazard rate lower then HR
    +==========================================================================+
    """
    post_prob = ProbLowerThanHR(
        posterior_samples,
        HR
    )
    return (prior_cdf*(1-post_prob))/(post_prob*(1-prior_cdf))

def BayesFactorBetweenHR(
        posterior_samples: np.ndarray, 
        prior_cdf: float, 
        HR1: float,
        HR2: float
    ) -> float:
    """
    +==========================================================================+
    Determination of BayesFactor for hazard rate lower than HR by counting 
    samples below cut-off HR from posterior distibution to determine an 
    empirical CDF and comparing it to the prior CDF.
        BF01 = p(H0|x)/p(H1|x) = BF0e/BF1e =
        BF01 = p(c1 < beta < c2| x)/p(c1 < beta < c2)/
                (p(beta <= c1 or beta >= c2| x)/p(beta <= c1 or beta >= c2))

    Args:
        posterior_samples (np.ndarray) : (N,) array with posterior samples
        prior_prob (float) : prior_prob cdf until cutoff
        HR (float) : hazard rate
    Returns:
        float : BayesFactor for hazard rate lower then HR
    +==========================================================================+
    """
    assert (HR1 < HR2)
    post_prob_lowerL = ProbLowerThanHR(
        posterior_samples,
        HR1,
    )
    post_prob_lowerH = ProbLowerThanHR(
        posterior_samples,
        HR2,
    )
    post_prob = post_prob_lowerH - post_prob_lowerL
    return 1/(prior_cdf*(1-post_prob)/(post_prob*(1-prior_cdf)))

def ProbLowerThanHR(
        posterior_samples: np.ndarray, 
        HR: float
    ) -> float:
    """
    +==========================================================================+
    Determination of BayesFactor for hazard rate lower than HR by counting 
    samples below cut-off HR from posterior distibution to determine an 
    empirical CDF and comparing it to the prior CDF.

    Args:
        posterior_samples (np.ndarray) : (N,) array with posterior samples
        HR (float) : hazard rate
    Returns:
        float : BayesFactor for hazard rate lower then HR
    +==========================================================================+
    """
    hdi = False
    if hdi:
        prob = az.hdi(posterior_samples, hdi_prob=1)
    else:
        numSamples = posterior_samples.size
        dp = 0.5/numSamples
        post_prob = (posterior_samples < np.log(HR)).mean()
    
    # add numerical error 
    if post_prob > 1 - dp:
        post_prob = 1 - dp
    elif post_prob < dp:
        post_prob = dp
    elif np.isnan(post_prob):
        post_prob = dp
    return post_prob
        

if __name__ == "__main__":

    pass
