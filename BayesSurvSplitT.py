import pandas as pd
import pymc as pm
import numpy as np

def loglwb_splitT(
        t_data: pm.CallableTensor, 
        beta1: pm.CallableTensor, 
        beta2: pm.CallableTensor,
        T_cut: pm.CallableTensor,
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
        t (pm.CallableTensor) : (N,2) "t", vector of drop/censoring times
                                      "StartTime", vector of start times
        beta1 (pm.CallableTensor) : (1,) hazard rate subject t_start<t_cut
        beta2 (pm.CallableTensor) : (1,) hazard rate subject t_start>=t_cut
        T_cut (pm.CallableTensor) : (1,) cut_off time
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

    # betas
    expB1 = pm.math.exp(beta1)
    expB2 = pm.math.exp(beta2)

    # determine group
    T_before = (t_data[:,1] < T_cut).astype(int)
    T_after = 1 - T_before

    # get time
    t = t_data[:,0]

    # baseline Hazard Weibull
    # trial
    pll = pm.math.sum(
        reference_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) - b*t**k) # true deads                 
        - (1-events)*b*t**k) # alive
    )  

    # real, beta1
    pll += pm.math.sum(
        T_before*(
            test_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) + beta1 - expB1*b*t**k) # true deads                 
            - (1-events)*expB1*b*t**k) # alive
        ) 
    ) 

    # real, beta2
    pll += pm.math.sum(
        T_after*(
            test_id*(events*(pm.math.log(b*k + eps) + (k-1)*pm.math.log(t) + beta2 - expB2*b*t**k) # true deads                 
            - (1-events)*expB2*b*t**k) # alive
        )  
    ) 
    
    return pll


def LifeTimesSplitT_WB(
        data: pd.DataFrame, 
        beta1_params: tuple[float, float]=(0.,2.), 
        beta2_params: tuple[float, float]=(0.,2.), 
        T_cut_param: float=5.,
        b_params: tuple[float, float]=(-2.,3.), 
        k_params: tuple[float, float]=(0.9,1.1),
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

        # protected names: 'EventOrCensoredAtT', 'Test', 'Reference', 't', 
        # protected names: 'StartTime'
        assert ("EventOrCensoredAtT" in data.columns), "'EventOrCensoredAtT' not in columns"
        assert ("Test" in data.columns),         "'Test' not in columns"
        assert ("StartTime" in data.columns),    "'TStartTime' not in columns"
        assert ("Reference" in data.columns),    "'Reference' not in columns"
        assert ("t" in data.columns),            "'t' not in columns"
        assert ("Test" in data.columns),         "'Test' not in columns"
    
        start_time = pm.Data("start_time", data["StartTime"])
        test_id = pm.Data("test_id", data["Test"])
        reference_id = pm.Data("reference_id", data["Reference"])
        events = pm.Data("events_id", data["EventOrCensoredAtT"])
        
        # parse data t, StartTime: order matters!
        obs_data = pm.Data(
            "obs_ts", data[["t", "StartTime"]], dims=("t", "StartTime")
        )

        # set up priors
        beta1 = pm.Normal('beta1', mu=beta1_params[0], sigma=beta1_params[1])
        beta2 = beta1 + pm.Normal('beta2', mu=beta2_params[0], sigma=beta2_params[1])
        T_cut = pm.DiscreteUniform(
            'T_cut', lower=3, upper=start_time.max()-2,
        )
        b = pm.LogNormal('b', mu=b_params[0], sigma=b_params[1])
        k = pm.Uniform('k', lower=k_params[0], upper=k_params[1])

        # set up auxilliaries
        aux1 = (start_time <= T_cut).astype(int)*test_id
        aux2 = (start_time > T_cut).astype(int)*test_id
        n1 = pm.Deterministic("n1", pm.math.sum(aux1))
        n2 = pm.Deterministic("n2", pm.math.sum(aux2))

        # set up custom pll
        pm.DensityDist("likeli", beta1, beta2, T_cut, b, k, 
                       test_id, reference_id, events, 
                       logp=loglwb_splitT, observed=obs_data)

    return model

def survWB_splitT( 
        params: dict, 
        x: np.ndarray, 
    ) -> np.ndarray:
    """
    +==========================================================================+
    Function to evaluate Weibull survival probability for parameter 'k' and 'b' 
    and beta1,2, for each part of the dataset. n1 
    is the number of samples belonging to beta1 and n2 is the number of 
    samples belonging to beta2. 

    p(x) = n1/(n1+n2)*exp(-b*x**k*exp(beta1)) + 
                        n2/(n1+n2)*exp(-b*x**k*exp(beta2))

    Args:
        params (dict) : dictionary with parameter 'k' and 'b'
        x (np.ndarray) : (n,) array of survival times to evaluate
        beta1 (float) : log hazard rate of part 1
        beta2 (float) : log hazard rate of part 2
    Returns:
        np.ndarray : survival probabilities for x
    +==========================================================================+
    """
    n1 = params["n1"]
    n2 = params["n2"]
    b = params['b']
    k = params['k']
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    return p1*np.exp(-np.exp(beta1)*b*x**k)+ p2*np.exp(-np.exp(beta2)*b*x**k)