import numpy as np
from scipy.stats import norm
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

SMALL_SIZE = 13.5
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 2


from BayesSurv import (
    BayesFactorLowerThanHR, 
    ProbLowerThanHR, 
    BayesFactorBetweenHR
)

def plotPosteriorBetaBayesFactors(
        PosteriorsT, 
        beta_params, 
        HRLs, 
        HRHs, 
        ax
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot the probability that HR > X over time, prior is norm
    ts = np.array([0.]+[PosteriorsT[i][0] for i in range(len(PosteriorsT))])
    beta_prior = norm(beta_params[0], beta_params[1])
    color = cm.BrBG(np.linspace(0, 1, len(HRLs+HRHs)))
    num_steps = len(PosteriorsT)

    HRs = HRLs + HRHs
    # 0
    ax.hlines(1, ts[0], ts[-1], color='k', label='_nolegend_', linewidth=1)

    for c, HR in zip(color[:len(HRLs)], HRLs):
        #prior_prob = beta_prior.cdf(np.log(HR)) - beta_prior.cdf(-np.log(HR)) 
        prior_prob = beta_prior.cdf(np.log(HR))
        bf = np.array([1.]+[1/BayesFactorLowerThanHR(
            PosteriorsT[i][1].beta.values.flatten(), 
            prior_prob, 
            HR,
        ) for i in range(num_steps)])
        ax.semilogy(ts, bf, color=c)
    
    for c, HR in zip(color[len(HRLs):], HRHs):
        #prior_prob = beta_prior.cdf(np.log(HR)) - beta_prior.cdf(-np.log(HR)) 
        prior_prob = beta_prior.cdf(np.log(HR))
        bf = np.array([1.]+[BayesFactorLowerThanHR(
            PosteriorsT[i][1].beta.values.flatten(), 
            prior_prob, 
            HR,
        ) for i in range(num_steps)])
        ax.semilogy(ts, bf, color=c)

    style = ['-.', '--', '-', ':', '-.']
    for i, BF in enumerate([3,10,30,100,300]):
        ax.hlines(
            BF, 
            ts[0], 
            ts[-1], 
            color='k', 
            alpha=0.3, 
            linestyle=style[i], 
            linewidth=1
        )

    for i, BF in enumerate([1/3,1/10,1/30,1/100,1/300]):
        ax.hlines(
            BF, 
            ts[0], 
            ts[-1], 
            color='k', 
            alpha=0.3, 
            linestyle=style[i], 
            linewidth=1
        )

    ax.legend([f"HR>{HR}" for HR in HRs], loc=(1.04, 0))
    ax.set_xlabel('time [months]')
    ax.set_ylabel('BF_01')
    ax.autoscale(enable=True, axis='x', tight=True)

    return ax

def plotPosteriorBetaBayesFactors_inBetween(
        PosteriorsT, 
        beta_params, 
        HRs, 
        ax
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot the probability that HR > X over time, prior is norm
    ts = np.array([PosteriorsT[i][0] for i in range(len(PosteriorsT))])
    beta_prior = norm(beta_params[0], beta_params[1])
    color = cm.Reds(np.linspace(0.2, 1, len(HRs)))
    num_steps = len(PosteriorsT)

    # 0
    ax.hlines(1, ts[0], ts[-1], color='k', label='_nolegend_', linewidth=1)

    for c, HR in zip(color, HRs):
        prior_prob = beta_prior.cdf(np.log(HR)) - beta_prior.cdf(-np.log(HR)) 
        bf = np.array([BayesFactorBetweenHR(
            PosteriorsT[i][1].beta.values.flatten(), 
            prior_prob, 
            np.exp(-np.log(HR)),
            HR
        ) for i in range(num_steps)])
        ax.semilogy(ts, bf, color=c)

    style = ['-.', '--', '-']
    for i, BF in enumerate([3,10,30]):
        ax.hlines(
            BF, 
            ts[0], 
            ts[-1], 
            color='k', 
            alpha=0.3, 
            linestyle=style[i], 
            linewidth=1
        )

    for i, BF in enumerate([1/3,1/10,1/30]):
        ax.hlines(
            BF, 
            ts[0], 
            ts[-1], 
            color='k', 
            alpha=0.3, 
            linestyle=style[i], 
            linewidth=1
        )

    ax.legend([f"{np.exp(-np.log(HR)):.2f}>HR>{HR}" for HR in HRs], loc=(1.04, 0))
    ax.set_xlabel('time [months]')
    ax.set_ylabel('BF_01')
    ax.autoscale(enable=True, axis='x', tight=True)

    return ax

def betaBayesFactors(
        PosteriorsT, 
        beta_params, 
        HRs
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot the probability that HR > X over time, prior is norm
    ts = np.array([PosteriorsT[i][0] for i in range(len(PosteriorsT))])
    beta_prior = norm(beta_params[0], beta_params[1])
    num_steps = len(PosteriorsT)

    outfileBF = pd.DataFrame(index=ts)
    outfileCI = pd.DataFrame(index=ts)
    outfileCF = pd.DataFrame(index=ts)

    for HR in zip(HRs):
        numSamples = len(PosteriorsT[0][1].beta.values.flatten())
        bf = np.array([BayesFactorLowerThanHR(
            PosteriorsT[i][1].beta.values.flatten(), 
            beta_prior.cdf(HR), 
            HR
        ) for i in range(num_steps)])
        ci = np.array([ProbLowerThanHR(
            PosteriorsT[i][1].beta.values.flatten(), 
            HR
        ) for i in range(num_steps)])
        outfileBF[HR] = bf
        outfileCI[HR] = ci

    for p in [0.5, 0.1, 0.05, 0.025, 0.01]:
        cf = [np.exp(
                np.sort(
                    PosteriorsT[i][1].beta.values.flatten()
                )[[int(numSamples*p), int(numSamples*(1-p))]]
              ) for i in range(num_steps)]

        outfileCF[p] = cf

    return outfileBF, outfileCI, outfileCF

def plotPosteriorBetaCI(
        PosteriorsT, 
        HRs, 
        ax,
        param_name = 'beta'
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    num_steps = len(PosteriorsT)
    ts = np.array([PosteriorsT[i][0] for i in range(num_steps)])

    # plot beta confidence intervals over time
    Confs = [0.975, 0.95, 0.9, 0.75, 0.5]
    CFs = [[1-x, x] for x in Confs]
    colorBeta = cm.BuGn(np.linspace(0.2, 1, len(Confs)))

    for c, CF in zip(colorBeta, CFs):
        y1 = np.array([np.exp(
            PosteriorsT[i][1].quantile([CF[0]])[param_name].values
        ) for i in range(num_steps)]).T[0]
        y2 = np.array([np.exp(
            PosteriorsT[i][1].quantile([CF[1]])[param_name].values
        ) for i in range(num_steps)]).T[0]
        ax.plot(ts, y1, color = c,)
        ax.plot(ts, y2, color = c, label='_nolegend_',)
        ax.fill_between(ts, y1, y2, alpha=0.15, 
                           color='b', label='_nolegend_')

    for HR in HRs:    
        ax.hlines(HR,
                  ts[0], 
                  ts[-1], 
                  color='k', 
                  alpha=0.3, 
                  label='_nolegend_', 
                  linewidth=1
        )

    ax.legend(["CI {}".format(Conf) for Conf in Confs], loc=(1.04, 0))
    ax.set_xlabel('time [months]')
    ax.set_ylabel('HR')
    ax.autoscale(enable=True, axis='x', tight=True)

    return ax

def plotKMPosteriorFits(
        T,
        Tend,
        PosteriorsT, 
        dataDF, 
        func, 
        ax, 
        plot_ref_intervals=False,
        prior=False,
        text="",
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    colorT = cm.Purples(np.linspace(0.2, 1, 3))
    colorR = cm.Oranges(np.linspace(0.2, 0.5, 3))
    
    # set up arrays for plotting
    xT = np.linspace(0,Tend,100)
    qts = [0.05, 0.95] if prior else [0.5, 0.05, 0.95]
    surv_test = np.zeros((len(qts),len(xT)))
    surv_ref = np.zeros((len(qts),len(xT)))

    posterior_models_test = eval_joint_posterior(
        func, 
        PosteriorsT, 
        xT, 
        ref=False
    )
    posterior_models_ref = eval_joint_posterior(
        func, 
        PosteriorsT, 
        xT, 
        ref=True
    )

    # eval qts
    for i in range(len(qts)):
        sorted_post_models = np.sort(posterior_models_test, axis=0)
        surv_test[i] = sorted_post_models[int(qts[i]*len(sorted_post_models))]
        
        sorted_post_models = np.sort(posterior_models_ref, axis=0)
        surv_ref[i] = sorted_post_models[int(qts[i]*len(sorted_post_models))]

        
    kind = ['Test', 'Reference']
    survs = [surv_test[0], surv_ref[0]] # median
    # plot KaplanMeier curves up until T

    # plot CIs
    skip = 0 if prior else 1
    if plot_ref_intervals:    
        for l_ref, u_ref in zip(surv_ref[skip::2], surv_ref[skip+1::2]):
            y1 = l_ref
            y2 = u_ref
            ax.plot(xT, y1, color = colorR[1], label='_nolegend_',)
            ax.plot(xT, y2, color = colorR[1], label='_nolegend_',)
            ax.fill_between(xT, y1, y2, alpha=0.15, 
                            color=colorR[1], label='_nolegend_')
          
    for l_test, u_test  in zip(surv_test[skip::2], surv_test[skip+1::2]):
        y1 = l_test
        y2 = u_test
        ax.plot(xT, y1, color = colorT[1], label='_nolegend_',)
        ax.plot(xT, y2, color = colorT[1], label='_nolegend_',)
        ax.fill_between(xT, y1, y2, alpha=0.15, 
                        color=colorT[1], label='_nolegend_')
     

    # plot survival functions at T
    color = [colorT[-1], colorR[-1]]
    for i in range(2):
        if not prior: ax.plot(xT, survs[i]/survs[i][0], color=color[i])

    Ts = [T, Tend]
    color = ['b', 'r']
    plotKaplanMeier(dataDF, kind, color, ax, Ts)
    
    ax.set_xlabel('time [months]')
    ax.set_ylabel('survival prob')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.text(0.9*Tend,0.9, f"T={T:.2f}")
    if text: ax.text(-0.1*Tend,1.1, text, fontweight="bold", fontsize=BIGGER_SIZE)

    return ax

def plotPosteriors(
        PosteriorsT,
        ax, 
        show_js, 
        key_params, 
        xlims_baseline = [[-1, 1], [0, 0.1], [0.5, 1.5]],
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot resulting posterior
    num_shows = len(show_js)
    if num_shows > 1:
        colorBeta = cm.BuGn(np.linspace(0.2, 1, num_shows))
        colorBaselines = cm.YlOrRd(np.linspace(0.2, 1, num_shows))
    else:
        colorBeta = cm.BuGn(np.array([1.]))
        colorBaselines = cm.YlOrRd(np.array([1.]))

    # plot prior
    for j, key in enumerate(key_params):
        grid, pdf = az.kde(PosteriorsT[0][2][key].values.flatten())
        ax[j].plot(grid, pdf/np.max(pdf),color='lightgrey')
    xt = [0]
    leg_list = ['prior']

    for i, iter in enumerate(show_js):

        cBeta = colorBeta[i]
        cBaseline = colorBaselines[i]
        colors = [cBaseline for _ in range(len(key_params))]
        colors[0] = cBeta

        # plot vars
        for j, key in enumerate(key_params):
            grid, pdf = az.kde(PosteriorsT[iter][1][key].values.flatten())
            xt.append(grid[np.argmax(pdf)])
            ax[j].plot(grid, pdf/np.max(pdf), color=colors[j])
            ax[j].set_xlabel('value [a.u.]')
            ax[j].set_ylabel('pdf [a.u.]')
            ax[j].set_xlim(xlims_baseline[j])

        # create legend
        leg_list.append('t={:.2f}'.format(PosteriorsT[iter][0]))

    for j, key in enumerate(key_params):
        ax[j].set_title(key)
    for ax_i in ax:
        ax_i.legend(leg_list, loc=(1.04, 1))
    ax[0].vlines(0, 0, 1, color='k')

    return ax

def plotKaplanMeier(
        DFX: pd.DataFrame, 
        cols: list,
        colors: list, 
        ax: plt.axis, 
        Ts: list | None = None,
    ):
    """
    +==========================================================================+
    Plot emperical Kaplan-Meier survival curve


    Args:
        DFX (pd.DataFrame) : data frame with data
        cols (list) : list of column names to plot
        colors (list) : colors to use. one per col.
        ax (plt.axis) : axis object to plot onto
        Ts list() : list of end times per column

    Returns:
        Updated plt.axis object with Kaplan-Meier curve.
    +==========================================================================+
    """
    if Ts is None:
        Ts = [-1 for col in cols]
    for col, color, T in zip(cols, colors, Ts):
        DF = DFX[DFX[col] == 1] # get cohort
        num_obs, num_col = DF.shape
        
        # copy into numpy array (times are float objects)
        if num_col > 2:
            data = np.zeros((num_obs, 3))
            data[:,0] = DF['EndTime'].astype(float).values
            data[:,1] = DF['StartTime'].astype(float).values
            data[:,2] = DF['Event'].astype(int).values
        else:
            data = np.zeros((num_obs, 2))
            data[:,0] = DF['EndTime'].astype(float).values
            data[:,1] = DF['Event'].astype(int).values

        # remove subjects who have not started
        if T != -1 and DF.shape[1] > 2:
            data = data[data[:,1] < T]
        N = len(data) #all current subjects

        # create absolute time dummy
        abs_time = data[:,0].copy()
        if DF.shape[1] > 2:
            abs_time += data[:,1]

        # reset running subjects to endpoint T
        if T != -1:
            #data = data[dummy < T]
            data[abs_time > T,-1] = 0
            data[abs_time > T, 0] = T - data[abs_time > T,1]
            #data = data[data[:,0] < T]
        else:
            T = np.max(abs_time)+1

        # find the number of subjects with endpoint, display, and sort
        M = len(data[(abs_time < T)])
        print("+=============================================================+")
        print(f"KaplanMeier Plot {col}: T={T}")
        print(f"Number of current subjects {N}")
        print(f"Number of subjects Event/Censored {M}")
        print("+=============================================================+")
        data = data[data[:,0].argsort()]

        # init
        x = [0]
        y = [1]
        y_curr = 1
        n_curr = N
        d_curr = 0

        for j, dat in enumerate(data):

            if j < N-1:
                if dat[-1] == 0: 
                    # case 1: no-event subject, no change in curve
                    # if censored, fewer subjects, but no change in emperical 
                    # survival prob
                    n_curr -= 1
                elif dat[0] == data[j+1][0]:
                    # case 2: equal timing to next subject and no-event
                    # no change in survival 
                    d_curr += 1
                    n_curr -= 1
                else:
                    # case 3: event, change curve
                    # change survival prob and update curve

                    # record step: left edge
                    x.append(dat[0])
                    y.append(y_curr)

                    d_curr += 1
                    y_curr *= (1 - d_curr/n_curr)
                    
                    n_curr -= 1
                    d_curr = 0

                    # record step: right edge
                    x.append(dat[0])
                    y.append(y_curr)

            elif j == N-1:
                # last element

                # record step: left edge
                if dat[0] > T:
                    x.append(T) # never happens?
                else:
                    x.append(dat[0])
                y.append(y_curr)

                d_curr += 1
                y_curr *= (1 - d_curr/n_curr)
                
                n_curr -= 1
                d_curr = 0

                # record step: right edge
                if dat[0] > T:
                    x.append(T) # never happens?
                else:
                    x.append(dat[0])
                y.append(y_curr)
        
        if num_col > 2:
            ax.plot(x[:-1], np.array(y[:-1]), color=color)
        else:
            ax.plot(x, np.array(y), color=color)

    # ax.set_title('KM-curve of data, censored subjects not displayed')
    ax.legend(['Test', 'Reference'], loc=(1.04, 0))
    ax.set_xlabel('time of X')
    ax.set_ylabel('Survival')

    return ax

def plot_pymc(
        distr, 
        ax=None, 
        xlims=None,
        log=False,
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    if ax is None:
        _, ax = plt.subplots()
    if xlims is None:
        samples = pm.draw(distr)
        if not log: x = np.linspace(min(samples), max(samples), 1000)
        else: x = np.logspace(min(samples), max(samples), 1000)
    else:
        if not log: x = np.linspace(xlims[0], xlims[1], 1000)
        else: x = np.logspace(xlims[0], xlims[1], 1000)
    pdf = np.exp(pm.logp(distr,x)).eval()

    if not log: ax.plot(x, pdf/np.max(pdf)) 
    else: ax.semilogx(x, pdf/np.max(pdf))
    
    ax.set_xlabel('value [a.u.]')
    ax.set_ylabel('pdf [a.u.]')

    return ax

def eval_joint_posterior(
        func, 
        posterior, 
        t, 
        ref=False
    ):
    """
    +==========================================================================+

    +==========================================================================+
    """
    params = dict()
    for key in posterior.keys():
        params[key] = posterior[key].values.flatten()
        if ref and "beta" in key: params[key][:] = 0
        numModels = len(params[key])

    model_evals = np.zeros((numModels, len(t)))
    for i in range(numModels):
        baseline_i = dict()
        for key in params.keys():
            baseline_i[key] = params[key][i]
        model_evals[i] = func(baseline_i, t)

    return model_evals   
    

if __name__ == "__main__":
    pass