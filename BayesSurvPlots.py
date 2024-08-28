import numpy as np
from scipy.stats import norm
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from BayesSurv import BayesFactorLowerThanHR

def plotPosteriorBetaBayesFactors(PosteriorsT, beta_params, HRs, ax):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot the probability that HR > X over time, prior is norm
    ts = np.array([PosteriorsT[i][0] for i in range(len(PosteriorsT))])
    beta_prior = norm(beta_params[0], beta_params[1])
    color = cm.Reds(np.linspace(0.2, 1, len(HRs)))
    num_steps = len(PosteriorsT)

    for c, HR in zip(color, HRs):
        bf = np.array([BayesFactorLowerThanHR(
            PosteriorsT[i][1].beta.values, beta_prior.cdf(HR), HR
        ) for i in range(num_steps)])
        ax.semilogy(ts, bf, color=c)

    ax.legend(["HR < {}".format(HR) for HR in HRs])
    ax.set_xlabel('time [months]')
    ax.set_ylabel('BF_01')

    return ax

def plotPosteriorBetaCI(PosteriorsT, HRs, ax):
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
            PosteriorsT[i][1].quantile([CF[0]]).beta.values
        ) for i in range(num_steps)]).T[0]
        y2 = np.array([np.exp(
            PosteriorsT[i][1].quantile([CF[1]]).beta.values
        ) for i in range(num_steps)]).T[0]
        ax[0].fill_between(ts, y1, y2, alpha=0.15, 
                           color='b', label='_nolegend_')
        ax[0].plot(ts, y1, color = c)
        ax[0].plot(ts, y2, color = c, label='_nolegend_')

    ax[0].legend(["CI {}".format(Conf) for Conf in Confs])
    ax[0].set_xlabel('time [months]')
    ax[0].set_ylabel('beta')

    # plot the probability that HR > X over time
    color = cm.Reds(np.linspace(0.2, 1, len(HRs)))

    for c, HR in zip(color, HRs):
        y1 = np.array([(
            PosteriorsT[i][1].beta.values < np.log(HR)
        ).sum() / \
            PosteriorsT[i][1].beta.values.size for i in range(num_steps)]).T
        ax[1].plot(ts, y1, color=c)

    ax[1].legend(["HR < {}".format(HR) for HR in HRs])
    ax[1].set_xlabel('time [months]')
    ax[1].set_ylabel('probability HR < X')

    return ax

def plotKMPosteriorFits(PosteriorsT, dataDF, func, ax, num_shows):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot the 50th quantile
    Tend = PosteriorsT[-1][0]
    num_steps = len(PosteriorsT)
    jump = num_steps // num_shows
    colorT = cm.Oranges(np.linspace(0.2, 1, num_shows))
    colorR = cm.Purples(np.linspace(0.2, 1, num_shows))

    # check if beta in parameter list
    if 'beta' in PosteriorsT[0][1].quantile([0.5]).keys():
        pass
    else:
        raise Exception('beta param not in posterior keys...')

    for iter in range(0, num_steps, jump):

        # grab values
        j = iter + jump - 1
        baseline = dict()
        for key in list(PosteriorsT[j][1].quantile([0.5]).keys()):
            if key == 'beta':
                beta = PosteriorsT[j][1].quantile([0.5])[key].values
            else:
                baseline[key] = PosteriorsT[j][1].quantile([0.5])[key].values
        T = PosteriorsT[j][0]

        # display result
        print("==================================================================")
        print("T = {}".format(T))
        print("==================================================================")
        for key in list(baseline.keys()):
            print(f'{key} (50th): \t\t {baseline[key]}')
        print(f'beta (50th): \t\t {beta}')
        print(f'exp(beta) (50th): \t {1/np.exp(beta)}')
        print

        # set up arrays for plotting
        xT = np.linspace(0,Tend,100)

        # evaluate survivals
        surv_ref = func(baseline, xT, 0)
        surv_test = func(baseline, xT, beta) 
        survs = [surv_test, surv_ref]
        kind = ['Test', 'Reference']

        # plot KaplanMeier curves up until T
        color = ['b', 'r']
        Ts = [T, Tend]
        plotKaplanMeier(dataDF, color, ax[j // jump], kind, Ts)

        # plot survival functions at T
        color = [colorR[j // jump], colorT[j // jump]]
        for i in range(2):
            ax[j // jump].plot(xT, survs[i]/survs[i][0], color=color[i])
        
        ax[j // jump].legend(['Test-KM, T={:.2f}'.format(T), 'Ref-KM', 'Test-fit', 'Ref-fit'])
        ax[j // jump].set_xlabel('time [months]')
        ax[j // jump].set_ylabel('survival prob')

    return ax

def plotPosteriors(PosteriorsT,
                   key,
                   ax, 
                   num_shows, 
                   beta_prior, 
                   key_prior, 
                   xlims_nu = [0, 0.1],
                   xlims_beta = [-1, 1]):
    """
    +==========================================================================+

    +==========================================================================+
    """
    # plot resulting posterior
    num_steps = len(PosteriorsT)
    jump = num_steps // num_shows
    colorBeta = cm.BuGn(np.linspace(0.2, 1, num_shows))
    colorNu = cm.YlOrRd(np.linspace(0.2, 1, num_shows))

    # plot prior
    plot_pymc(pm.LogNormal.dist(mu=key_prior[0],sigma=key_prior[1], size=1000), ax[1], xlims_nu)
    plot_pymc(pm.Normal.dist(mu=beta_prior[0],sigma=beta_prior[1], size=1000), ax[0], xlims_beta)
    xt = [0]
    leg_list = ['prior']

    for iter in range(0, num_steps, jump):

        # select color
        i = iter + jump - 1
        cBeta = colorBeta[i // jump]
        cNu = colorNu[i // jump]

        # plot beta
        lam = PosteriorsT[i][1].quantile([0.5]).beta.values
        if lam < 0:
            lam = 0.001
        grid, pdf = az.kde(PosteriorsT[i][1].beta.values.flatten())
        ax[0].plot(grid, pdf/np.max(pdf), color=cBeta)
        ax[0].set_xlabel('value [a.u.]')
        ax[0].set_ylabel('pdf [a.u.]')
        ax[0].set_xlim(xlims_beta)
        ax[0].legend(['T={:.2f}'.format(PosteriorsT[i][0])])

        # plot nu
        grid, pdf = az.kde(PosteriorsT[i][1][key].values.flatten())
        xt.append(grid[np.argmax(pdf)])
        ax[1].plot(grid, pdf/np.max(pdf), color=cNu)
        ax[1].set_xlabel('value [a.u.]')
        ax[1].set_ylabel('pdf [a.u.]')
        ax[1].set_xlim(xlims_nu)

        # create legend
        leg_list.append('t={:.2f}'.format(PosteriorsT[i][0]))

    ax[0].set_title('beta')
    ax[1].set_title('nu')
    ax[0].legend(leg_list)
    ax[1].legend(leg_list)
    ax[0].vlines(0, 0, 1, color='k')

    return ax

def plotKaplanMeier(DFX, colors, ax, cols=['Test'], Ts=[-1.]):
    """
    +==========================================================================+

    +==========================================================================+
    """
    for col, color, T in zip(cols, colors, Ts):
        DF = DFX[DFX[col] == 1] # get cohort
        num_obs, num_col = DF.shape
        
        # copy into numpy array (times are float objects)
        if num_col > 2:
            data = np.zeros((num_obs, 3))
            data[:,0] = DF['EndTime'].astype(float).values
            data[:,1] = DF['StartTime'].astype(float).values
            data[:,2] = DF['Censored'].astype(int).values
        else:
            data = np.zeros((num_obs, 2))
            data[:,0] = DF['EndTime'].astype(float).values
            data[:,1] = DF['Censored'].astype(int).values

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
            data[abs_time > T, 0] = abs_time[abs_time > T] - T
            #data = data[data[:,0] < T]
        else:
            T = 1e8

        # find the number of subjects with endpoint, display, and sort
        M = len(data[(abs_time < T)])
        print("+=============================================================+")
        print(f"KaplanMeier Plot {col}: T={T}")
        print(f"Number of current subjects {N}")
        print(f"Number of subjects censored {M}")
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
                    # case 1: uncensored subject, no change in curve
                    # if censored, fewer subjects, but no change in emperical 
                    # survival prob
                    n_curr -= 1
                elif dat[0] == data[j+1][0]:
                    # case 2: equal timing to next subject and censored 
                    # no change in survival 
                    d_curr += 1
                    n_curr -= 1
                else:
                    # case 3: censored, change curve
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
    ax.legend(['Test', 'Reference'])
    ax.set_xlabel('time of X')
    ax.set_ylabel('Survival')

    return ax

def plot_pymc(distr, ax=None, xlims=None):
    """
    +==========================================================================+

    +==========================================================================+
    """
    if ax is None:
        _, ax = plt.subplots()
    if xlims is None:
        samples = pm.draw(distr)
        x = np.linspace(min(samples), max(samples), 1000)
    else:
        x = np.linspace(xlims[0], xlims[1], 1000)
    pdf = np.exp(pm.logp(distr,x)).eval()

    ax.plot(x, pdf/np.max(pdf) )
    ax.set_xlabel('value [a.u.]')
    ax.set_ylabel('pdf [a.u.]')

    return ax

if __name__ == "__main__":

    pass