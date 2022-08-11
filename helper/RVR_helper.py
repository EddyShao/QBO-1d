import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator

import os.path
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from qbo1d import utils
from qbo1d import adsolver
from qbo1d import emulate
from qbo1d.stochastic_forcing import WaveSpectrum
from qbo1d.stochastic_forcing import sample_sf_cw

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

from sklearn_rvm.em_rvm import EMRVR

import joblib
import json



# Define the Dataloader

sfe = [3.7e-3, 3.8e-3, 3.2e-3, 3.8e-3, 3.8e-3]
sfv = [1e-8, 9e-8, 9e-8, 9e-10, 9e-8]
cwe = [32, 32, 40, 32, 32]
cwv = [225, 225, 225, 225, 225]
corr = [0.75, 0.75, 0.75, 0.75, -0.75]
seed = [int(21*9+8), int(21*9+7), int(21*6+15), int(21*12+5), int(21*2+10)]

para_mat = np.array([sfe, sfv, cwe, cwv, corr, seed]).T

def load_parameters(dir, para_id='1'):
    assert isinstance(para_id, str), 'para_id should be a string'

    dir = dir.strip()
    if dir[-1] == '/':
        path = dir + 'paras.json'
    else:
        path = dir +'/paras.json'

    with open(path, 'r') as f:
        object = json.load(f)
    return object[para_id]

def data_loader(state=1):
    '''
    Input: state(0~4)
    STATE = 0 -> old control
    STATE = 1 -> new control
    STATE = 2 -> different mean
    STATE = 3 -> different variance
    STATE = 4 -> anti-correlation(non-physical) 

    Output: (u, s, sf, cw, solver)
    '''

    # Load the data manually
    # it takes 40 seconds

    t_max = 360 * 108 * 86400
    nsteps = 360 * 108
    nspinup = 360 * 12
    ntot = int(nsteps - nspinup)

    torch.set_default_dtype(torch.float64)

    # scenario 0 (control)
    # --------------------
    solver = adsolver.ADSolver(t_max=t_max, w=3e-4)
    model = WaveSpectrum(solver, *para_mat[state])
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    return u, model.s, model.sf, model.cw, solver



# Data preprocessing

def dataset(u, s, sf, cw, K=1, nsteps=360*108, nspinup=360*12, test_size=0.8):
    U = u[nspinup:nsteps, :]
    SF = sf[nspinup:nsteps]
    CW = cw[nspinup:nsteps]

    # n_features of U = 73 + 2 = 75
    # n_features of S = 73
    U = torch.hstack([U[:, 1:-1], SF.view(-1, 1), CW.view(-1, 1)])
    S = s[nspinup:nsteps, :]

    sc_U = StandardScaler()
    sc_S = StandardScaler()

    # Here U is the features and s is the label
    U_train, U_test, s_train, s_test = train_test_split(U, S, test_size=test_size, random_state=42)

    sc_U.fit(U_train)
    U_train = sc_U.transform(U_train)
    U_test = sc_U.transform(U_test)
    U_train[:, -2:] = K * U_train[:, -2:]
    U_test[:, -2:] = K * U_test[:, -2:]

    sc_S.fit(s_train)
    s_train = sc_S.transform(s_train)
    s_test = sc_S.transform(s_test)

    return U_train, U_test, s_train, s_test, sc_U, sc_S

def save_np_array(array, filename):
    np.save(filename+'.npy', array)


def training_1d(U_train, U_test, s_train, s_test, level):
    assert level < 72 and level > 0
    
    s_train_level, s_test_level = s_train[:, level], s_test[:, level]
    
    with joblib.parallel_backend(backend='threading', n_jobs=16):
        rvr = EMRVR(kernel='rbf', gamma='auto', bias_used=True)
        rvr.fit(U_train, s_train_level)
        report_level = {
            str(level)+'_r2_train': rvr.score(U_train, s_train_level),
            str(level)+'_r2_test': rvr.score(U_test, s_test_level),
            str(level)+'_n_supported': int(rvr.relevance_.shape[0]),
        }
    
    return rvr, report_level


def training_multidim(U_train, U_test, s_train, s_test):
    # Multiregression
    with joblib.parallel_backend(backend='threading', n_jobs=16):
        rvr_1d = EMRVR()
        mr = MultiOutputRegressor(rvr_1d, n_jobs=8)
        mr.fit(U_train, s_train[:, 1:-1])
        
        prediction = mr.predict(U_test)
        zero_padding = np.zeros(prediction.shape[0]).reshape(-1, 1)
        prediction = np.hstack([zero_padding, prediction, zero_padding])
        
        rmse = rMSE(prediction, s_test, s_test.std()+1e-32)

        n_support_array = np.zeros(len(mr.estimators_))
        
        for i in range(len(mr.estimators_)):
            n_support_array[i] = mr.estimators_[i].relevance_.shape[0]

        report_multidim = {
                'multidim_r2_train': mr.score(U_train, s_train[:, 1:-1]),
                'multidim_r2_test': mr.score(U_test, s_test[:, 1:-1]),
                'multidim_mean_rmse': rmse.mean(),
                'mean_n_support': float(n_support_array.mean()),
                '1_r2_train': mr.estimators_[0].score(U_train, s_train[:, 1]),
                '1_r2_test': mr.estimators_[0].score(U_test, s_test[:, 1]),
                '35_r2_train': mr.estimators_[34].score(U_train, s_train[:, 35]),
                '35_r2_test': mr.estimators_[34].score(U_test, s_test[:, 35]),
                '70_r2_train': mr.estimators_[69].score(U_train, s_train[:, 70]),
                '70_r2_test': mr.estimators_[69].score(U_test, s_test[:, 70])
            }
    
    return mr, report_multidim

def save_model(model, filename):
    joblib.dump(model, filename+'.pkl')

def load_model(dir):
    model = joblib.load(dir)

    return model
    
    

# Plotting functions

def ax_pos_inch_to_absolute(fig_size, ax_pos_inch):
    ax_pos_absolute = []
    ax_pos_absolute.append(ax_pos_inch[0]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[1]/fig_size[1])
    ax_pos_absolute.append(ax_pos_inch[2]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[3]/fig_size[1])
    
    return ax_pos_absolute

def plot_76_tensors(u, solver, amp25=None, amp20=None, period=None, isu=True, text='model_0', file_prefix='1'):
    
    fig_size = (06.90, 02.20+01.50)
    fig = plt.figure(figsize=fig_size)

    ax = []

    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [00.75, 01.25, 06.00, 02.00])))

    cmin = -u.abs().max()
    cmax = u.abs().max()
    
    xmin = 84.
    xmax = 96.
    ymin = 17.
    ymax = 35.

    ax[0].set_xlim(left=84.)
    ax[0].set_xlim(right=96.)
    ax[0].set_ylim(bottom=17.)
    ax[0].set_ylim(top=35.)

    h = []

    h.append(ax[0].contourf(solver.time/86400/360, solver.z/1000, u.T,
                21, cmap="RdYlBu_r", vmin=cmin, vmax=cmax))

    
    ax[0].set_ylabel('Km', fontsize=10)

    ax[0].set_xlabel('model year', fontsize=10)

    # Set ticks
    xticks_list = np.arange(xmin, xmax+1, 1)
    ax[0].set_xticks(xticks_list)

    yticks_list = np.arange(ymin, ymax+2, 2)
    ax[0].set_yticks(yticks_list)

    xticklabels_list = list(xticks_list)
    xticklabels_list = [ '%.0f' % elem for elem in xticklabels_list ]
    ax[0].set_xticklabels(xticklabels_list, fontsize=10)

    ax[0].xaxis.set_minor_locator(MultipleLocator(1.))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1.))

    ax[0].tick_params(which='both', left=True, right=True, bottom=True, top=True)
    ax[0].tick_params(which='both', labelbottom=True)

    # if u, the display \tau and \sigma
    if isu:
        ax[0].axhline(25., xmin=0, xmax=1, color='white', linestyle='dashed', linewidth=1.)
        ax[0].axhline(20., xmin=0, xmax=1, color='white', linestyle='dashed', linewidth=1.)

        ax[0].text(95.50, 25, r'$\sigma_{25}$ = ' '%.1f' %amp25 + r'$\mathrm{m s^{-1}}$',
        horizontalalignment='right', verticalalignment='bottom', color='black')

        ax[0].text(95.50, 20, r'$\sigma_{20}$ = ' '%.1f' %amp20 + r'$\mathrm{m s^{-1}}$',
        horizontalalignment='right', verticalalignment='bottom', color='black')

        ax[0].text(84.50, 25, r'$\tau_{25}$ = ' '%.0f' %period + 'months',
        horizontalalignment='left', verticalalignment='bottom', color='black')

    # The label it displays
    # u/s has different dimension
    if isu:
        label = r'$\mathrm{m s^{-1}}$'
    else:
        label = r'$\mathrm{m s^{-2}}$'
    
    # Color bars
    if isu:
        cbar_ax0 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [01.00, 00.50, 05.50, 00.10])) 
        ax[0].figure.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r"), cax=cbar_ax0, format='% 2.0f', 
        boundaries=np.linspace(cmin, cmax, 21), orientation='horizontal',
        label=label)
    else:
        cbar_ax0 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [01.00, 00.50, 05.50, 00.10])) 
        ax[0].figure.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r"), cax=cbar_ax0, format='% .2e', 
        boundaries=np.linspace(cmin, cmax, 11), orientation='horizontal',
        label=label)
    
    plt.title(text)
    plt.legend()

    if file_prefix:
        plt.savefig(file_prefix+'_zonal_wind.png')

    plt.show()

    

def rMSE(s_pred, s_gt, s_std=None):
    error = (s_gt - s_pred)
    SSE = sum(error ** 2)
    MSE = SSE/s_gt.shape[0]
    RMSE = MSE**.5
    if s_std:
        return RMSE/s_std
    else:
        return s_std
    
def plot_rmse(*args):
    for i in range(len(args)):
        plt.plot(range(len(args[i])), args[i], label='model' + str(i))
    
    plt.xlabel('z level(indices)')
    plt.ylabel('rMSE')
    plt.title('Plot of rMSEs')
    plt.legend()
    plt.show()


def para_for_plotting(solver, u):
    # calculate amplitude and period
    spinup_time = 12*360*86400

    amp25 = utils.estimate_amplitude(solver.time, solver.z, u, height=25e3, spinup=spinup_time)
    amp20 = utils.estimate_amplitude(solver.time, solver.z, u, height=20e3, spinup=spinup_time)
    tau25 = utils.estimate_period(solver.time, solver.z, u, height=25e3, spinup=spinup_time)

    return {'amp25': float(amp25), 'amp20': float(amp20), 'period': float(tau25)}

def qbo_objective(amp25=None, amp20=None, period=None):
    return (amp25-33.)**2 / (33.)**2 + (amp20-19.)**2 / (19.)**2 + (period-28.)**2 / (28.)**2

def plot_wind_level(u, level=35, text='MODEL_0', file_prefix='1'):
    plt.plot(u[:, level])
    plt.xlabel('timestep')
    plt.ylabel('zonal wind at z=' + str(level))
    plt.title(text)

    if file_prefix:
        plt.savefig(file_prefix+'_zonal_wind_'+str(level)+'.png')

    plt.show()
    
def online_testing(initial_condition, model, sc_U, sc_S, K, t_max=360*96*86400, w=3e-4, state=0):
    # use the last slice of the control run
    
    torch.set_default_dtype(torch.float64)
    solver_ML = adsolver.ADSolver(t_max=t_max, w=w, initial_condition=initial_condition)

    sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *para_mat[state])

    # Set up the ML model to pass in the PDE
    def model_ML_aug(x):
        x = torch.hstack([x[1:-1], sf_ML[solver_ML.current_step], cw_ML[solver_ML.current_step]])
        x = x.reshape(1, 73)
        x = sc_U.transform(x)
        x[:, -2:] = K * x[:, -2:]
        y = model.predict(x)
        y = sc_S.inverse_transform(y)
    
        return torch.tensor(y[0])

    start_time = time.time()


    u_ML = solver_ML.solve(source_func=model_ML_aug)
    u_ML = u_ML.detach()

    end_time = time.time()

    duration = end_time - start_time
    
    return solver_ML, u_ML, duration

##################################################################################################
# The following version of online testing is the one with zero padding
# Namely, the MultiOutputRegressor's output is 71 dimensional
##################################################################################################
# def online_testing(initial_condition, model, sc_U, sc_S, K, t_max=360*96*86400, w=3e-4, state=0):
#     # use the last slice of the control run
    
#     torch.set_default_dtype(torch.float64)
#     solver_ML = adsolver.ADSolver(t_max=t_max, w=w, initial_condition=initial_condition)

#     sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *para_mat[state])

#     # Set up the ML model to pass in the PDE
#     def model_ML_aug(x):
#         x = torch.hstack([x[1:-1], sf_ML[solver_ML.current_step], cw_ML[solver_ML.current_step]])
#         x = x.reshape(1, -1)
#         x = sc_U.transform(x)
#         x[:, -2:] = K * x[:, -2:]
#         y = model.predict(x)
#         zero_padding = np.array([0.]).reshape(1, -1)
#         y = np.hstack([zero_padding, y, zero_padding])
#         y = sc_S.inverse_transform(y)

#         return torch.from_numpy(y[0])

#     start_time = time.time()

#     with joblib.parallel_backend(backend='threading', n_jobs=16):
#         u_ML = solver_ML.solve(source_func=model_ML_aug)
#         u_ML = u_ML.detach()

#     end_time = time.time()

#     duration = end_time - start_time
    
#     return solver_ML, u_ML, duration

def fast_online_testing(initial_condition, model, U_train, sc_U, sc_S, K, t_max=360*96*86400, w=3e-4, state=0):
    # use the last slice of the control run
    
    torch.set_default_dtype(torch.float64)
    solver_ML = adsolver.ADSolver(t_max=t_max, w=w, initial_condition=initial_condition)

    sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *para_mat[state])

    # Set up the ML model to pass in the PDE
    def weights(model, vectors):
        '''
        Generate the weights for predicting
        including the weights for the intercept
        '''
        output = np.zeros((len(model.estimators_), vectors.shape[0]))
        for i in range(len(model.estimators_)):
            output[i, model.estimators_[i].support_] = model.estimators_[i].dual_coef_
        
        return torch.tensor(output.T)    

    def base(x, vectors, model):
        '''
        note that now x is a one dimensional flat array
        '''
        output = torch.ones(vectors.shape[0])
        for i in range(vectors.shape[0]):
            output[i] = torch.dot(x - vectors[i], x - vectors[i])
        gamma = 1.0 / 73
            
        return torch.exp(-gamma*output)

    def intercepts(model):
        out = torch.tensor([float(model.estimators_[i].intercept_) for i in range(len(model.estimators_))])
        
        return out

    def fast_predict(x, weights, intercepts, model, vectors):
        return base(x, vectors, model) @ weights + intercepts

    u_mean, u_scale = torch.from_numpy(sc_U.mean_), torch.from_numpy(sc_U.scale_)
    s_mean, s_scale = torch.from_numpy(sc_S.mean_), torch.from_numpy(sc_S.scale_)
    vectors = torch.from_numpy(U_train)

    w = weights(model, vectors)
    b = intercepts(model)


    def model_ML_aug(x):
        # def model_ML_aug_0(x):
        #     x = torch.hstack([x[1:-1], sf_ML[solver_ML.current_step], cw_ML[solver_ML.current_step]])
        #     x = x.reshape(1, 73)
        #     x = sc_U.transform(x)
        #     x[:, -2:] = K * x[:, -2:]
        #     y = model.predict(x)
        #     y = sc_S.inverse_transform(y)
    
        #     return torch.tensor(y[0])
        # gt = model_ML_aug_0(x)

        x = torch.hstack([x[1:-1], sf_ML[solver_ML.current_step], cw_ML[solver_ML.current_step]])
        x = (x - u_mean) / u_scale
        x[-2:] = K * x[-2:]

        y = fast_predict(x=x, weights=w, intercepts=b, model=model, vectors=vectors)

        y = y * s_scale + s_mean
        
        # print(y - gt)

        return y

    start_time = time.time()

    with joblib.parallel_backend(backend='threading', n_jobs=16):
        u_ML = solver_ML.solve(source_func=model_ML_aug)
        u_ML = u_ML.detach()

    end_time = time.time()

    duration = end_time - start_time
    

    
    return solver_ML, u_ML, duration
