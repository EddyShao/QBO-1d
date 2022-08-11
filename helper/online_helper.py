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

sfe = [3.7e-3, 3.6e-3, 3.2e-3, 3.8e-3, 3.8e-3]
sfv = [1e-8, 9e-8, 9e-8, 9e-10, 9e-8]
cwe = [32, 32, 40, 32, 32]
cwv = [256, 256, 256, 256, 256]
# cwv = [225, 225, 225, 225, 225]
corr = [0.75, 0.75, 0.75, 0.75, -0.75]
seed = [int(21*9+8), int(21*6+7+1), int(21*6+15), int(21*12+5), int(21*2+10)]

para_mat = np.array([sfe, sfv, cwe, cwv, corr, seed]).T



def online_testing(initial_condition, model, sc_U, sc_S, K, t_max=360*96*86400, w=3e-4, state=0):
    
    """function that performs online testing
    
    Args:
        initial_condition(function): Usuallly it is a constant function 
                                     that returns the last slice of the control run
        model(sklearn object): Trained machine learning model
        sc_U(sklearn object): scalar transformer when generating the training data for U
        sc_S(sklearn object): scalar transformer when generating the training data for s
        K(float): scaling term for sf and cw
        t_max(int): emulation time
        w(float): constant in the Adsolver 
        state: can be an integer, a list, and a dictionary
        
    Returns:
        solver, u, duration(testing time consumed)
    """    
    # use the last slice of the control run
    
    torch.set_default_dtype(torch.float64)
    solver_ML = adsolver.ADSolver(t_max=t_max, w=w, initial_condition=initial_condition)
    
    if isinstance(state, int):   
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *para_mat[state])
    
    elif isinstance(state, dict):
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], **state)
        
    elif isinstance(state, list):
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *state)
        
        

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


def fast_online_testing(initial_condition, model, U_train, sc_U, sc_S, K, gamma, t_max=360*96*86400, w=3e-4, state=0):
    
    """function that performs online testing.
       This method is faster when the supported vectors are dense
    
    Args:
        initial_condition(function): Usuallly it is a constant function 
                                     that returns the last slice of the control run
        model(sklearn object): Trained machine learning model
        sc_U(sklearn object): scalar transformer when generating the training data for U
        sc_S(sklearn object): scalar transformer when generating the training data for s
        K(float): scaling term for sf and cw
        t_max(int): emulation time
        w(float): constant in the Adsolver 
        
    Returns:
        solver, u, duration(testing time consumed)
    """    
    
    # use the last slice of the control run
    
    torch.set_default_dtype(torch.float64)
    if initial_condition:
        solver_ML = adsolver.ADSolver(t_max=t_max, w=w, initial_condition=initial_condition)
    else:
        solver_ML = adsolver.ADSolver(t_max=t_max, w=w)


    if isinstance(state, int):   
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *para_mat[state])
    
    elif isinstance(state, dict):
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], **state)
        
    elif isinstance(state, list):
        sf_ML, cw_ML = sample_sf_cw(solver_ML.time.shape[0], *state)

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

    def base(x, vectors, model, gamma):
        '''
        note that now x is a one dimensional flat array
        '''
        output = torch.ones(vectors.shape[0])
        for i in range(vectors.shape[0]):
            output[i] = torch.dot(x - vectors[i], x - vectors[i])
            
        return torch.exp(-gamma*output)

    def intercepts(model):
        out = torch.tensor([float(model.estimators_[i].intercept_) for i in range(len(model.estimators_))])
        
        return out

    def fast_predict(x, weights, intercepts, model, vectors, gamma):
        return base(x, vectors, model, gamma) @ weights + intercepts

    u_mean, u_scale = torch.from_numpy(sc_U.mean_), torch.from_numpy(sc_U.scale_)
    s_mean, s_scale = torch.from_numpy(sc_S.mean_), torch.from_numpy(sc_S.scale_)
    vectors = torch.from_numpy(U_train)

    weight = weights(model, vectors)
    b = intercepts(model)


    def model_ML_aug(x):

        x = torch.hstack([x[1:-1], sf_ML[solver_ML.current_step], cw_ML[solver_ML.current_step]])
        x = (x - u_mean) / u_scale
        x[-2:] = K * x[-2:]

        y = fast_predict(x=x, weights=weight, intercepts=b, model=model, vectors=vectors, gamma=gamma)

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


class IntegratedSVR():
    def __init__(self, model, sc_U, sc_S, U_train, K, gamma):
        self.model = model
        self.sc_U = sc_U
        self.sc_S = sc_S
        self.U_train = U_train
        self.K = K
        self.gamma = gamma
    

    def offline_predict(self, x):
        # predict 2D array
        x = torch.hstack((x[:, 1:-3], x[:, -2:]))
        x = self.sc_U.transform(x)
        x[:, -2:] = self.K * x[:, -2:]
        y = self.model.predict(x)
        y = self.sc_S.inverse_transform(y)

        return torch.from_numpy(y)

    def online_predict(self, x):
        x = torch.hstack((x[1:-3], x[-2:]))
        x = x.reshape(1, 73)
        x = self.sc_U.transform(x)
        x[:, -2:] = self.K * x[:, -2:]
        y = self.model.predict(x)
        y = self.sc_S.inverse_transform(y)
    
        return torch.from_numpy(y[0])

    def fast_online_predict(self, x):

        def weights(model, vectors):
            '''
            Generate the weights for predicting
            including the weights for the intercept
            '''
            output = np.zeros((len(model.estimators_), vectors.shape[0]))
            for i in range(len(model.estimators_)):
                output[i, model.estimators_[i].support_] = model.estimators_[i].dual_coef_
            
            return torch.tensor(output.T)    

        def base(x, vectors, model, gamma):
            '''
            note that now x is a one dimensional flat array
            '''
            output = torch.ones(vectors.shape[0])
            for i in range(vectors.shape[0]):
                output[i] = torch.dot(x - vectors[i], x - vectors[i])
                
            return torch.exp(-gamma*output)

        def intercepts(model):
            out = torch.tensor([float(model.estimators_[i].intercept_) for i in range(len(model.estimators_))])
            
            return out

        def fast_predict(x, weights, intercepts, model, vectors, gamma):
            return base(x, vectors, model, gamma) @ weights + intercepts
        

        u_mean, u_scale = torch.from_numpy(self.sc_U.mean_), torch.from_numpy(self.sc_U.scale_)
        s_mean, s_scale = torch.from_numpy(self.sc_S.mean_), torch.from_numpy(self.sc_S.scale_)
        vectors = torch.from_numpy(self.U_train)

        weight = weights(self.model, vectors)
        b = intercepts(self.model)
        
        x = torch.hstack((x[1:-3], x[-2:]))
        x = (x - u_mean) / u_scale
        x[-2:] = self.K * x[-2:]

        y = fast_predict(x=x, weights=weight, intercepts=b, model=self.model, vectors=vectors, gamma=self.gamma)

        y = y * s_scale + s_mean

        return y


            



