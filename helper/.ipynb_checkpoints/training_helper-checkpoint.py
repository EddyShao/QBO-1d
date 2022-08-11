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

# from dask.distributed import Client

import joblib
import json


def rMSE(s_pred, s_gt, s_std=None):
    
    """Function that generate rMSE list
    
    Args:
        s_pred(array): prediction(all levels)
        s_gt(array): ground truth(all levels)
        
    Returns:
        RMSE(array): rMSE list, normalized if s_std provided
    """    
    
    error = (s_gt - s_pred)
    SSE = sum(error ** 2)
    MSE = SSE/s_gt.shape[0]
    RMSE = MSE**.5
    if s_std:
        return RMSE/s_std
    else:
        return RMSE

def svr_training_1d(U_train, U_test, s_train, s_test, epsilon, C, gamma='auto'):
    """Training function for 1d svr model, mainly for experiment

    Args:
        U_train (array): data for training
        U_test (array): data for testing
        s_train (array): data for training
        s_test (array): data for training
        epsilon (float): hyperparameter, tolerance
        C (float): hyperparameter, coefs for penalty term

    Returns:
        python dict object: report in dict
    """    
    
    report_1d = {}
    for level in [1, 35, 70]:
        s_train_level = s_train[:, level]
        s_test_level = s_test[:, level]
        with joblib.parallel_backend(backend='threading', n_jobs=16):
            svr = SVR(kernel='rbf', gamma=gamma, epsilon=epsilon, C=C)
            svr.fit(U_train, s_train_level)
            report_level = {
                str(level)+'_r2_train': svr.score(U_train, s_train_level),
                str(level)+'_r2_test': svr.score(U_test, s_test_level),
                str(level)+'_n_supported': int(svr.n_support_[0]),
            }
        report_1d.update(report_level)
    
    return report_1d


def svr_training_multidim(U_train, U_test, s_train, s_test, epsilon, C, gamma='auto'):
    """Training function for SVR model, all levels

    Args:
        U_train (array): data for training
        U_test (array): data for testing
        s_train (array): data for training
        s_test (array): data for training
        epsilon (float): hyperparameter, tolerance
        C (float): hyperparameter, coefs for penalty term

    Returns:
        mr(sklearn model object): SVRs wrapped in a MultiOutputRegressor
        python dict object: report in dict
    """    
    
    # Multiregression
    with joblib.parallel_backend(backend='threading', n_jobs=16):
        svr_1d = SVR(kernel = 'rbf', gamma=gamma, epsilon=epsilon, C=C)
        mr = MultiOutputRegressor(svr_1d, n_jobs=8)
        mr.fit(U_train, s_train)
        
        prediction = mr.predict(U_test)
        rmse = rMSE(prediction, s_test, s_test.std()+1e-32)

        n_support_array = np.zeros(len(mr.estimators_))
        for i in range(len(mr.estimators_)):
            n_support_array[i] = mr.estimators_[i].n_support_

        report_multidim = {
                'multidim_r2_train': mr.score(U_train, s_train),
                'multidim_r2_test': mr.score(U_test, s_test),
                'multidim_mean_rmse': rmse.mean(),
                'mean_n_support': float(n_support_array.mean())
            }
    
    return mr, report_multidim



def rvr_training_1d(U_train, U_test, s_train, s_test, level, gamma='auto', threshold_alpha=1e5, alpha_max=1e9):
    """ Training function for relevance vector machine model
        Since training multi-dimensional rvr model is too time-consuming,
        this is the actual function used in practice

    Args:
        U_train (array): data for training
        U_test (array): data for testing
        s_train (array): data for training
        s_test (array): data for training
        level (int): level index
        threshold_alpha(int): , default=1e-5

    Returns:
       model(sklearn model object): RVR model
       python dict object: report in dict
    """    
    
    assert level < 72 and level > 0
    
    s_train_level, s_test_level = s_train[:, level], s_test[:, level]
    
    with joblib.parallel_backend(backend='threading', n_jobs=16):
        rvr = EMRVR(kernel='rbf', gamma=gamma, bias_used=True, threshold_alpha=threshold_alpha, alpha_max=alpha_max)
        rvr.fit(U_train, s_train_level)
        report_level = {
            str(level)+'_r2_train': rvr.score(U_train, s_train_level),
            str(level)+'_r2_test': rvr.score(U_test, s_test_level),
            str(level)+'_n_supported': int(rvr.relevance_.shape[0]),
        }
    
    return rvr, report_level


def rvr_training_multidim(U_train, U_test, s_train, s_test):
    
    """ Training function for relevance vector machine model
        Since training multi-dimensional rvr model is too time-consuming,
        this is the actual function used in practice

    Args:
        U_train (array): data for training
        U_test (array): data for testing
        s_train (array): data for training
        s_test (array): data for training

    Returns:
       model(sklearn model object): RVR model
       python dict object: report in dict
    """    
    
   
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

def save_model(model, filename):
    joblib.dump(model, filename+'.pkl')

def load_model(dir):
    model = joblib.load(dir)
    return model