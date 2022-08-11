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
    

def para_for_plotting(solver, u):
    """Function that generates QBO parameters

    Args:
        solver (Adsolver object): solver object is necessary for computing amplitude and period
        u (torch.tensor): Zonal wind

    Returns:
        (python dict object): QBO parameters packed in a dictionary 
    """    
    # calculate amplitude and period
    spinup_time = 12*360*86400

    amp25 = utils.estimate_amplitude(solver.time, solver.z, u, height=25e3, spinup=spinup_time)
    amp20 = utils.estimate_amplitude(solver.time, solver.z, u, height=20e3, spinup=spinup_time)
    tau25 = utils.estimate_period(solver.time, solver.z, u, height=25e3, spinup=spinup_time)

    return {'amp25': float(amp25), 'amp20': float(amp20), 'period': float(tau25)}

def qbo_objective(amp25=None, amp20=None, period=None):
    """Function for computing qbo objective

    Args:
        amp25 (float): amplitude at 25km. 
        amp20 (float): amplitude at 20km. 
        period (float): period at 20km.

    Returns:
        objective(float): QBO objective
    """    
    objective = (amp25-33.4)**2 / (33.4)**2 + (amp20-18.5)**2 / (18.5)**2 + (period-27.)**2 / (27.)**2
    return objective


def qbo_objective_annual_w(amp25=None, amp20=None, period=None):

    """Function for computing qbo objective with annual cycle

    Args:
        amp25 (float): amplitude at 25km. 
        amp20 (float): amplitude at 20km. 
        period (float): period at 20km.

    Returns:
        objective(float): QBO objective
    """    
    objective = (amp25-33.9)**2 / (33.9)**2 + (amp20-18.3)**2 / (18.3)**2 + (period-27.)**2 / (27.)**2
    return objective
