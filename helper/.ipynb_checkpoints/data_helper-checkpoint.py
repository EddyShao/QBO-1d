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

from qbo1d.complex_forcing import ComplexWaveSpectrum
from qbo1d.complex_forcing import sample_sf_cw_complex

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

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
    """function for loading parameters

    Args:
        dir (str): JSON file storing parameters
        para_id (str): key for retrieving the parameters

    Returns:
        dict: parameter dictionary, can be passed to the function as kwargs
    """    
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
    """Data loader

    Args:
        state (int, optional): Specify the state. Defaults to 1.
        
    STATE = 0 -> old control
    STATE = 1 -> new control
    STATE = 2 -> different mean
    STATE = 3 -> different variance
    STATE = 4 -> anti-correlation(non-physical) 

    Returns:
        u, s, sf, cw, solver
    """    
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
    
    if isinstance(state, int):
        model = WaveSpectrum(solver, *para_mat[state])
    elif isinstance(state, dict):
        model = WaveSpectrum(solver, **state)
    elif isinstance(state, list):
        model = WaveSpectrum(solver, *state)
        
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    return u, model.s, model.sf, model.cw, solver



def complex_data_loader(state=1):
    """Complex Data Loader

    Args:
        state (int, optional): Specify the state. Defaults to 1.
        
    STATE = 0 -> old control
    STATE = 1 -> new control
    STATE = 2 -> different mean
    STATE = 3 -> different variance
    STATE = 4 -> anti-correlation(non-physical) 

    Returns:
        u, s, sf, cw, solver
    """    
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
    model = ComplexWaveSpectrum(solver, *para_mat[state])
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    return u, model.s, model.sf, model.cw, solver


# Data preprocessing

def dataset(u, s, sf, cw, K=1, nsteps=360*108, nspinup=360*12, test_size=0.8):
    
    """function generating the datasets given raw data

    Returns:
        train dataset, testing dataset, scaling transformers for u and s
    """    
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