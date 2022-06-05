from termios import CWERASE
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import os.path
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from qbo1d import utils
from qbo1d import adsolver
from qbo1d import emulate
from qbo1d.stochastic_forcing import WaveSpectrum

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from qbo1d.stochastic_forcing import sample_sf_cw


#=========================================#
# Specify the state

#=========================================#

def data_generator(state):
    '''
    STATE = 0 -> old control
    STATE = 1 -> new control
    STATE = 2 -> different mean
    STATE = 3 -> different variance
    STATE = 4 -> anti-correlation(non-physical) 
    '''

    # parameter dicts
    sfe = [3.7e-3, 3.8e-3, 3.2e-3, 3.8e-3, 3.8e-3]
    sfv = [1e-8, 9e-8, 9e-8, 9e-10, 9e-8]
    cwe = [32, 32, 40, 32, 32]
    cwv = [225, 225, 225, 225, 225]
    corr = [0.75, 0.75, 0.75, 0.75, -0.75]
    seed = [int(21*9+8), int(21*9+7), int(21*6+15), int(21*12+5), int(21*2+10)]

    # generate the matrix form

    para_mat = np.array([sfe, sfv, cwe, cwv, corr, seed]).T


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
    model = WaveSpectrum(solver, *para_mat[2])
    time = solver.time
    z = solver.z
    u = solver.solve(source_func=model)

    s = model.s

    cw, sf = model.cw, model.sf

    return 


