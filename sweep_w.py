# import dask
# from dask.distributed import Client
import netCDF4 as nc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import os.path
from qbo1d import adsolver
from qbo1d import utils
from qbo1d import emulate as emulate
from qbo1d import stochastic_forcing

from helper.online_helper import IntegratedSVR
from helper.data_helper import load_parameters
from helper.training_helper import load_model
from helper.loss_helper import para_for_plotting
from helper.plotting_helper import plot_76_tensors

import argparse
import joblib



# global parameters
t_max = 360 * 108 * 86400
nsteps = 360 * 108
nspinup = 360 * 12
ntot = int(nsteps - nspinup)

torch.set_default_dtype(torch.float64)

sfv = 9e-8
cwv = 256
corr = 0.75

solver = adsolver.ADSolver(t_max=t_max, w=3e-4)
time = solver.time
z = solver.z
nlev = z.shape[0]

# sweep parameters
sfe = torch.arange(3., 5.05, 0.1) * 1e-3  # surface flux Pa
cwe = torch.arange(25., 46., 1.)  # half width m s^{-1}

# seeds for reproducibility -- different seed for each case
seed = torch.arange(1, 442, dtype=int).reshape((21, 21))

# change accordingly
# MODEL_NAME = 'your_model_name_here'

# a wrapper around a pair of solver and model producing u, s
# the function also return sf, cw, to be stored in the nc file

# dir = '/scratch/zs1542/QBO-1d/experiments_grid_search/'
# model_id = '828'

def integrated_model(dir, model_id):
    model_dir = dir + 'model_' + model_id + '/'
    model = load_model(model_dir + 'model_' + model_id +'.pkl')
    sc_U = load_model(model_dir + 'sc_U_' + model_id +'.pkl')
    sc_S = load_model(model_dir + 'sc_S_' + model_id +'.pkl')
    U_train = np.load(model_dir + 'U_train_' + model_id + '.npy')
    w = np.load(model_dir + 'w_' + model_id + '.npy')
    para = load_parameters(dir, model_id)
    K = para['K']
    gamma = para['gamma']
    test_size = para.get('test_size', 0.8)
    

    return IntegratedSVR(model, sc_U, sc_S, U_train, K, gamma), test_size, w

def integrate(sfe, cwe, seed, dir, model_id):
    model, test_size, w = integrated_model(dir, model_id)

    solver = adsolver.ADSolver(t_max=t_max, w=w)

    # emulator

    if test_size < 0.95:
        print('Using the usual way')
        u = solver.emulate(source_func=model.online_predict, sfe=sfe, sfv=sfv,
        cwe=cwe, cwv=cwv, seed=seed, corr=corr)
    else:
        print('Using the matrix multiplication')
        u = solver.emulate(source_func=model.fast_online_predict, sfe=sfe, sfv=sfv,
        cwe=cwe, cwv=cwv, seed=seed, corr=corr)

    sf, cw = stochastic_forcing.sample_sf_cw(n=nsteps+1,
    sfe=sfe, sfv=sfv, cwe=cwe, cwv=cwv, corr=corr, seed=seed)

    # s = model.s
    s = torch.zeros_like(u)
    s[:, :] = model.offline_predict(torch.hstack((u[:, :], sf.view(-1, 1), cw.view(-1, 1))))


    return u, s, sf, cw, solver

def main(dir, model_id, state):
    sfe_id = state // 21
    cwe_id = state % 21

    print('start emulating')
    with joblib.parallel_backend(backend='threading', n_jobs=8):
        u, s, sf, cw, solver = integrate(sfe[sfe_id], cwe[cwe_id], seed[sfe_id, cwe_id], dir, model_id)


    # qbo_paras = para_for_plotting(solver, u)

    # text = f'sfe = {sfe_id}, cwe = {cwe_id} | model_id = {model_id} '

    # plot_76_tensors(u, solver, isu=True, text=text, file_prefix=str(state)+'_perturbed_'+model_id, **qbo_paras)

    file_name = '/scratch/zs1542/QBO-1d/sweep_w_nc/' + 'model_' + model_id + '/' + str(state) + '.nc'
    ds = nc.Dataset(file_name, 'w', format='NETCDF4')

    heights = ds.createDimension('z', z.shape[0])
    times = ds.createDimension('time', ntot)

    heights = ds.createVariable('z', 'f8', ('z'))
    times = ds.createVariable('time', 'f8', ('time'))

    wind = ds.createVariable('u', 'f8', ('time', 'z'))
    source = ds.createVariable('S', 'f8', ('time', 'z'))
    sf_samp = ds.createVariable('sf_samp', 'f8', ('time'))
    cw_samp = ds.createVariable('cw_samp', 'f8', ('time'))

    heights[:] = z
    times[:] = time[nspinup: nsteps]

    wind[:, :] = u[nspinup:nsteps, :]
    source[:, :] = s[nspinup:nsteps, :]
    sf_samp[:] = sf[nspinup:nsteps]
    cw_samp[:] = cw[nspinup:nsteps]

    ds.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--state', help='Specify state the id', type=int)
    parser.add_argument('-m', '--model', help='Specify model id')
    parser.add_argument('-d', '--directory', help='base-directory')

    args = parser.parse_args()
    
    state_id = args.state
    model_id = args.model
    dir = args.directory

    dir = dir.strip()
    if dir[-1] != '/':
        dir = dir + '/'

    main(dir, model_id, state_id)
    
    

