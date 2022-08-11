import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from matplotlib import patches

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


# Plotting functions

def ax_pos_inch_to_absolute(fig_size, ax_pos_inch):

    ax_pos_absolute = []
    ax_pos_absolute.append(ax_pos_inch[0]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[1]/fig_size[1])
    ax_pos_absolute.append(ax_pos_inch[2]/fig_size[0])
    ax_pos_absolute.append(ax_pos_inch[3]/fig_size[1])
    
    return ax_pos_absolute

def plot_76_tensors(u, solver, amp25=None, amp20=None, period=None, isu=True, text='model_0', file_prefix='1'):
    """ Function for visualizing 73 dim tensor(such as zonal wind and source term)
        76 in function name is a typo. Not changing it due to strong dependency.

    Args:
        u (tensor object): Zonal wind(source term) tensor, should have shape
        solver (Adsolver object): necessary for plotting (for passing in the timeline)
        amp25 (float): amplitude at 25km. Defaults to None.
        amp20 (float): amplitude at 20km. Defaults to None.
        period (float): period at 20km. Defaults to None.
        isu (bool): True means the tensor passed in is zonal wind. Defaults to True.
        text (str): title to show on the plot. Defaults to 'model_0'.
        file_prefix (str, optional): filename prefix for saving . Defaults to '1', 'None' means no saving
    """     
     
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
    

    
    
def plot_rmse(*args):
    """Plotting function for comparison of rmses
    Args:
        rmses (tensor object): rmse lists
    
    """    
    
    for i in range(len(args)):
        plt.plot(range(len(args[i])), args[i], label='model' + str(i))
    
    plt.xlabel('z level(indices)')
    plt.ylabel('rMSE')
    plt.title('Plot of rMSEs')
    plt.legend()
    plt.show()



def plot_wind_level(u, level=35, text='MODEL_0', file_prefix='1'):
    """Function for plotting zonal wind at certain level

    Args:
        u (torch.tensor): Zonal wind at all levels
        level (int, optional): Level to visualize. Defaults to 35.
        text (str, optional): text showing on the figure. Defaults to 'MODEL_0'.
        file_prefix (str, optional): prefix of file name when saving. Defaults to '1'.
    """    
    plt.plot(u[:, level])
    plt.xlabel('timestep')
    plt.ylabel('zonal wind at z=' + str(level))
    plt.title(text)

    if file_prefix:
        plt.savefig(file_prefix+'_zonal_wind_'+str(level)+'.png')

    plt.show()
    
def plot_sweep_sfe_cwe(sf, cw, amp25, amp20, tau25, objective_function, sfv, cwv, save_path=None):
    fig_size = (06.50, 05.50)  #02.20+01.50)
    fig = plt.figure(figsize=fig_size)

    ax = []

    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [00.75, 03.25, 02.00, 02.00])))  #ax0
    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [04.00, 03.25, 02.00, 02.00])))  #ax1
    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [00.75, 00.50, 02.00, 02.00])))  #ax2
    ax.append(fig.add_axes(ax_pos_inch_to_absolute(fig_size, [04.00, 00.50, 02.00, 02.00])))  #ax3


    sf0 = 3.7
    cw0 = 32


    secax = []

    panel_name = ["(a)", "(b)", "(c)", "(d)",
                "(e)", "(f)", "(g)", "(h)",
                "(i)", "(j)", "(k)", "(l)",
                "(m)", "(n)", "(o)", "(p)",
                "(q)", "(r)", "(s)", "(t)"]


    h = []


    ##### ax0
    clevels = np.arange(0., 65., 5.) 

    h.append(ax[0].contourf(cw, 1.e3*sf, amp25, levels=10, cmap='Blues'))
    cl0 = ax[0].contour(cw, 1.e3*sf, amp25, levels=10, colors='Black', linewidths=0.5)
    ax[0].clabel(cl0, inline=1, fontsize=8)
    # h.append(ax[0].pcolor(cw, 1.e3*sf, amp25, cmap='Blues', edgecolor='black', linewidth=0.5))


    ax[0].add_patch(patches.Ellipse((cw0, sf0), width=cwv, height=1.e3*sfv, linewidth=1, edgecolor='white', fill=False))

    ax[0].plot(cw0, sf0, color='black', marker='.')

    # ax[0].plot(32., 3.7, color='black', marker='.')
    # ax[0].plot(40., 3.2, color='black', marker='x')

    ax[0].set_xlim([25, 45])
    ax[0].set_ylim([3, 5])

    cbar_ax0 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [02.85, 03.25, 00.10, 02.00])) 
    cbar0 = mpl.colorbar.ColorbarBase(cbar_ax0, cmap=plt.cm.get_cmap('Blues'), orientation='vertical',
                                    norm=colors.Normalize(vmin=np.min(amp25), vmax=np.max(amp25)),
                                    extend='neither',
                                    boundaries=clevels,
                                    spacing='proportional')

    cbar0.ax.set_title(label=r'$\mathrm{[m^{} s^{-1}]}$', fontsize=10) #, rotation=0, x=-1, y=1.1)

    ax[0].set_title(r'(a) High level amplitude', loc='left', fontsize=10)
    ax[0].set_xlabel(r'$\mathrm{mean} \, (c_{w}) \, [\mathrm{m \, s^{-1}}]$')
    ax[0].set_ylabel(r'$\mathrm{mean} \, (F_{S0}) \, [\mathrm{mPa}]$')




    ##### ax1
    clevels = np.arange(5., 45., 5.) 

    h.append(ax[1].contourf(cw, 1.e3*sf, amp20, levels=10, cmap='Greens'))
    cl1 = ax[1].contour(cw, 1.e3*sf, amp20, levels=10, colors='Black', linewidths=0.5)
    ax[1].clabel(cl1, inline=1, fontsize=8)
    # h.append(ax[1].pcolor(cw, 1.e3*sf, amp20, cmap='Greens', edgecolor='black', linewidth=0.5))

    ax[1].add_patch(patches.Ellipse((cw0, sf0), width=cwv, height=1.e3*sfv, linewidth=1, edgecolor='white', fill=False))

    ax[1].plot(cw0, sf0, color='black', marker='.')

    # ax[1].plot(32., 3.7, color='black', marker='.')
    # ax[1].plot(40., 3.2, color='black', marker='x')

    cbar_ax1 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [06.10, 03.25, 00.10, 02.00])) 
    cbar1 = mpl.colorbar.ColorbarBase(cbar_ax1, cmap=plt.cm.get_cmap('Greens'), orientation='vertical',
                                    norm=colors.Normalize(vmin=np.min(amp20), vmax=np.max(amp20)),
                                    extend='neither',
                                    boundaries=clevels,
                                    spacing='proportional')

    cbar1.ax.set_title(label=r'$\mathrm{[m^{} s^{-1}]}$', fontsize=10) #, rotation=0, x=-1, y=1.1)

    ax[1].set_title(r'(b) Low level amplitude', loc='left', fontsize=10)
    ax[1].set_xlabel(r'$\mathrm{mean} \, (c_{w}) \, [\mathrm{m \, s^{-1}}]$')
    ax[1].set_ylabel(r'$\mathrm{mean} \, (F_{S0}) \, [\mathrm{mPa}]$')


    ax[1].set_xlim([25, 45])
    ax[1].set_ylim([3, 5])



    ##### ax2
    clevels = np.arange(10., 70., 10.)

    h.append(ax[2].contourf(cw, 1.e3*sf, tau25, norm=colors.Normalize(vmin=np.min(tau25), vmax=np.max(tau25)), levels=10, cmap='Purples'))
    cl2 = ax[2].contour(cw, 1.e3*sf, tau25, norm=colors.Normalize(vmin=np.min(tau25), vmax=np.max(tau25)), levels=10, colors='Black', linewidths=0.5)
    ax[2].clabel(cl2, inline=1, fontsize=8)
    # h.append(ax[2].pcolor(cw, 1.e3*sf, tau25, norm=colors.Normalize(vmin=np.min(tau25), vmax=np.max(tau25)), cmap='Purples', edgecolor='black', linewidth=0.5))

    ax[2].add_patch(patches.Ellipse((cw0, sf0), width=cwv, height=1.e3*sfv, linewidth=1, edgecolor='white', fill=False))

    ax[2].plot(cw0, sf0, color='black', marker='.')

    # ax[2].plot(32., 3.7, color='black', marker='.')
    # ax[2].plot(40., 3.2, color='black', marker='x')

    cbar_ax2 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [02.85, 00.50, 00.10, 02.00])) 
    cbar2 = mpl.colorbar.ColorbarBase(cbar_ax2, cmap=plt.cm.get_cmap('Purples'), orientation='vertical',
                                    norm=colors.Normalize(vmin=10, vmax=60),
                                    extend='neither',
                                    boundaries=clevels,
                                    spacing='proportional')


    cbar2.ax.set_title(label=r'$\mathrm{[months]}$', fontsize=10) #, rotation=0, x=-1, y=1.1)
    # cbar2.set_ticklabels(['10', '', '30', '', '50', '', '70', '', '90', ''])
    cbar2.set_ticklabels(['10', '', '30', '', '50', ''])
    

    ax[2].set_title(r'(c) Period', loc='left', fontsize=10)
    ax[2].set_xlabel(r'$\mathrm{mean} \, (c_{w}) \, [\mathrm{m \, s^{-1}}]$')
    ax[2].set_ylabel(r'$\mathrm{mean} \, (F_{S0}) \, [\mathrm{mPa}]$')

    ax[2].set_xlim([25, 45])
    ax[2].set_ylim([3, 5])



    ##### ax3
    clevels = np.arange(-3.5, 1.0, 0.5)

    h.append(ax[3].contourf(cw, 1.e3*sf, np.log10(objective_function), levels=10, norm=colors.Normalize(vmin=np.min(np.log10(objective_function)), vmax=np.max(np.log10(objective_function))), cmap='Reds'))
    cl3 = ax[3].contour(cw, 1.e3*sf, np.log10(objective_function), levels=10, norm=colors.Normalize(vmin=np.min(np.log10(objective_function)), vmax=np.max(np.log10(objective_function))), colors='Black', linewidths=0.5)
    ax[3].clabel(cl3, inline=1, fontsize=8)
    # h.append(ax[3].pcolor(cw, 1.e3*sf, np.log10(objective_function), norm=colors.Normalize(vmin=np.min(np.log10(objective_function)), vmax=np.max(np.log10(objective_function))), cmap='Reds', edgecolor='black', linewidth=0.5))

    ax[3].add_patch(patches.Ellipse((cw0, sf0), width=cwv, height=1.e3*sfv, linewidth=1, edgecolor='white', fill=False))

    ax[3].plot(cw0, sf0, color='black', marker='.')

    # ax[3].plot(32., 3.7, color='black', marker='.')
    # ax[3].plot(40., 3.2, color='black', marker='x')


    cbar_ax3 = fig.add_axes(ax_pos_inch_to_absolute(fig_size, [06.10, 00.50, 00.10, 02.00])) 
    cbar3 = mpl.colorbar.ColorbarBase(cbar_ax3, cmap=plt.cm.get_cmap('Reds'), orientation='vertical',
                                    norm=colors.Normalize(vmin=np.min(np.log10(objective_function)), vmax=np.max(np.log10(objective_function))),
                                    extend='neither',
                                    boundaries=clevels,
                                    spacing='proportional')

    cbar3.set_ticklabels(['', '-3', '', '-2', '', '-1', '', '0', ''])
    # cbar3.set_ticklabels(['', '-2', '', '-1', '', '0', '', '1'])
    ax[3].set_title(r'(d) $\mathrm{Log}_{10}$ (Objective function)', loc='left', fontsize=10)
    ax[3].set_xlabel(r'$\mathrm{mean} \, (c_{w}) \, [\mathrm{m \, s^{-1}}]$')
    ax[3].set_ylabel(r'$\mathrm{mean} \, (F_{S0}) \, [\mathrm{mPa}]$')

    ax[3].set_xlim([25, 45])
    ax[3].set_ylim([3, 5])
    
    if save_path:
        plt.savefig(save_path)