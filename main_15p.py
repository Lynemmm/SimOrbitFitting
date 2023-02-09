#%%
from __future__ import absolute_import, unicode_literals, print_function
from cmath import inf
import numpy as np
from numpy import pi, cos
from pymultinest.solve import solve
import os
try: os.mkdir('chains')
except OSError: pass
import corner 

import copy
import time
from scipy import constants
import matplotlib.pyplot as plt
import astropy.constants as cons
import astropy.units as u
from PyAstronomy.pyTiming import pyPeriod
#import pandas as pd
#import pickle
#from matplotlib.colors import LogNorm

# from constant import *
from inputsimobs import *
from inputbayes import *

import DataGenFunc
from math import radians
i_tofit = 0
#%%


print(time.ctime())

# ref star

# number of dimensions our problem has
n_params = 15
# name of the output files
prefix = "chains/2-"

M0_cal = DataGenFunc.Cal_M0(T0 ,begin_year, begin_month, begin_day, begin_time, P_day)
M02_cal = DataGenFunc.Cal_M0(T02 ,begin_year, begin_month, begin_day, begin_time, P_day2)
M03_cal = DataGenFunc.Cal_M0(T03 ,begin_year, begin_month, begin_day, begin_time, P_day3)


time_con = np.genfromtxt('t_all.dat')

#%%
N_time = len(time_con)

#%%
##########################
## parallel tempering
i_dxdy_name =  "%s%s%s" % ("delta_dx_dy_", int(i_tofit), ".dat")
# genformtxt
delta_dx_dy_onerefstar = np.genfromtxt(i_dxdy_name)
delta_dx_dy_sig = np.genfromtxt("delta_dx_dy_sig.dat")

p_mass_max1 = p_mass_max    #6.1   #10
p_mass_min1 = p_mass_min    #4.1   #0.1

p_mass_max2 = p_mass_max    #3.53  #7.53
p_mass_min2 = p_mass_min    #1.53  #0.001

period_max1 = 1200.
period_min1 = 500.

period_max2 = 2700.
period_min2 = 1500.

def myprior(cube):
    cube[0] = cube[0] * 2. - 1.
    cube[1] = cube[1] 
    cube[2] = cube[2] * radians(180.)
    cube[3] = cube[3] * radians(360.)+radians(180.)
    cube[4] = cube[4] * radians(360.)
    cube[5] = cube[5] * (p_mass_max1 - p_mass_min1) + p_mass_min1
    cube[6] = cube[6] * var_uk_err_max
    cube[7] = cube[7] * (period_max1 - period_min1) + period_min1
    cube[8] = cube[8] * 2. - 1.
    cube[9] = cube[9] 
    cube[10] = cube[10] * radians(180.)
    cube[11] = cube[11] * radians(360.)+radians(180.)
    cube[12] = cube[12] * radians(360.)
    cube[13] = cube[13] * (p_mass_max2 - p_mass_min2) + p_mass_min2
    cube[14] = cube[14] * (period_max2 - period_min2) + period_min2

    return cube 
    # pass

def sum_likelihood(delta_dx_dy_sig, var_uke_init, delta_dx_dy_obs, delta_dx_dy_mc):
    # Gregory 2005
    import numpy as np
    N = len(delta_dx_dy_sig)
    llkl_ra = 0
    for i in range(N):
        sig_power = (delta_dx_dy_sig[i,0]**2.0+var_uke_init**2.0)
        AC_one = np.log((2*np.pi)**(-0.5)) + np.log(sig_power**(-0.5))
        exp_one = -(delta_dx_dy_mc[i,0]-delta_dx_dy_obs[i,0])**2.0/2/sig_power
        llkl_ra = llkl_ra+AC_one+exp_one
    llkl_dec = 0
    for i in range(N):
        sig_power = (delta_dx_dy_sig[i,1]**2.0+var_uke_init**2.0)
        AC_one = np.log((2*np.pi)**(-0.5)) + np.log(sig_power**(-0.5))
        exp_one = -(delta_dx_dy_mc[i,1]-delta_dx_dy_obs[i,1])**2.0/2/sig_power
        llkl_dec = llkl_dec+AC_one+exp_one
    llkl = llkl_ra + llkl_dec
    return llkl

def gen_dx_dy_mc_2p(x, time_con):
    from inputsimobs import ms_Msun,d_pc
    from DataGenFunc import GenModelPos_oneplanet, GenRelativePos
    import numpy as np
    as_cal_con1 = GenModelPos_oneplanet(x[1], x[0], x[2], x[3], x[4], x[7], x[5], ms_Msun, d_pc, RA0, DEC0, time_con)

    as_cal_con2 = GenModelPos_oneplanet(x[9], x[8], x[10], x[11], x[12], x[14], x[13], ms_Msun, d_pc, 0., 0., time_con)

    as_cal_con = as_cal_con1 + as_cal_con2

    delta_dx_dy2 = GenRelativePos(as_cal_con, 0)
    delta_dx_dy = np.transpose(delta_dx_dy2)
    return delta_dx_dy 	

def myloglike(cube):
    modelNow = gen_dx_dy_mc_2p(cube,time_con)
    # chi = np.sum(-0.5*((modelNow - zdata)** 2./ yunc ** 2.))
	# return chi
    chi = sum_likelihood(delta_dx_dy_sig, cube[6], delta_dx_dy_onerefstar, modelNow)
    return chi


#%%
# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, n_dims=n_params, 
    evidence_tolerance = 0.01,
    # sampling_efficiency = 0.3,
    n_live_points=1000,
    outputfiles_basename=prefix, 
    verbose=True, resume = True)

np.savetxt(f"{prefix}result_15p.dat", result['samples'])


#########  corner plot #############
#plot.plot_chains_list(chains_list, n_PT, time_dt)
font = {'family' : 'serif', #monospace
'weight' : 'bold', #bold
'size'   : 30,
}

#%%

results = np.genfromtxt(f'{prefix}result_15p.dat')

m = results.shape[0]
n = results.shape[1]
print(m,n)

i_ch_mergeomega_all = np.zeros((m,n), dtype=float)

#        cos_incl = [0]
i_ch_mergeomega_all[:,0] = results[:,0]

#        ecc = [1]
i_ch_mergeomega_all[:,1] = results[:,1]

#        Omega :[2]
i_ch_mergeomega_all[:,2] = results[:,2] 

#        omega : [3]
i_ch_mergeomega_all[:,3] = results[:,3] -radians(180.)

#        M0 = [4]
i_ch_mergeomega_all[:,4] = results[:,4]

#        mp = [5]
i_ch_mergeomega_all[:,5] = results[:,5]

#        period = [6]
i_ch_mergeomega_all[:,6] = results[:,7]

#        var_uke = [7]
# i_ch_mergeomega_all[:,7] = i_ch[:,6]

#        cos_incl = [7]
i_ch_mergeomega_all[:,7] = results[:,8]

#        ecc = [8]
i_ch_mergeomega_all[:,8] = results[:,9]

#        Omega : 9
i_ch_mergeomega_all[:,9] =  results[:,10]

#        omega : 9
i_ch_mergeomega_all[:,10] =  results[:,11] -radians(180.)

#        M0 = [10]
i_ch_mergeomega_all[:,11] = results[:,12]

#        mp = [11]
i_ch_mergeomega_all[:,12] = results[:,13]

#        period = [12]
i_ch_mergeomega_all[:,13] = results[:,14]

#        var_uke = [13]
i_ch_mergeomega_all[:,14] = results[:,6]

p_fit2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
q_50 = 0.0
for i in range(len(p_fit2)):
    x = i_ch_mergeomega_all2[:,i]
    q_16, q_50, q_84 = corner.quantile(x, [0.16, 0.5, 0.84], weights=None)
    p_fit2[i] = q_50

as_cal_con_fit = DataGenFunc.GenModelPos_oneplanet(p_fit2[1], p_fit2[0], p_fit2[2], p_fit2[3]+radians(180.), p_fit2[4], p_fit2[6], p_fit2[5], ms_Msun, d_pc, RA0, DEC0, time_con)
as_cal_con_fit2 = DataGenFunc.GenModelPos_oneplanet(p_fit2[8], p_fit2[7], p_fit2[9], p_fit2[10]+radians(180.), p_fit2[11], p_fit2[13], p_fit2[12], ms_Msun, d_pc, 0., 0., time_con)
as_mu_fit_p = np.transpose(as_cal_con_fit)
as_mu_fit_p2 = np.transpose(as_cal_con_fit2)
as_mu_fit = np.zeros((N_time-1,2),dtype=np.float64)
as_mu_fit[:,0] = as_mu_fit_p[1:N_time,0] - as_mu_fit_p[0,0] + as_mu_fit_p2[1:N_time,0] - as_mu_fit_p2[0,0]
as_mu_fit[:,1] = as_mu_fit_p[1:N_time,1] - as_mu_fit_p[0,1] + as_mu_fit_p2[1:N_time,1] - as_mu_fit_p2[0,1]
np.savetxt(f"{prefix}as_mu_fit.dat", np.transpose([as_mu_fit[:,0],as_mu_fit[:,1]]))
delta_dx_dy_onerefstar_res = np.zeros((N_time-1, 2), dtype= np.float64)
delta_dx_dy_onerefstar_res[:,0] = delta_dx_dy_onerefstar[:,0] - as_mu_fit[:,0] 
delta_dx_dy_onerefstar_res[:,1] = delta_dx_dy_onerefstar[:,1] - as_mu_fit[:,1] 

#%%
# 1
clp_nra_1 = pyPeriod.Gls((time_con[1:N_time], delta_dx_dy_onerefstar_res[:,0]))
clp_nra_1.info()
clp_nra_list_1 = (clp_nra_1.power).tolist()
max_ra_index_1 = clp_nra_list_1.index(max(clp_nra_list_1))
clp_ndec_1 = pyPeriod.Gls((time_con[1:N_time], delta_dx_dy_onerefstar_res[:,1]))
clp_ndec_1.info()
clp_ndec_list_1 = (clp_ndec_1.power).tolist()
max_dec_index_1 = clp_ndec_list_1.index(max(clp_ndec_list_1))

T_ra_1 = 1./clp_nra_1.freq[max_ra_index_1]
T_dec_1 = 1./clp_ndec_1.freq[max_dec_index_1]
T_clp_1 = (T_ra_1 + T_dec_1)/2.

clp_fig_1 = plt.figure()
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.plot(clp_nra_1.freq, clp_nra_1.power, 'b.-')
plt.plot(clp_ndec_1.freq, clp_ndec_1.power, 'r*-')
# plt.text(clp_nra.freq)
plt.text(clp_nra_1.freq[max_ra_index_1], clp_nra_1.power[max_ra_index_1] - 0.1, 'T_ra = ' + str(T_ra_1))
plt.text(clp_ndec_1.freq[max_dec_index_1], clp_ndec_1.power[max_dec_index_1] - 0.1, 'T_dec = ' + str(T_dec_1))
clp_fig_1.savefig("LombScargle-2-15p.png", dpi=300)


# mc_inc, mc_e, mc_Omega+mc_omega, mc_M0, mc_P, mc_M_p, mc_sigma_A
n_truePara_all = [cos_i_orbit2, e_orbit2, ascend_node_Omega2, periapsis_omega2_in, M02_cal, mp_Mearth2, P_day2, cos_i_orbit3, e_orbit3, ascend_node_Omega3, periapsis_omega3_in, M03_cal, mp_Mearth3, P_day3, 0.]

figure1 = corner.corner(i_ch_mergeomega_all,
        labels = [
    r"$cos(i)$",
    r"$e$",
    r'$\Omega$ $[rad]$',#
    r'$\omega$ $[rad]$',
    r'$M_{0}$ $[rad]$', #
    r'$M_\mathrm{p}$ $[\mathrm{M}_{\oplus}]$',
    r"$P$ $[\mathrm{days}]$",
    r"$cos(i2)$",
    r"$e2$",
    r'$\Omega 2$ $[rad]$',#
    r'$\omega 2$ $[rad]$',
    r'$M_{0}2$ $[rad]$', #
    r'$M_\mathrm{p}2$ $[\mathrm{M}_{\oplus}]$',
    r"$P2$ $[\mathrm{days}]$",
    r'$\sigma_{A}$ $[\mu \mathrm{as}]$',
            ],
        label_kwargs={"fontsize": 18},
        truths=n_truePara_all,
        quantiles=[0.16, 0.5, 0.84],
        levels=1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
        show_titles=True,
        title_kwargs={"fontsize": 13}, #
        smooth=True,
        color='#0066CC',#'#3399FF',
        truth_color='r',
        ) 

figure1.savefig(f"Omege_omege_corner_chain{str(i_tofit)}_7_2_15p.png", dpi=300)
plt.savefig(f'Omege_omege_corner_chain{str(i_tofit)}_7_2_15p.pdf')

print(time.ctime())

print("end")

# %%
