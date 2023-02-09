#%%

#  corner plot ; LombScargle; GoodnessOfFit

from __future__ import absolute_import, unicode_literals, print_function
from cmath import inf

import numpy as np
from numpy import pi, cos
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

#from matplotlib.colors import LogNorm

from inputsimobs import *

from math import radians

import matplotlib as mpl
import matplotlib.ticker as mpltk
import pylab as pl
i_tofit = 0 #6 #5 #4 #3 #2 #1 #0
#%%
print(time.ctime())

# number of dimensions our problem has
n_params = 22
# name of the output files
prefix = "chains/all-"
prefix1 = "chains/8p-2-"

time_con = np.genfromtxt('t_all.dat')

#%%
N_time = len(time_con)

#%%
##########################
## parallel tempering
i_dxdy_name =  "%s%s%s" % ("delta_dx_dy_", int(i_tofit), ".dat")

delta_dx_dy_onerefstar = np.genfromtxt(i_dxdy_name)
delta_dx_dy_sig = np.genfromtxt("delta_dx_dy_sig.dat")

font = {'family' : 'serif', #monospace
'weight' : 'bold', #bold
'size'   : 30,
}

#%%
results = np.genfromtxt(f'{prefix}result_22p.dat')

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

#        cos_inc = [14]
i_ch_mergeomega_all[:,14] = results[:,15]

#        ecc = [15]
i_ch_mergeomega_all[:,15] = results[:,16]

#        Omega : 16
i_ch_mergeomega_all[:,16] = results[:,17]

#        omega : 17
i_ch_mergeomega_all[:,17] = results[:,18] -radians(180.)

#        M0 = [18]
i_ch_mergeomega_all[:,18] = results[:,19]

#        mp = [19]
i_ch_mergeomega_all[:,19] = results[:,20]

#        period = 20
i_ch_mergeomega_all[:,20] = results[:,21]

#        var_uke = [21]
i_ch_mergeomega_all[:,21] = results[:,6]

# mc_inc, mc_e, mc_Omega+mc_omega, mc_M0, mc_P, mc_M_p, mc_sigma_A
n_truePara_all = [cos_i_orbit, e_orbit, ascend_node_Omega, periapsis_omega_in, M0_cal, mp_Mearth, P_day, cos_i_orbit2, e_orbit2, ascend_node_Omega2, periapsis_omega2_in, M02_cal, mp_Mearth2, P_day2, cos_i_orbit3, e_orbit3, ascend_node_Omega3, periapsis_omega3_in, M03_cal, mp_Mearth3, P_day3, 0.]
p_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
q_50 = 0.0
for i in range(len(p_fit)):
    x = i_ch_mergeomega_all[:,i]
    q_16, q_50, q_84 = corner.quantile(x, [0.16, 0.5, 0.84], weights=None)
    p_fit[i] = q_50

i_save_name_load =  "%s%s%s" % ("as_mu_fit_N_", int(i_tofit), ".dat")
as_mu_fit_N = np.genfromtxt(i_save_name_load)
res_N = np.zeros((N_time-1, 2), dtype= np.float64)
res_N[:,0] = delta_dx_dy_onerefstar[:,0] - as_mu_fit_N[:,0] 
res_N[:,1] = delta_dx_dy_onerefstar[:,1] - as_mu_fit_N[:,1]

times = np.linspace(0, ObservationPeriod, timesteps)
O_time = len(times)
times_s = ((times * u.year).to('s')).value 

#as_cal_con_fit = DataGenFunc.GenModelPos_oneplanet(p_fit[1], p_fit[0], p_fit[2], p_fit[3] + radians(180.), p_fit[4], p_fit[6], p_fit[5], ms_Msun, d_pc, RA0, DEC0, times_s)
#as_cal_con_fit1 = DataGenFunc.GenModelPos_oneplanet(p_fit[8], p_fit[7], p_fit[9], p_fit[10]+radians(180.), p_fit[11], p_fit[13], p_fit[12], ms_Msun, d_pc, 0., 0., times_s)
#as_cal_con_fit2 = DataGenFunc.GenModelPos_oneplanet(p_fit[15], p_fit[14], p_fit[16], p_fit[17]+radians(180.), p_fit[18], p_fit[20], p_fit[19], ms_Msun, d_pc, 0., 0., times_s)
#as_mu_fit_p = np.transpose(as_cal_con_fit + as_cal_con_fit1 + as_cal_con_fit2)
#as_mu_fit = np.zeros((O_time-1,2),dtype=np.float64)
#as_mu_fit[:,0] = as_mu_fit_p[1:O_time,0] - as_mu_fit_p[0,0]
#as_mu_fit[:,1] = as_mu_fit_p[1:O_time,1] - as_mu_fit_p[0,1]
#i_save_name =  "%s%s%s" % ("as_mu_fit_", int(i_tofit), ".dat")
#np.savetxt(i_save_name, np.transpose([as_mu_fit[:,0],as_mu_fit[:,1]]))

i_save_name_load =  "%s%s%s" % ("as_mu_fit_", int(i_tofit), ".dat")
as_mu_fit = np.genfromtxt(i_save_name_load)

time_con_day = ((time_con * u.s).to('day')).value
times_s_day = ((times_s * u.s).to('day')).value

res_ra_N = res_N[:,0]
res_dec_N = res_N[:,1]

res_ra_err = np.std(res_ra_N)# 
res_dec_err = np.std(res_dec_N)

#
as_mu_fit_8p = np.genfromtxt(f"{prefix1}as_mu_fit.dat")
res_N_8p = np.zeros((N_time-1, 2), dtype= np.float64)
res_N_8p = delta_dx_dy_onerefstar - as_mu_fit_8p 
res_ra_N_8p = res_N_8p[:,0]
res_dec_N_8p = res_N_8p[:,1] 

as_mu_fit_15p = np.genfromtxt(f"{prefix}as_mu_fit.dat")
res_N_15p = np.zeros((N_time-1, 2), dtype= np.float64)
res_N_15p = delta_dx_dy_onerefstar - as_mu_fit_15p
res_ra_N_15p = res_N_15p[:,0]
res_dec_N_15p = res_N_15p[:,1] 

# 1
figure1 = corner.corner(i_ch_mergeomega_all,
        labels = [
    r"$cos(i_{1})$",
    r"$e_{1}$",
    r'$\Omega_{1}$ $[\mathrm{rad}]$',#
    r'$\omega_{1}$ $[\mathrm{rad}]$',
    r'$M_{0,1}$ $[\mathrm{rad}]$', #
    r'$M_{\mathrm{p},1}$ $[\mathrm{M}_{\oplus}]$',
    r"$P_{1}$ $[\mathrm{d}]$",
    r"$cos(i_{2})$",
    r"$e_{2}$",
    r'$\Omega_{2}$ $[\mathrm{rad}]$',#
    r'$\omega_{2}$ $[\mathrm{rad}]$',
    r'$M_{0,2}$ $[\mathrm{rad}]$', #
    r'$M_{\mathrm{p},2}$ $[\mathrm{M}_{\oplus}]$',
    r"$P_{2}$ $[\mathrm{d}]$",
    r"$cos(i_{3})$",
    r"$e_{3}$",
    r'$\Omega _{3}$ $[\mathrm{rad}]$',#
    r'$\omega _{3}$ $[\mathrm{rad}]$',
    r'$M_{0,3}$ $[\mathrm{rad}]$', #
    r'$M_{\mathrm{p},3}$ $[\mathrm{M}_{\oplus}]$',
    r"$P_{3}$ $[\mathrm{d}]$",
    r'$\sigma_{A}$ $[\mu \mathrm{as}]$',
            ],
        label_kwargs={"fontsize": 18},
        truths=n_truePara_all,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".3f",
        levels=1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
        show_titles=True,
        title_kwargs={"fontsize": 13,"loc":"left"}, #
        smooth=True,
        color='dodgerblue',#0066CC',#'#3399FF',
        truth_color='r',
        ) 
mpl.rcParams['font.size'] = 13.
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 13.
mpl.rcParams['ytick.labelsize'] = 13.
majorformatterresy = mpltk.FormatStrFormatter('%.1f')

left, width, gap = 0.58, 0.18, 0.23
bottom, height = 0.60, 0.28

# color = ['#969696','#993300']
color = ['lightseagreen','r']
marker = ['.','-']

lengedpos = None#(1.18,1.06)
lengedloc = None#'center'#'lower left'
zorder = [1, 5]

rect_scatter = [left, bottom + 0.1, width, height]
rect_histx = [left, bottom, width, 0.1]

rect_scatter2 = [left+gap, bottom + 0.1, width, height]
rect_histx2 = [left+gap, bottom, width, 0.1]

ax1 = plt.axes(rect_scatter2)

ax1.minorticks_on()

pl.setp(ax1.get_xticklabels(),
        visible=False)

ax1.errorbar(time_con_day[1:N_time], delta_dx_dy_onerefstar[:,0], 
            yerr=delta_dx_dy_sig[:,0], color=color[0],
            alpha=1, marker=marker[0],
            fmt='.', label="Observations",
            zorder =zorder[0])
ax1.plot(times_s_day[1:O_time], as_mu_fit[:,0], color=color[1], alpha=1,
                     linestyle = marker[1], #'dashed',
                     linewidth=2,
                     label="Fitted",
                     zorder =zorder[1])
ax1.legend(fontsize=12, ncol = 2, bbox_to_anchor = lengedpos, loc = lengedloc)

axres = plt.axes(rect_histx2)
axres.minorticks_on()
axres.errorbar(time_con_day[1:N_time], res_ra_N, yerr=res_ra_err, color=color[0],
            alpha=1, fmt='.',
            marker=marker[0], label=None,
            zorder =zorder[0])  
axres.plot([axres.get_xlim()[0], axres.get_xlim()[1]], [0, 0], 
            color = color[1], linestyle = '--', linewidth=2,
            zorder =zorder[1])
axres.set_xlim(ax1.get_xlim()[0], ax1.get_xlim()[1])
axres.yaxis.set_major_formatter(majorformatterresy)
axres.set_xlabel('times[' + '$\mathrm{d}$'+']')
ax1.set_ylabel('$\Delta$'+'ra ['+'$\mu \mathrm{as}$'+']')      
axres.set_ylabel('residuals ['+'$\mu \mathrm{as}$'+']')   

ax2 = plt.axes(rect_scatter)
ax2.minorticks_on()

pl.setp(ax2.get_xticklabels(),
        visible=False)

ax2.errorbar(time_con_day[1:N_time], delta_dx_dy_onerefstar[:,1], 
            yerr=delta_dx_dy_sig[:,1], color=color[0],
            alpha=1, marker=marker[0],
            fmt='.', label="Observations",
            zorder =zorder[0])
ax2.plot(times_s_day[1:O_time], as_mu_fit[:,1], color=color[1], alpha=1,
                     linestyle = marker[1], #'dashed',
                     linewidth=2,
                     label="Fitted",
                     zorder =zorder[1])
ax2.legend(fontsize=12, ncol = 2, bbox_to_anchor = lengedpos, loc = lengedloc)

axres2 = plt.axes(rect_histx)
axres2.minorticks_on()
axres2.errorbar(time_con_day[1:N_time], res_dec_N, yerr=res_dec_err, color=color[0],
            alpha=1, fmt='.',
            marker=marker[0], label=None,zorder =zorder[0])  
axres2.plot([axres.get_xlim()[0], axres.get_xlim()[1]], [0, 0], 
            color = color[1], linestyle = '--', linewidth=2,
            zorder =zorder[1])

axres2.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])
axres2.yaxis.set_major_formatter(majorformatterresy)
axres2.set_xlabel('times[' + '$\mathrm{d}$'+']')
ax2.set_ylabel('$\Delta$'+'dec ['+'$\mu \mathrm{as}$'+']')      
axres2.set_ylabel('residuals ['+'$\mu \mathrm{as}$'+']')

i_save_name =  "%s%s%s" % ("Omege_omege_corner_chain", int(i_tofit), "_with_fitted_three.png")
figure1.savefig(i_save_name, dpi=300)
plt.savefig(f'Omege_omege_corner_chain'+str(i_tofit)+'_with_fitted_three.pdf', doi=700)

#%%
#################
from PyAstronomy.pyTiming import pyPeriod

mpl.rcParams['font.size'] = 13.
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 13.
mpl.rcParams['ytick.labelsize'] = 13.
majorformatterresy = mpltk.FormatStrFormatter('%.1f')

i_dxdy_name_res =  "%s%s%s" % ("delta_dx_dy_", int(i_tofit), ".dat")
StarDistIndirect = np.genfromtxt(i_dxdy_name_res)
i_save_name_load =  "%s%s%s" % ("as_mu_fit_N_", int(i_tofit), ".dat")
as_mu_fit_N = np.genfromtxt(i_save_name_load)

res_ra_N = delta_dx_dy_onerefstar[:,0] - as_mu_fit_N[:,0] 
res_dec_N = delta_dx_dy_onerefstar[:,1] - as_mu_fit_N[:,1]

as_mu_fit_8p = np.genfromtxt(f"{prefix1}as_mu_fit.dat")
res_ra_N_8p = delta_dx_dy_onerefstar[:,0] - as_mu_fit_8p[:,0]
res_dec_N_8p = delta_dx_dy_onerefstar[:,1] - as_mu_fit_8p[:,1] 

as_mu_fit_15p = np.genfromtxt(f"{prefix}as_mu_fit.dat")
res_ra_N_15p = delta_dx_dy_onerefstar[:,0] - as_mu_fit_15p[:,0]
res_dec_N_15p = delta_dx_dy_onerefstar[:,1] - as_mu_fit_15p[:,1]

clp_nra_1 = pyPeriod.Gls((time_con[1:N_time], StarDistIndirect[:,1]))
clp_nra_1.info()
clp_nra_list_1 = (clp_nra_1.power).tolist()
max_ra_index_1 = clp_nra_list_1.index(max(clp_nra_list_1))
clp_ndec_1 = pyPeriod.Gls((time_con[1:N_time], StarDistIndirect[:,0]))
clp_ndec_1.info()
clp_ndec_list_1 = (clp_ndec_1.power).tolist()
max_dec_index_1 = clp_ndec_list_1.index(max(clp_ndec_list_1))

T_ra_1 = 1./clp_nra_1.freq[max_ra_index_1]
T_dec_1 = 1./clp_ndec_1.freq[max_dec_index_1]
# T_clp_1 = (T_ra_1 + T_dec_1)/2.
T_ra_day = T_ra_1 *u.s.to('day')
T_dec_day = T_dec_1 *u.s.to('day')

# Define FAP levels of 10%, 5%, and 1%
fapLevels = np.array([0.1, 0.05, 0.01])
# Obtain the associated power thresholds
plevels_nra_1 = clp_nra_1.powerLevel(fapLevels)
plevels_ndec_1 = clp_ndec_1.powerLevel(fapLevels)

# 
clp_nra_2 = pyPeriod.Gls((time_con[1:N_time], res_ra_N_8p))
clp_nra_2.info()
clp_nra_list_2 = (clp_nra_2.power).tolist()
max_ra_index_2 = clp_nra_list_2.index(max(clp_nra_list_2))
clp_ndec_2 = pyPeriod.Gls((time_con[1:N_time], res_dec_N_8p))
clp_ndec_2.info()
clp_ndec_list_2 = (clp_ndec_2.power).tolist()
max_dec_index_2 = clp_ndec_list_2.index(max(clp_ndec_list_2))

T_ra_2 = 1./clp_nra_2.freq[max_ra_index_2]
T_dec_2 = 1./clp_ndec_2.freq[max_dec_index_2]
# T_clp_1 = (T_ra_1 + T_dec_1)/2.
T_ra_day_2 = T_ra_2 *u.s.to('day')
T_dec_day_2 = T_dec_2 *u.s.to('day')

# Define FAP levels of 10%, 5%, and 1%
fapLevels = np.array([0.1, 0.05, 0.01])
# Obtain the associated power thresholds
plevels_nra_2 = clp_nra_2.powerLevel(fapLevels)
plevels_ndec_2 = clp_ndec_2.powerLevel(fapLevels)

# 
clp_nra_3 = pyPeriod.Gls((time_con[1:N_time], res_ra_N_15p))
clp_nra_3.info()
clp_nra_list_3 = (clp_nra_3.power).tolist()
max_ra_index_3 = clp_nra_list_3.index(max(clp_nra_list_3))
clp_ndec_3 = pyPeriod.Gls((time_con[1:N_time], res_dec_N_15p))
clp_ndec_3.info()
clp_ndec_list_3 = (clp_ndec_3.power).tolist()
max_dec_index_3 = clp_ndec_list_3.index(max(clp_ndec_list_3))

T_ra_3 = 1./clp_nra_3.freq[max_ra_index_3]
T_dec_3 = 1./clp_ndec_3.freq[max_dec_index_3]
# T_clp_1 = (T_ra_1 + T_dec_1)/2.
T_ra_day_3 = T_ra_3 *u.s.to('day')
T_dec_day_3 = T_dec_3 *u.s.to('day')

# Define FAP levels of 10%, 5%, and 1%

# Obtain the associated power thresholds
plevels_nra_3 = clp_nra_3.powerLevel(fapLevels)
plevels_ndec_3 = clp_ndec_3.powerLevel(fapLevels)

#
clp_nra = pyPeriod.Gls((time_con[1:N_time], res_ra_N))
clp_nra.info()
clp_nra_list = (clp_nra.power).tolist()
max_ra_index = clp_nra_list.index(max(clp_nra_list))
clp_ndec = pyPeriod.Gls((time_con[1:N_time], res_dec_N))
clp_ndec.info()
clp_ndec_list = (clp_ndec.power).tolist()
max_dec_index = clp_ndec_list.index(max(clp_ndec_list))

T_ra = 1./clp_nra.freq[max_ra_index]
T_dec = 1./clp_ndec.freq[max_dec_index]
# T_clp_1 = (T_ra_1 + T_dec_1)/2.
T_ra_day_ = T_ra *u.s.to('day')
T_dec_day_ = T_dec *u.s.to('day')

# Define FAP levels of 10%, 5%, and 1%
# fapLevels = np.array([0.1, 0.05, 0.01])
# Obtain the associated power thresholds
plevels_nra = clp_nra.powerLevel(fapLevels)
plevels_ndec = clp_ndec.powerLevel(fapLevels)

lengedpos1 = (0.46,1.25)    #None   #
lengedloc1 = 'center'

left, width = 0.15, 0.8
bottom, height = 0.15, 0.18
figuresize = (6.0,7.6)#None#

rect_scatter = [left, bottom+3*height, width, height]

rect_scatter2 = [left, bottom+2*height, width, height]
rect_scatter3 = [left, bottom+height, width, height]
rect_scatter4 = [left, bottom, width, height]

color_fap = ['steelblue', 'coral', 'forestgreen','firebrick', 'blueviolet', 'saddlebrown']

fig = plt.figure(figsize=figuresize)
ax1 = plt.axes(rect_scatter)
pl.setp(ax1.get_xticklabels(),visible=False)

ax1.semilogx(clp_nra_1.freq*3600*24, clp_nra_1.power, 'b-', label="$\Delta$"+"ra", alpha=0.7)
ax1.semilogx(clp_ndec_1.freq*3600*24, clp_ndec_1.power, 'r-', label="$\Delta$"+"dec", alpha=0.7)

ax1.text(clp_nra_1.freq[max_ra_index_1]*3600*24+0.000000005*3600*24, clp_nra_1.power[max_ra_index_1] - 0.12, '$T_\mathrm{ra}$'+' = ' + str(round(T_ra_day,3))+' days')
ax1.text(clp_nra_1.freq[max_ra_index_1]*3600*24+0.000000005*3600*24, clp_nra_1.power[max_ra_index_1] - 0.34, '$T_\mathrm{dec}$'+' = ' + str(round(T_dec_day,3))+' days')

# Add the FAP levels to the plot
for i in range(len(fapLevels)):
    ax1.plot([min(clp_nra_1.freq*3600*24), max(clp_nra_1.freq*3600*24)], [plevels_nra_1[i]]*2, color =color_fap[2*i], linestyle = '-',#[plevels_nra_1[i]]*2
             label="$FAP_\mathrm{ra}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)
    ax1.plot([min(clp_ndec_1.freq*3600*24), max(clp_ndec_1.freq*3600*24)], [plevels_ndec_1[i]]*2, color =color_fap[2*i+1], linestyle = '--',#[plevels_ndec_1[i]]*2
             label="$FAP_\mathrm{dec}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)
ax1.legend(fontsize=9, ncol = 4, bbox_to_anchor = lengedpos1, loc = lengedloc1)

#        
ax2 = plt.axes(rect_scatter2)
pl.setp(ax2.get_xticklabels(),visible=False)

ax2.semilogx(clp_nra_2.freq*3600*24, clp_nra_2.power, 'b-', label="ra", alpha=0.7)
ax2.semilogx(clp_ndec_2.freq*3600*24, clp_ndec_2.power, 'r-', label="dec", alpha=0.7)

ax2.text(clp_ndec_2.freq[max_dec_index_2]*3600*24+0.00000001*3600*24, clp_ndec_2.power[max_dec_index_2] - 0.12, '$T_\mathrm{ra}$'+' = ' + str(round(T_ra_day_2,3))+' days')
ax2.text(clp_ndec_2.freq[max_dec_index_2]*3600*24+0.00000001*3600*24, clp_ndec_2.power[max_dec_index_2] - 0.25, '$T_\mathrm{dec}$'+' = ' + str(round(T_dec_day_2,3))+' days')

# Add the FAP levels to the plot
for i in range(len(fapLevels)):
    ax2.plot([min(clp_nra_2.freq*3600*24), max(clp_nra_2.freq*3600*24)], [plevels_nra_2[i]]*2, color =color_fap[2*i], linestyle = '-',#[plevels_nra_1[i]]*2
             label="$FAP_\mathrm{ra}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)
    ax2.plot([min(clp_ndec_2.freq*3600*24), max(clp_ndec_2.freq*3600*24)], [plevels_ndec_2[i]]*2, color =color_fap[2*i+1], linestyle = '--',#[plevels_ndec_1[i]]*2
             label="$FAP_\mathrm{dec}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)

#
ax3 = plt.axes(rect_scatter3)
pl.setp(ax3.get_xticklabels(),visible=False)

ax3.semilogx(clp_nra_3.freq*3600*24, clp_nra_3.power, 'b-', label="ra", alpha=0.7)
ax3.semilogx(clp_ndec_3.freq*3600*24, clp_ndec_3.power, 'r-', label="dec", alpha=0.7)

ax3.text(clp_nra_3.freq[max_ra_index_3]*3600*24+0.00000001*3600*24, clp_nra_3.power[max_ra_index_3] - 0.15, '$T_\mathrm{ra}$'+' = ' + str(round(T_ra_day_3,3))+' days')
ax3.text(clp_nra_3.freq[max_ra_index_3]*3600*24+0.00000001*3600*24, clp_nra_3.power[max_ra_index_3] - 0.28, '$T_\mathrm{dec}$'+' = ' + str(round(T_dec_day_3,3))+' days')

# Add the FAP levels to the plot
for i in range(len(fapLevels)):
    ax3.plot([min(clp_nra_3.freq*3600*24), max(clp_nra_3.freq*3600*24)], [plevels_nra_3[i]]*2, color =color_fap[2*i], linestyle = '-',#[plevels_nra_1[i]]*2
             label="$FAP_\mathrm{ra}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)
    ax3.plot([min(clp_ndec_3.freq*3600*24), max(clp_ndec_3.freq*3600*24)], [plevels_ndec_3[i]]*2, color =color_fap[2*i+1], linestyle = '--',#[plevels_ndec_1[i]]*2
             label="$FAP_\mathrm{dec}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)

ax4 = plt.axes(rect_scatter4)

ax4.semilogx(clp_nra.freq*3600*24, clp_nra.power, 'b-', label="ra_res", alpha=0.7)
ax4.semilogx(clp_ndec.freq*3600*24, clp_ndec.power, 'r-', label="dec_res", alpha=0.7)

ax4.text(0.00001, 0.145, 'Power', fontsize=15.,rotation='vertical')
ax4.text(0.0004, -0.032, 'Frequency('+ '$\mathrm{d}^{-1}$'+')', fontsize=15.)

# Add the FAP levels to the plot
for i in range(len(fapLevels)):
    ax4.plot([min(clp_nra.freq*3600*24), max(clp_nra.freq*3600*24)], [plevels_nra[i]]*2, color = color_fap[2*i], linestyle = '-',#
             label="$FAP_\mathrm{ra}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)
    ax4.plot([min(clp_ndec.freq*3600*24), max(clp_ndec.freq*3600*24)], [plevels_ndec[i]]*2, color = color_fap[2*i+1], linestyle = '--',
             label="$FAP_\mathrm{dec}$"+" = %4.1f%%" % (fapLevels[i]*100), alpha=0.6)

plt.savefig(f'LombScargle_topBottom_'+str(i_tofit)+'_three.pdf')

print(time.ctime())
print("end")

# %%
from scipy import stats

N_time1 = N_time - 1
chisl = np.zeros((N_time1,2),dtype=float)
chisl[:, 0] = ( res_N[:,0]) **2./ (delta_dx_dy_sig[:,0] **2. + p_fit[21] **2.)
chisl[:, 1] = ( res_N[:,1]) **2./ (delta_dx_dy_sig[:,1] **2. + p_fit[21] **2.)

chisl2 = np.sum(chisl)
dof = 2*N_time1 - 8
print(chisl2/dof)

P_value = 1 - stats.chi2.cdf(chisl2,df=dof)
print(P_value)

mean_res_ra = np.mean( res_N[:,0])
mean_res_dec = np.mean( res_N[:,1])

std_res_ra = np.std( res_N[:,0])
std_res_dec = np.std( res_N[:,1])

print([mean_res_ra, mean_res_dec, std_res_ra, std_res_dec])
np.savetxt(f'{prefix}GoodnessOfFit.dat', np.transpose([chisl2/dof,P_value,mean_res_ra, mean_res_dec, std_res_ra, std_res_dec]))

# %%
print("end")
# %%
