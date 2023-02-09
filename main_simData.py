#%%
from turtle import end_fill
import numpy as np
import copy
import time
from scipy import constants
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPeriod
import astropy.constants as cons
import astropy.units as u
#import pandas as pd
#import pickle
#from matplotlib.colors import LogNorm
#%%
# from constant import *
from inputsimobs import *
from inputbayes import *
import class_mcmc
import func_orbit,func_bayes
import corner 

import DataGenFunc
from math import radians
import rebound

debug = False #True



print(time.ctime())

# new time seq or not
new_time = False #True # 

# create new ref astrometry
create_sim_delta_dxdy = True #False #

# ref star
i_tofit = 0

if new_time:
    ##########################
    time_con = DataGenFunc.create_time_series(timesteps, ObservationPeriod)

    np.savetxt("t_all.dat", np.transpose([time_con]))
else:
    time_con = np.genfromtxt('t_all.dat')

#%%

N_time = len(time_con)

# ##########################
# # astrometry sequence

P_s = (P_day * u.day).to('s')
P_s2 = (P_day2 * u.day).to('s')
P_s3 = (P_day3 * u.day).to('s')
mu = cons.G * (ms_Msun * cons.M_sun + mp_Mearth * cons.M_earth)
mu2 = cons.G * (ms_Msun * cons.M_sun + mp_Mearth2 * cons.M_earth)
mu3 = cons.G * (ms_Msun * cons.M_sun + mp_Mearth3 * cons.M_earth)
ap =  pow(mu * P_s **2. / (4. * np.pi ** 2.), 1/3 ) / (1. * cons.au)
ap2 =  pow(mu2 * P_s2 **2. / (4. * np.pi ** 2.), 1/3 ) / (1. * cons.au)
ap3 =  pow(mu3 * P_s3 **2. / (4. * np.pi ** 2.), 1/3 ) / (1. * cons.au)

M0_cal = DataGenFunc.Cal_M0(T0 ,begin_year, begin_month, begin_day, begin_time, P_day)
M02_cal = DataGenFunc.Cal_M0(T02 ,begin_year, begin_month, begin_day, begin_time, P_day2)
M03_cal = DataGenFunc.Cal_M0(T03 ,begin_year, begin_month, begin_day, begin_time, P_day3)
    
sim = rebound.Simulation()
sim.add(m=ms_Msun)     
sim.add(m=mp_Mearth * 3e-6, e=e_orbit, a=ap, inc=radians(i_orbit), Omega=ascend_node_Omega, omega=periapsis_omega_in, M=M0_cal)     
sim.add(m=mp_Mearth2 * 3e-6, e=e_orbit2, a=ap2, inc=radians(i_orbit2), Omega=ascend_node_Omega2, omega=periapsis_omega2_in, M=M02_cal)    
sim.add(m=mp_Mearth3 * 3e-6, e=e_orbit3, a=ap3, inc=radians(i_orbit3), Omega=ascend_node_Omega3, omega=periapsis_omega3_in, M=M03_cal)    
sim.move_to_com() 

figObit, ax = rebound.OrbitPlot(sim, unitlabel="[AU]")
plt.savefig('Orbit.pdf')

Pos = np.zeros((2,N_time),dtype=float)
for i in range(N_time):
    sim.integrate(2*np.pi/(365.25*u.day.to('s'))*time_con[i])
    Pos[0,i] = sim.particles[0].x
    Pos[1,i] = sim.particles[0].y
Pos_aus = (np.arctan(Pos*cons.au/(d_pc*cons.pc)).to('uas')).value
as_cal_con = Pos_aus
as_mu = np.transpose(Pos_aus)

if debug:
    np.savetxt("as_mu.dat", np.transpose([as_mu[:,0],as_mu[:,1]]))

#%%
as_cal_con_c = DataGenFunc.GenModelPos_oneplanet(e_orbit, cos_i_orbit, ascend_node_Omega, periapsis_omega, M0_cal, P_day, mp_Mearth, ms_Msun, d_pc, 0., 0., time_con)
as_cal_con2_c = DataGenFunc.GenModelPos_oneplanet(e_orbit2, cos_i_orbit2, ascend_node_Omega2, periapsis_omega2, M02_cal, P_day2, mp_Mearth2, ms_Msun, d_pc, 0., 0., time_con)
as_cal_con3_c = DataGenFunc.GenModelPos_oneplanet(e_orbit3, cos_i_orbit3, ascend_node_Omega3, periapsis_omega3, M03_cal, P_day3, mp_Mearth3, ms_Msun, d_pc, 0., 0., time_con)
as_mu1_c = np.transpose(as_cal_con_c)
as_mu2_c = np.transpose(as_cal_con2_c)
as_mu3_c = np.transpose(as_cal_con3_c)
as_mu_c = as_mu1_c + as_mu2_c + as_mu3_c

np.savetxt("as_mu_c.dat", np.transpose([as_mu_c[:,0],as_mu_c[:,1]]))

#%%
fig = plt.figure()
plt.scatter(as_mu[:,0], as_mu[:,1])
plt.scatter(as_mu_c[:,0], as_mu_c[:,1])
fig.savefig("OriginalData.png", dpi=300)

fig_n = plt.figure()
plt.subplot(2,1,1)
plt.plot(time_con, as_mu[:,0],'r-')
plt.plot(time_con, as_mu_c[:,0],'b-')
plt.xlabel("times -s")
plt.ylabel("RA -uas")

plt.subplot(2,1,2)
plt.plot(time_con, as_mu[:,1],'r-')
plt.plot(time_con, as_mu_c[:,1],'b-')
plt.xlabel("times -s")
plt.ylabel("DEC -uas")
fig_n.savefig("OriginalDataInTime.png", dpi=300)

fig_r = plt.figure()
plt.subplot(2,1,1)
plt.plot(time_con, as_mu[:,0] - as_mu_c[:,0],'r-')
plt.xlabel("times -s")
plt.ylabel("RA res -uas")

plt.subplot(2,1,2)
plt.plot(time_con, as_mu[:,1] - as_mu_c[:,1],'r-')
plt.xlabel("times -s")
plt.ylabel("DEC res -uas")
fig_n.savefig("OriginalDataInTime_two.png", dpi=300)
#%%
if create_sim_delta_dxdy:
    ##########################
    ##### simulation delta dx dy

    delta_dx_dy_pm = DataGenFunc.GenSimDataFunc(as_cal_con, RefStarPos, n_ref_stars, time_con, noise_mean, noise_std, noise_mean_Jitter, noise_std_Jitter)
   
    if debug:
        for i in range(n_ref_stars):
            delta_dx_dy_name = "%s%s%s" % ("delta_dx_dy_", i, ".dat")
            np.savetxt(delta_dx_dy_name, np.transpose([delta_dx_dy_pm[i], delta_dx_dy_pm[i + n_ref_stars]]))
    ##### calc mean

    delta_dx_dy_sig = np.ones((int(N_time)-1, 2), dtype=np.float64) * np.sqrt(noise_std ** 2. + noise_std ** 2.)

    if debug:

        np.savetxt("delta_dx_dy_sig.dat", np.transpose([delta_dx_dy_sig[:,0], delta_dx_dy_sig[:,1]]))

delta_dx_dy_sig = np.genfromtxt("delta_dx_dy_sig.dat")

#%%
fig1 = plt.figure()
plt.scatter(as_mu[1:N_time,0] - as_mu[0,0], as_mu[1:N_time,1] - as_mu[0,1], color = 'r')
plt.scatter(delta_dx_dy_pm[0], delta_dx_dy_pm[8])
fig1.savefig("SimData.png", dpi=300)
fig1_n = plt.figure()
plt.subplot(2,1,1)
plt.plot(time_con[1:N_time], delta_dx_dy_pm[0],'bo')
plt.plot(time_con[1:N_time], as_mu[1:N_time,0] - as_mu[0,0],'r-')
plt.xlabel("times -s")
plt.ylabel("RA -uas")

plt.subplot(2,1,2)
plt.plot(time_con[1:N_time], delta_dx_dy_pm[8],'bo')
plt.plot(time_con[1:N_time], as_mu[1:N_time,1] - as_mu[0,1],'r-')
plt.xlabel("times -s")
plt.ylabel("DEC -uas")
fig1_n.savefig("OriginalDataInTime.png", dpi=300)

#%%
time_con = np.genfromtxt('t_all.dat')
N_time = len(time_con)
StarDistIndirect = np.genfromtxt('delta_dx_dy_0.dat')
# 1
clp_nra_1 = pyPeriod.Gls((time_con[1:N_time], StarDistIndirect[:,0]))
clp_nra_1.info()
clp_nra_list_1 = (clp_nra_1.power).tolist()
max_ra_index_1 = clp_nra_list_1.index(max(clp_nra_list_1))
clp_ndec_1 = pyPeriod.Gls((time_con[1:N_time], StarDistIndirect[:,1]))
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
plt.text(clp_ndec_1.freq[max_dec_index_1], clp_ndec_1.power[max_dec_index_1] - 0.2, 'T_dec = ' + str(T_dec_1))
clp_fig_1.savefig("LombScargle.png", dpi=300)


print(time.ctime())

print("end")



# %%
