#%%
from constant import year, degree1_mu
from math import radians,cos
import numpy as np

one_planet = True #False #
one_planet_sim = False #True #

#target star:
RA0 = 176.937688 #* degree1_mu
DEC0 = 0.79911997 #* degree1_mu

#field of view:
w_ra_dec = 0.44

# number reference stars:
n_ref_stars = 8
# pm, mu as/year
ref_stars_pm_ra = [0.000, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.00] 
# pm, mu as/year
ref_stars_pm_dec = [0.000, 0.0, 0.00, 0.00, -0.00, -0.0, -0.0, -0.00] 
#ref_stars_pm_dec = [0.001, 0.1, 0.01, 0.02, -0.02, -0.01, -0.1, -0.001] 
RefStarRA = [176.7618325- RA0, 177.0431241- RA0, 176.9780821- RA0, 176.9180022- RA0, 177.1363097- RA0, 176.9824735- RA0, 176.8289534- RA0, 177.0720686- RA0] 
RefStarDEC = [0.732969848- DEC0, 0.905005658- DEC0, 0.929610296- DEC0, 0.613761862- DEC0, 0.759649228- DEC0, 0.833993777- DEC0, 0.888085912- DEC0, 0.775217915- DEC0] 
RefStarPos = np.array([RefStarRA, RefStarDEC]) * degree1_mu

# observation period ( time in days )
t0 = 0
t1 = year*5
# observation frequency
N_time_init = 300.0
drop_ratio = 0.15 # drop_ratio*100%
drop_seg_min = 5
drop_seg_max = 8

# noise simulation
noise_mean = 0.0
noise_std = 1. #0.1#

noise_mean_Jitter = 0.
noise_std_Jitter = 3600.

# proxima A,B
# Proxima C
#mp_Mearth = 312.0
#ms_Msun = 0.1221
#d_pc = 1.3012
#a_AU = 5.2

ms_Msun = 1.1
d_pc = 5.2
#
mp_Mearth = 1.38

# a_AU = 1.2
P_day = 457.797

e_orbit = 0.316
# i_orbit: degree: 0-90 or 0-180 ?
i_orbit = 56.
cos_i_orbit = cos(radians(i_orbit))
# ascend_node_Omega: phase
ascend_node_Omega =  radians(100.) # 0-360
# periapsis_omega: phase
periapsis_omega = radians(86.0+180.)  # 0-360
periapsis_omega_in = radians(86.)
# M0: phase
# M0 = radians(98.03252677508938) # 0-360
T0 = 245674.
#

# RA02 = 150.0
# DEC02 = 14.0

mp_Mearth2 = 7.6

# a_AU2 = 1.799992816943883
P_day2 = 841.015

e_orbit2 = 0.13
i_orbit2 = 205.3  #1.8327830575923905
cos_i_orbit2 = cos(radians(i_orbit2))#-0.259 #
ascend_node_Omega2 =   radians(49.0) # 0-360
periapsis_omega2 =  radians(28.0+180.)  # 0-360
periapsis_omega2_in = radians(28.)
# M02 =  radians(204.78537620368868) # 0-360
T02 = 2456478.
#

mp_Mearth3 = 32.

# a_AU2 = 3.3
P_day3 = 2087.63

e_orbit3 = 0.08
i_orbit3 = 82.3  #1.8327830575923905
cos_i_orbit3 = cos(radians(i_orbit3))#-0.259 #
ascend_node_Omega3 =   radians(138.0) # 0-360
periapsis_omega3 =  radians(116.0+180.)  # 0-360
periapsis_omega3_in = radians(116.)
# M02 =  radians(204.78537620368868) # 0-360
T03 = 2456598.

# %%
########## start time 2025.01.01 12:00 ###########
#unit: day
begin_year = 2025
begin_month = 1
begin_day = 1
begin_time = (12. - 12.) / 24 
timesteps = 300 
ObservationPeriod = 5. #* u.year