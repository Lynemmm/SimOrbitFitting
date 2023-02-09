#%%
from constant import year, degree1_mu
from math import radians,cos,sin
import numpy as np

one_planet = True #False #
one_planet_sim = False #True #

#target star:
RA0 = 323.3912519 #* degree1_mu
DEC0 = -49.0126304 #* degree1_mu

#field of view:
w_ra_dec = 0.44

# number reference stars:
n_ref_stars = 8
# pm, mu as/year
ref_stars_pm_ra = [0.000, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.00] 
# pm, mu as/year
ref_stars_pm_dec = [0.000, 0.0, 0.00, 0.00, -0.00, -0.0, -0.0, -0.00] 
#ref_stars_pm_dec = [0.001, 0.1, 0.01, 0.02, -0.02, -0.01, -0.1, -0.001] 
RefStarRA =  [  323.544365- RA0,     323.3458495- RA0,    323.5172585- RA0,    323.2590608- RA0,    323.4500217- RA0,    323.2923339- RA0,    323.4506681- RA0,    323.3883039- RA0]
RefStarDEC = [  -49.02745022- DEC0,   -48.96336483- DEC0,   -48.98091961- DEC0,    -48.89512317- DEC0,  -49.00220616- DEC0,   -48.96792283- DEC0,   -49.08274028- DEC0,   -48.91450989- DEC0]

# RefStarRA = [176.7618325, 177.0431241, 176.9780821, 176.9180022, 177.1363097, 176.9824735, 176.8289534, 177.0720686] 
# RefStarDEC = [0.732969848, 0.905005658, 0.929610296, 0.613761862, 0.759649228, 0.833993777, 0.888085912, 0.775217915] 
RefStarPos = np.array([RefStarRA, RefStarDEC]) * degree1_mu

# noise simulation
noise_mean = 0.0
noise_std = 1.0

noise_mean_Jitter = 0.
noise_std_Jitter = 3600.

# proxima A,B
# Proxima C
#mp_Mearth = 312.0
#ms_Msun = 0.1221
#d_pc = 1.3012
#a_AU = 5.2

ms_Msun = 0.45
d_pc = 4.95
#
i_orbit2 = 67.0
mp_Mearth2 = 5.4 / sin(radians(i_orbit2))

# a_AU = 0.16250966975474035
#alpha = 1.2841980446302246
P_day2 = 35.68

e_orbit2 = 0.18
# i_orbit: degree: 0-90 or 0-180 ?

cos_i_orbit2 = cos(radians(i_orbit2))
# ascend_node_Omega: phase
ascend_node_Omega2 =  radians(105.3) # 0-360
# periapsis_omega: phase
periapsis_omega2 = radians(10.0+180.)  # 0-360
periapsis_omega2_in = radians(10.)
# M0: phase
# M0 = radians(98.03252677508938) # 0-360
M02 = radians(165.)
# T0 = 2456882.
#

# RA02 = 150.0
# DEC02 = 14.0

i_orbit = 38.3 
mp_Mearth = 216. / sin(radians(i_orbit))

# a_AU2 = 3.562527358673324
# alpha = 1672.1630485766104
P_day = 3657.

e_orbit = 0.08
 
cos_i_orbit = cos(radians(i_orbit))#-0.259 #
ascend_node_Omega =   radians(63.7) # 0-360
periapsis_omega =  radians(246.0+180.)  # 0-360
periapsis_omega_in = radians(246.)
M0 =  radians(307.) # 0-360
# T02 = 2456478.
#



# %%
########## start time 2025.01.01 12:00 ###########
#unit: day
begin_year = 2025
begin_month = 1
begin_day = 1
begin_time = (12. - 12.) / 24 
timesteps = 300 
ObservationPeriod = 5. #* u.year