#   author : Yaning Liu
#   Date : 2022/01/30 
#   Version : V1.0

#   Date : 2022/02/07 
#   Version : V1.1
#   modified: add newton_solver function

#   Date : 2022/02/22 
#   Version : V1.2
#   modified: 1, change GenTimeSerials to three functions: Time2Jdn, Cal_M0, create_time_series
#             2, add GenSimDataFunc
#             3, GenRelativePos function: change "return StarDist" to "return StarDist[:, 1:length]"
#%%  
import numpy as np
import astropy.constants as cons
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.ticker import FormatStrFormatter

#%%


def Time2Jdn(ref_year, ref_month, ref_day, ref_time):
    ref_a= int((14 - ref_month) / 12)
    ref_y = ref_year + 4800 - ref_a
    ref_m = ref_month + 12 * ref_a - 3

    ref_JDN = ref_day + int((153*ref_m+2)/5) + 365 * ref_y + int(ref_y/4) - int(ref_y/100) + int(ref_y/400) - 32045 + ref_time 

    return ref_JDN # unit: day

def Cal_M0(T0,begin_year, begin_month, begin_day, begin_time, P):

    begin_JDN = Time2Jdn(begin_year, begin_month, begin_day, begin_time)

    t_ref = begin_JDN - T0

    n = 2. * np.pi / P
    M0 = n * (t_ref - int(t_ref / P) * P)   # fixed 4.918510422606191 rad, 281.80988955110905 deg

    return M0

def create_time_series(timesteps, ObservationPeriod):

    # begin_JDN = Time2Jdn(begin_year, begin_month, begin_day, begin_time)

    times = np.linspace(0, ObservationPeriod, timesteps)# + ((begin_JDN * u.day).to('year')).value

    #### remove 50 random positons from 300 ###
    timegap_pos = np.random.randint(0,timesteps,(5,))   
    timegap = np.zeros((50,), dtype=int)

    for i in range(5):
        timegap[10 * i : 10 * i + 10] = np.linspace(timegap_pos[i], timegap_pos[i] + 9, 10, dtype = int)

    times_random = []
    for i in range(timesteps):
        if i not in timegap:
            times_random.append(times[i])
    ############################################

    times_s = ((times_random * u.year).to('s')).value 

    return times_s

def newton_solver(M, e, EE0=None):
    tolerance=1e-9
    max_iter=100
    M = np.array(M)
    if EE0 is None:
        EE = np.copy(M)
    else:
        EE = np.copy(EE0)
    EE -= (EE - (e * np.sin(EE)) - M) / (1.0 - (e * np.cos(EE)))
    diff = (EE - (e * np.sin(EE)) - M) / (1.0 - (e * np.cos(EE)))
    abs_diff = abs(diff)
    ind = np.where(abs_diff > tolerance)
    niter = 0
    while ((ind[0].size > 0) and (niter <= max_iter)):
        EE[ind] -= diff[ind]
        if niter == (max_iter//2):
            EE[ind] = np.pi
        diff[ind] = (EE[ind] - (e * np.sin(EE[ind])) - M[ind]) / (1.0 - (e * np.cos(EE[ind])))
        abs_diff[ind] = abs(diff[ind])
        ind = np.where(abs_diff > tolerance)
        niter += 1
    return EE

def GenModelPos_oneplanet(e, cos_inc, Omega, omega, M0, P, mplanet, mstar, Dist, RA0, DEC0, times_s):#ap, 

    # mstar = mstar * cons.M_sun 
    # mplanet = mplanet * cons.M_earth 
    # ap = ap * cons.au 
    # Omega = Omega * u.rad
    # omega = omega * u.rad
    # inc = inc * u.rad
    P_in = (P * u.day).to('s')
    # M0 = M0 * u.rad
	

    mu = cons.G * (mstar * cons.M_sun + mplanet * cons.M_earth)
    ap =  pow(mu * P_in **2. / (4. * np.pi ** 2.), 1/3 )  # unit : m
    # P = np.sqrt(4. * np.pi ** 2. * ap ** 3. / mu)

    n = 2. * np.pi / P_in.value
    mean_anomalies = n * times_s + M0   # unit : rad
    alpha_s = 3. * mplanet * (ap / (1. * cons.au))/ mstar / Dist  # unit : uas

    A = alpha_s * (np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*cos_inc)   # unit : uas
    B = alpha_s * (np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*cos_inc)
    F = alpha_s * (- np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*cos_inc)
    G = alpha_s * (- np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*cos_inc)

    E_as = newton_solver(mean_anomalies, e)

    ra_x = np.cos(E_as) - e                       # unit : none
    dec_y = np.sqrt(1 - e ** 2) * np.sin(E_as)    # unit : none

    RA = A*ra_x + F*dec_y  # unit : uas
    DEC = B*ra_x + G*dec_y # unit : uas

    IdealPosPeriod = np.array([RA,DEC])

    TarStarIdealPos = np.zeros((2,len(times_s)), dtype = np.float64)
    TarStarIdealPos[0] = RA0 + IdealPosPeriod[0]
    TarStarIdealPos[1] = DEC0 + IdealPosPeriod[1]

    return TarStarIdealPos # 2 rows ,uas

def GenRefStarSerials(RefStarPos_list, times_s, n_ref_stars): 

    length = len(times_s)
    referStar_RA = (np.array(RefStarPos_list[0]).repeat(length, axis=0)).reshape(n_ref_stars, length) 
    referStar_DEC = (np.array(RefStarPos_list[1]).repeat(length, axis=0)).reshape(n_ref_stars, length)

    RefStarPos = np.zeros((2 * n_ref_stars, length), dtype = np.float64)
    RefStarPos[0: n_ref_stars] = referStar_RA
    RefStarPos[n_ref_stars: 2 * n_ref_stars] = referStar_DEC

    return RefStarPos   # 2 * n_ref_stars rows ,uas

def AddJitterNoise(noise_mean, noise_std, TarStarPos, RefStarPos, n_ref_stars):

    length = len(TarStarPos[0])
    AllStarPos_J = np.zeros((2 * n_ref_stars + 2, length), dtype = np.float64)

    noise = np.random.normal(noise_mean, noise_std, [2, length - 1])
    AllStarPos_J[[0, n_ref_stars + 1], 0] = TarStarPos[0: 2,0]
    AllStarPos_J[[0, n_ref_stars + 1], 1: length] = TarStarPos[0: 2, 1: length] + noise

    noise_16 = np.repeat(noise, [n_ref_stars, n_ref_stars], axis=0)
    AllStarPos_J[1: n_ref_stars + 1, 0] = RefStarPos[0: n_ref_stars, 0]
    AllStarPos_J[1: n_ref_stars + 1, 1: length] = RefStarPos[0: n_ref_stars, 1: length] + noise_16[0: n_ref_stars]

    AllStarPos_J[n_ref_stars + 2: 2 * n_ref_stars + 2, 0] = RefStarPos[n_ref_stars: 2 * n_ref_stars, 0]
    AllStarPos_J[n_ref_stars + 2: 2 * n_ref_stars + 2, 1: length] = RefStarPos[n_ref_stars: 2 * n_ref_stars, 1: length] + noise_16[n_ref_stars: 2 * n_ref_stars]

    return AllStarPos_J  # 2 + 2 * n_ref_stars  rows [tar_ra; ref1-8_ra; tar_dec; ref1-8_dec] ,uas

def AddAberrationNoise(noise_Aberration_hight, noise_std, AllStarPos_J, n_ref_stars):
    
    length = len(AllStarPos_J[0])
    A_all = TwoDimen_GausianLike_Fun(AllStarPos_J[0: n_ref_stars+1], AllStarPos_J[n_ref_stars+1: 2*n_ref_stars+2], AllStarPos_J[0][0], AllStarPos_J[n_ref_stars+1][0], noise_std, noise_Aberration_hight)

    Delta_ra = AllStarPos_J[0: n_ref_stars+1] - AllStarPos_J[0][0]
    Delta_dec= AllStarPos_J[n_ref_stars+1: 2*n_ref_stars+2] - AllStarPos_J[n_ref_stars+1][0]
    
    AberrationNoise = np.zeros((2 * n_ref_stars + 2, length), dtype = np.float64)

    for i in range(0, n_ref_stars + 1):
        for j in range(0, length):
            if Delta_ra[i][j] == 0:
                AberrationNoise[i][j] = 0.
                if Delta_dec[i][j] == 0.:
                    AberrationNoise[i + n_ref_stars + 1][j] = 0.
                elif Delta_dec[i][j] > 0.:
                    AberrationNoise[i + n_ref_stars + 1][j] = A_all[i][j]
                else : 
                    AberrationNoise[i + n_ref_stars + 1][j] = - A_all[i][j]
            else:
                theta = np.arctan(Delta_dec[i][j] / Delta_ra[i][j])
                if theta == 0.:
                    AberrationNoise[i + n_ref_stars + 1][j] = 0.  
                    if Delta_ra[i][j] > 0.:
                        AberrationNoise[i][j] = A_all[i][j]
                    else: 
                        AberrationNoise[i][j] = - A_all[i][j]
                        
                elif theta > 0.:
                    if Delta_ra[i][j] > 0.:
                        AberrationNoise[i][j] = A_all[i][j] * np.cos(theta)
                        AberrationNoise[i + n_ref_stars + 1][j] = A_all[i][j] * np.sin(theta)
                    else:
                        AberrationNoise[i][j] = - A_all[i][j] * np.cos(theta)
                        AberrationNoise[i + n_ref_stars + 1][j] = - A_all[i][j] * np.sin(theta)
                elif theta < 0.:
                    if Delta_ra[i][j] > 0.:
                        AberrationNoise[i][j] = A_all[i][j] * np.cos(theta)
                        AberrationNoise[i + n_ref_stars + 1][j] = - A_all[i][j] * np.sin(theta)
                    else:
                        AberrationNoise[i][j] = - A_all[i][j] * np.cos(theta)
                        AberrationNoise[i + n_ref_stars + 1][j] = A_all[i][j] * np.sin(theta)
    
    AllStar_FinalPos = np.zeros((2 * n_ref_stars + 2, length), dtype = np.float64)
    AllStar_FinalPos = AllStarPos_J + AberrationNoise

    return AllStar_FinalPos  # 2 + 2 * n_ref_stars  rows [tar_ra; ref1-8_ra; tar_dec; ref1-8_dec] ,uas

def GenRelativePos(AllStar_FinalPos, n_ref_stars):

    length  = len(AllStar_FinalPos[0])
    
    RepeatNum = np.repeat(1, 2)
    RepeatNum[0] = length

    PosRept = np.repeat(AllStar_FinalPos[0:2 + 2 * n_ref_stars,0:2], RepeatNum, axis=1)

    StarDist = np.zeros((2 * n_ref_stars + 2, length), dtype = np.float64)

    StarDist = AllStar_FinalPos - PosRept[0: 2 * n_ref_stars + 2, 0:length]

    return StarDist[:, 1:length]     # 2 * n_ref_stars + 2  rows ,uas

def GenSimRelativePos(StarDist, noise_mean, noise_std, n_ref_stars):

    length  = len(StarDist[0])
    
    noise = np.random.normal(noise_mean, noise_std, [2 * n_ref_stars + 2, length])

    StarDistDelta = np.zeros((2 * n_ref_stars + 2, length), dtype = np.float64)
    # StarDistDelta[0: 2 * n_ref_stars + 2, 0] = StarDist[0: 2 * n_ref_stars + 2, 0]
    # StarDistDelta[0: 2 * n_ref_stars + 2, 1:length] = StarDist[0: 2 * n_ref_stars + 2, 1:length] + noise
    StarDistDelta = StarDist + noise

    # StarDistMean = np.zeros((2, length), dtype = np.float64)
    # StarDistMean[0] = np.mean(StarDistDelta[0: n_ref_stars],0)
    # StarDistMean[1] = np.mean(StarDistDelta[n_ref_stars: 2 * n_ref_stars],0)

    return StarDistDelta # 2 * n_ref_stars + 2  rows ,uas


def GenRelativeIndirectPos(StarDistDelta, n_ref_stars):
    
    length  = len(StarDistDelta[0])

    RepeatNum = np.repeat(1, 2 * n_ref_stars + 2)
    RepeatNum[0] = n_ref_stars
    RepeatNum[n_ref_stars + 1] = n_ref_stars

    PosRept = np.repeat(StarDistDelta, RepeatNum, axis=0)

    StarDistIndirect = np.zeros((2 * n_ref_stars, length), dtype = np.float64)

    StarDistIndirect[0: n_ref_stars] = PosRept[0: n_ref_stars] - PosRept[n_ref_stars: 2 * n_ref_stars]
    StarDistIndirect[n_ref_stars: 2 * n_ref_stars] = PosRept[2 * n_ref_stars: 3 * n_ref_stars] - PosRept[3 * n_ref_stars: 4 * n_ref_stars]

    return StarDistIndirect     # 2 * n_ref_stars  rows ,uas

# def GenSimRefStarPos(AllStar_FinalPos, n_ref_stars):
    
#     length = len(AllStar_FinalPos[1])
    
#     SimRefStarPos = np.zeros((2 * n_ref_stars, length), dtype = np.float64)

#     SimRefStarPos[0: n_ref_stars] = AllStar_FinalPos[1: n_ref_stars + 1]
#     SimRefStarPos[n_ref_stars: 2 * n_ref_stars] = AllStar_FinalPos[n_ref_stars + 2: 2 * n_ref_stars + 2]

#     return SimRefStarPos  # 2 * n_ref_stars  rows ,uas

# def GenSimTarStarPos(StarDistDelta, SimRefStarPos, n_ref_stars):
    
#     length  = len(StarDistDelta[0])

#     SimTarStarPos = SimRefStarPos - StarDistDelta

#     SimTarStarPosMean = np.zeros((2, length), dtype = np.float64)
#     SimTarStarPosMean[0] = np.mean(SimTarStarPos[0: n_ref_stars],0)
#     SimTarStarPosMean[1] = np.mean(SimTarStarPos[n_ref_stars: 2 * n_ref_stars],0)

#     return SimTarStarPosMean # 2 rows, uas

def TwoDimen_GausianLike_Fun(x, y, x0, y0, sigma, Aberration_hight):
    # assum sigma = 100. uas, G_hight = 10000 uas

    # G = 1./ (2. * np.pi * sigma ** 2) * np.exp( - ((x - x0) ** 2 + (y - y0) ** 2) / (2. * sigma ** 2))
    Amplitude_Aberration = Aberration_hight * ( 1. - np.exp( - ((x - x0) ** 2 + (y - y0) ** 2) / (2. * sigma ** 2)))

    return Amplitude_Aberration # as long as x and y, uas

# def GetSimDataSerials(SimTarStarPosMean, SimRefStarPos, n_ref_stars, times_s): 

#     length = len(times_s)
#     SimRefStarOffset = np.zeros((2 * n_ref_stars, length - 1), dtype = np.float64)
#     RepeatNum = [length - 1, 1] 
#     SimRefStarOffsetOri = np.repeat(SimRefStarPos[0: 2 * n_ref_stars, 0: 2], RepeatNum, axis=1)
#     SimRefStarOffset = SimRefStarPos[0: 2 * n_ref_stars, 1: length] - SimRefStarOffsetOri[0: 2 * n_ref_stars, 0: length - 1]

#     SimRefStarOffsetMean = np.zeros((2, length - 1), dtype = np.float64)
#     SimRefStarOffsetMean[0] = np.mean(SimRefStarOffset[0: n_ref_stars], 0)
#     SimRefStarOffsetMean[1] = np.mean(SimRefStarOffset[n_ref_stars: 2 * n_ref_stars], 0)

#     SimTarStarOffset = np.zeros((2, length), dtype = np.float64)
#     SimTarStarOffset[0:2, 0] = SimTarStarPosMean[0:2, 0]
#     SimTarStarOffset[0:2, 1: length] = SimTarStarPosMean[0:2, 1: length] - SimRefStarOffsetMean

#     return SimTarStarOffset     # 2 rows ,uas

def GenSimDataFunc(TarStarPos, RefStarPos_list, n_ref_stars, times_s, noise_mean, noise_std, noise_mean_Jitter, noise_std_Jitter, ):

    RefStarPos = GenRefStarSerials(RefStarPos_list, times_s, n_ref_stars)

    AllStarPosJ = AddJitterNoise(noise_mean_Jitter, noise_std_Jitter, TarStarPos, RefStarPos, n_ref_stars)

    #  from here : length - 1
    StarDist = GenRelativePos(AllStarPosJ, n_ref_stars)  

    StarDistDelta = GenSimRelativePos(StarDist, noise_mean, noise_std, n_ref_stars)

    StarDistIndirect = GenRelativeIndirectPos(StarDistDelta, n_ref_stars)

    return StarDistIndirect     # 2 * n_ref_stars rows ,uas
