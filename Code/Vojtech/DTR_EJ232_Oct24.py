import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from lmfit import Model
from numpy import exp, pi, sqrt
#from math import exp, erfc, erf
from matplotlib.lines import Line2D
from scipy import special
import scipy.optimize
from scipy.optimize import curve_fit
from lmfit.models import ExponentialGaussianModel
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from scipy import signal
from scipy import special
from scipy.interpolate import CubicSpline
from scipy.stats import binned_statistic
from scipy.stats import skew
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
from IPython.display import display

plt.rcParams.update({'font.size': 12})

def GausBkg(x, amp, mean, sigma,bkg):
    """
    1-d Gaussian function
    """
    return amp*exp(-(x-mean)**2/(2*(sigma**2))) + bkg

gBkgmodel = Model(GausBkg)

def Gaus(x, amp, mean, sigma):
    """
    1-d Gaussian function
    """
    return amp*exp(-(x-mean)**2/(2*(sigma**2)))

gmodel = Model(Gaus)

def sum2Gaus(x, amp, mean, sigma, amp2, mean2, sigma2):
    """
    Sum of two 1-d Gaussian functions
    """
    return amp*exp(-(x-mean)**2/(2*(sigma**2))) + amp2*exp(-(x-mean2)**2/(2*(sigma2**2)))

gmodel2 = Model(sum2Gaus)

def monoExp(x, amp, lambda1):
    return amp * np.exp(-lambda1 * x)
gmonoexpmodel = Model(monoExp)

def convExpGaus(x, amplitude, mu, sigma, lambda1):
    """
    analytical cnvolution between
    Gaus and Exponential function
    """
    expo = amplitude*lambda1*0.5*exp(-lambda1*((x-mu) - (sigma**2)*lambda1*0.5))
    # expo = lambda1*0.5*exp(-lambda1*((x-mu) - (sigma**2)*lambda1*0.5))
    errfun = special.erf((x-mu-(sigma**2)*lambda1)/(sqrt(2)*sigma))
    return expo*(1 + errfun)

gexpmodel = Model(convExpGaus) #ExponentialGaussianModel() #

def convExpGausNew(x, mu, sigma, lambda1):
    """
    analytical cnvolution between
    Gaus and Exponential function
    """
    expo = lambda1*0.5*exp(-lambda1*((x-mu) - (sigma**2)*lambda1*0.5))
    # expo = lambda1*0.5*exp(-lambda1*((x-mu) - (sigma**2)*lambda1*0.5))
    errfun = special.erf((x-mu-(sigma**2)*lambda1)/(sqrt(2)*sigma))
    return expo*(1 + errfun)

gexpmodelNew = Model(convExpGausNew) #ExponentialGaussianModel() #

def lin_interp(x, y, i, half):
    """
    Linear interpolation of two points
    returning the exact x corresponding to half,
    where x lies between x[i] and x[i+1]
    """
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    """
    Compute the FWHM of a function given its x,y values
    """
    #find the half of the maximum
    half = max(y)/2.0
    # return -1 if y-half < 0
    #         0 if y-half = 0
    #         1 if y-half > 0
    signs = np.sign(np.add(y, -half))
    # return true when y-half crosses zero
    zero_crossings = (signs[0:-2] != signs[1:-1])
    # return the corresponding x
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def tenth_max_x(x, y):
    """
    Compute the FWHM of a function given its x,y values
    """
    #find the half of the maximum
    tenth = max(y)/10.0
    # return -1 if y-half < 0
    #         0 if y-half = 0
    #         1 if y-half > 0
    signs = np.sign(np.add(y, -tenth))
    # return true when y-half crosses zero
    zero_crossings = (signs[0:-2] != signs[1:-1])
    # return the corresponding x
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], tenth),
            lin_interp(x, y, zero_crossings_i[1], tenth)]

# fig12, ax12 = plt.subplots()

sipm_size = 3 #3 or 6
data_type = 'Oct24' #
sample = 'Urethane'
#sample = 'EJ232 48'
folder = '../data/DTR_testEJ232_Oct24/'
folder2 = '../data/DTR_test_June24/'
folder3 = '../data/DTR_testGGAG_Dec24/'
folder4 = '../data/CPB_JanFeb25/'
folder5 = '../data/DTR_samples_Ildefonso_March25/'
# folder = '../data/SamplesTribikram/'


# cmap = plt.cm.get_cmap('plasma')
cmap = mpl.colormaps['plasma']
cmap1 = mpl.colormaps['viridis']
cmapred = mpl.colormaps['hot']

if (sipm_size == 3 and data_type == 'Oct24'):
    if (sample == 'compare'):
        dataName = [folder + '' + 'DATA--EJ232_48V--00000.txt',folder + '' + 'DATA--EJ232_48V_100mVdiv--00000.txt']
        legNames = ['EJ232 50 mV/div','EJ232 100 mV/div']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = 'EJ232'
        alphaIndex = [1,1]
        gate = [4.8, 4.8] # not sure about the gates!
    if (sample == 'GGAG'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_20kV_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_30kV_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt']
        legNames = ['GGAG 20 kV', 'GGAG 30 kV','GGAG 40 kV']
        # div = [4.6]
        colorIndex = [cmap(0.2),cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0,0]
        line_style = ['-','-','-']
        title = 'GGAG'
        alphaIndex = [1,1,1]
        gate = [5.8, 5.8, 5.8]
    if (sample == 'GGAGrepRate'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_RR_1kHz_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt']
        legNames = ['GGAG 1 kHz','GGAG 10 kHz']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.8, 5.8]
    if (sample == 'GGAGposition'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div_far--00000.txt']
        legNames = ['GGAG near','GGAG far']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.8, 5.8]
    if (sample == 'GGAGextraOD'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_1_3_OD_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div_long--00000.txt']
        legNames = ['GGAG extra 1.3 OD','GGAG']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.8, 5.8]
    if (sample == 'GGAGlong'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div_long--00000.txt']
        legNames = ['GGAG shorter','GGAG longer']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.8, 5.8]
    if (sample == 'GGAGvsEJ232'):
        dataName = [folder + '' + 'DATA--EJ232_47V--00000.txt', folder3 + '' + 'DATA--GGAG_47V_XraySource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt']
        legNames = ['EJ232','GGAG']
        # div = [4.6]
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [4.8, 5.8] # not sure about EJ gate though!
    if (sample == 'GGAGBa'):
        dataName = [folder3 + '' + 'DATA--GGAG_47V_XraySource_30kV_100mVdiv_5mVtrig_2to78div--00000.txt', folder3 + '' + 'DATA--GGAG_47V_BaSource_40kV_100mVdiv_5mVtrig_2to78div--00000.txt']
        legNames = ['GGAG X-ray at 30 keV','GGAG Ba source']
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.8, 5.8]
    if (sample == 'LSOposition'):
        dataName = [folder3 + '' + 'DATA--LSO_47V_XraySource_40kV_100mVdiv_5mVtrig_2to85div_near--00000.txt', folder3 + '' + 'DATA--LSO_47V_XraySource_40kV_100mVdiv_5mVtrig_2to85div_middle--00000.txt', folder3 + '' + 'DATA--LSO_47V_XraySource_40kV_100mVdiv_5mVtrig_2to85div_furthest--00000.txt']
        legNames = ['LSO near', 'LSO middle','LSO far']
        colorIndex = [cmap(0.2),cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0,0]
        line_style = ['-','-','-']
        title = 'Distance test'
        alphaIndex = [1,1,1]
        gate = [6.5, 6.5, 6.5]
    if (sample == 'SICCASpos'):
        dataName = [folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to96div_near--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to96div_middle--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to85div_far--00000.txt']
        legNames = ['GGAG SICCAS near','GGAG SICCAS middle','GGAG SICCAS far']
        colorIndex = [cmap(0.2),cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [1,1,1]
        line_style = ['-','-','-']
        title = 'Distance test'
        alphaIndex = [1,1,1]
        gate = [8.6,8.6,7.5]
    if (sample == 'SICCASposition'):
        dataName = [folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to96div_near--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to96div_middle_new--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to85div_far_one_turn_right--00000.txt']
        legNames = ['GGAG SICCAS near','GGAG SICCAS middle','GGAG SICCAS far']
        colorIndex = [cmap(0.2),cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [1,1,1]
        line_style = ['-','-','-']
        title = 'Distance test'
        alphaIndex = [1,1,1]
        gate = [8.6,8.6,7.5]
    if (sample == 'SICCASpositionNew'):
        dataName = [folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_10mVtrig_1to85div_far--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_5mVtrig_1to9div_short--00000.txt', folder3 + '' + 'DATA--SICCAS_47V_XraySource_40kV_100mVdiv_10mVtrig_1to85div_near--00000.txt']
        legNames = ['GGAG SICCAS far','GGAG SICCAS middle 1-8.5','GGAG SICCAS near']
        colorIndex = [cmap(0.2),cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [1,1,1]
        line_style = ['-','-','-']
        title = 'Distance test'
        alphaIndex = [1,1,1]
        gate = [7.5,8,7.5]
    if (sample == 'CPB'):
        dataName = [folder4 + '' + 'DATA--CPB_AHFS_25C_100mVdiv_10mVtrig_2to73div--00000.txt',folder4 + '' + 'DATA--CPB_AHFS_110C_100mVdiv_10mVtrig_2to65div_short--00000.txt']
        legNames = ['CPB AHFS 25C','CPB AHFS 110C']
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.3,4.5]
    if (sample == 'CPB2'):
        dataName = [folder4 + '' + 'DATA--CPB_AHFS_25C_100mVdiv_10mVtrig_2to73div--00000.txt']
        legNames = ['CPB AHFS 25C']
        colorIndex = [cmap(0.3)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0]
        line_style = ['-']
        title = ''
        alphaIndex = [1]
        gate = [5.3]
    if (sample == 'LSOthenNow'):
        dataName = [folder3 + '' + 'DATA--LSO_47V_XraySource_40kV_100mVdiv_5mVtrig_2to85div_near--00000.txt', folder5 + '' + 'DATA--LSO_47V_XraySource_40kV_100mVdiv_5mVtrig_2to82div--00000.txt']
        legNames = ['LSO then','LSO now']
        colorIndex = [cmap(0.5),cmap(0.8)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = 'Compatibility test'
        alphaIndex = [1,1]
        gate = [6.5, 6.2]
    if (sample == 'UrethaneVsCPB'):
        dataName = [folder4 + '' + 'DATA--CPB_AHFS_25C_100mVdiv_10mVtrig_2to73div--00000.txt',folder5 + '' + 'DATA--Ur_QD_1perc_47V_XraySource_40kV_20mVdiv_10mVtrig_2to9div--00000.txt']
        legNames = ['CPB AHFS 25C','Urethane']
        colorIndex = [cmap(0.3),cmap(0.7)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0,0]
        line_style = ['-','-']
        title = ''
        alphaIndex = [1,1]
        gate = [5.3,7]
    if (sample == 'Urethane'):
        dataName = [folder5 + '' + 'DATA--Ur_QD_1perc_47V_XraySource_40kV_20mVdiv_10mVtrig_2to9div--00000.txt']
        legNames = ['Urethane']
        colorIndex = [cmap(0.3)]
        normalization = 0 # select normalization: 0 to normalize by max, 1 to norm by run duration
        duration = [0]
        line_style = ['-']
        title = ''
        alphaIndex = [1]
        gate = [7]
    Nintervals = 8 #number of intervals for time walk correction

N = len(dataName)


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
if(N>1):
    fig5, ax5 = plt.subplots(1,N, figsize=(12,5))
    fig6, ax6 = plt.subplots(1,N, figsize=(12,5))
    fig7, ax7 = plt.subplots(1,N, figsize=(12,5))
    fig8, ax8 = plt.subplots(1,N, figsize=(12,5))
    fig9, ax9 = plt.subplots(1,N, figsize=(12,5))
    fig13, ax13 = plt.subplots(1,N, figsize=(12,5))
    fig14, ax14 = plt.subplots(1,N, figsize=(12,5))
else:
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()
    fig8, ax8 = plt.subplots()
    fig9, ax9 = plt.subplots()
    fig13, ax13 = plt.subplots()
    fig14, ax14 = plt.subplots()
fig10, ax10 = plt.subplots()
fig11, ax11 = plt.subplots()
fig12, ax12 = plt.subplots()
fig15, ax15 = plt.subplots()


DTR_histo = [0]*N
DTR_fit = [0]*N
DTR_10_histo = [0]*N
DTR_10_fit = [0]*N

DTR_bef_tw_histo = [0]*N
DTR_bef_tw_fit = [0]*N
DTR_bef_tw_10_histo = [0]*N
DTR_bef_tw_10_fit = [0]*N

SNR = [0]*N
LO = [0]*N
ILY = [0]*N
LO_amp = [0]*N
ILY_amp = [0]*N

istart = 0
istop = N


for i in range(istart,istop):
    # Here we are reading the necessary information: amp_time to additionally clean the data, remove anything above 20 mV
    #                                                charge_full to see the full charge -> 0-10 div
    #                                                amp to see the amplitude of the signal
    #                                                rise to perform the rise time correction
    #                                                dt15mV to see the delay between trigger and stop (from the scintillator)
    #                                                rms to clean the data, i.e. reject anything with rms > x
    #                                                charge to see the full signal (but still only 2-6.4 div to not account for the overshoot)
    #                                                mean_ped is the mean charge before the signal comes (pedestal) -> we can use it to subtract it from the measured charge and amplitude

    df = pd.read_csv(dataName[i], sep="\t", index_col=False, names=["amp_time","charge_full", "amp", "rise", "dt15mV", "rms", "charge", "mean_ped"])


    print('\n\nAnalysing file:' + dataName[i] + '\n')
    # conversion from negative to positive values, easier to interpret
    df.loc[:,'amp_time'] *= -1.
    df.loc[:,'amp'] *= -1.
    df.loc[:,'charge'] *= -1.
    df.loc[:,'charge_full'] *= -1.
    df.loc[:,'mean_ped'] *= -1.



    #conversions
    df.loc[:,'amp_time'] *= 1e3 #mV
    df.loc[:,'amp'] *= 1e3 #mV
    df.loc[:,'charge'] *= 1e9 # nWb
    df.loc[:,'charge_full'] *= 1e9 # nWb
    df.loc[:,'dt15mV'] *= 1e9 #picoseconds
    df.loc[:,'rise'] *= 1e12 #picoseconds
    df.loc[:,'rms'] *= 1e3 #mV
    df.loc[:,'mean_ped'] *= 1e3 #mV
    #df.loc[]

    # print('Before subtraction:\n ')
    # print(df)

    df.loc[:,'charge'] -= df.loc[:,'mean_ped']*gate[i]*0.02 # subtract the offset in charge, gate_width*0.02 because we have x divisions of 20 ns

    df.loc[:,'amp'] -= df.loc[:,'mean_ped'] #subtract the offset in amplitude
    #df.loc[:,'charge'] -= df.loc[:,'mean_ped']*0.096 # subtract the offset in charge, x*0.2 because we have x divisions of 20 ns


    # print('After subtraction:\n ')
    # print(df)


    print('Number of events before selections: ' + str(len(df)) + '\n')
    #selecting events with reasonable rise time
    df = df.loc[df['rise']>0]
    df = df.loc[df['rise']<2000]
    print('\nNumber of events after rise time selections: ' + str(len(df)) + '\n')
    df = df.loc[df['rms']>0.5]
    df = df.loc[df['rms']<5]
    print('\nNumber of events after rms selections: ' + str(len(df)) + '\n')
    df = df.loc[df['dt15mV']>-1.1]
    print('\nNumber of events after time delay selections: ' + str(len(df)) + '\n')
    df = df.loc[df['amp_time']<10]
    print('\nNumber of events after amplitude time selections: ' + str(len(df)) + '\n')

    # print(df) # dont be confused that it prints all the way up to max number of events,
    # the numbering stays the same but the data is selected out

    ####################################################
    #################### ENERGY ########################
    ####################################################
    #### CHARGE ####

    xmin = -1
    xmax = 250
    binwidth = 0.05
    nbin = int ((xmax -xmin)/binwidth)
    binsCh = np.linspace(xmin,xmax,nbin+1)
    bin_heights_ch, bin_lefts_ch = np.histogram(df['charge'][:], bins=binsCh)
    if(normalization == 0):
        bin_heights_ch = bin_heights_ch/max(bin_heights_ch)
    elif(normalization == 1):
        bin_heights_ch = bin_heights_ch/duration[i]
    else:
        print('Wrong normalization settings!')
        exit()

    charge_range = 1
    max_charge = bin_heights_ch.max() # find max charge
    rangex_ch_max_ar = np.where(bin_heights_ch == max_charge) # find bin with max charge
    if (len(rangex_ch_max_ar[0]) != 1):
        rangex_ch_max = rangex_ch_max_ar[0][0] + xmin/binwidth
    else:
        rangex_ch_max = rangex_ch_max_ar[0] + xmin/binwidth # correct for min bin, which is not 0
    left_bar = int(rangex_ch_max-charge_range/binwidth)
    right_bar = int(rangex_ch_max+charge_range/binwidth)
    rangex_ch = bin_lefts_ch[(bin_lefts_ch>=left_bar*binwidth) & (bin_lefts_ch<right_bar*binwidth)] # create x range for charge fit -> charge_range nWb from max charge left and right
    rangey_ch = bin_heights_ch[(bin_lefts_ch[:-1]>=left_bar*binwidth) & (bin_lefts_ch[:-1]<right_bar*binwidth)] # create y range for fit


    params_ch = gBkgmodel.make_params(amp=max_charge, mean=rangex_ch_max*binwidth, sigma=1, bkg = 0)
    result_ch = gBkgmodel.fit(rangey_ch, params_ch, x=rangex_ch)

    # results of fit parameter to get the x corresponding to the peak
    res_param1_ch = result_ch.params['amp'].value
    res_param2_ch = result_ch.params['mean'].value
    res_param3_ch = result_ch.params['sigma'].value
    res_param3_ch = result_ch.params['bkg'].value
    LO[i] = res_param2_ch

    print('Mean charge for ' + legNames[i] + ' is: ' + str(res_param2_ch) + 'nWb \n')

    #max_charge_pos = rangex[np.where(rangey == amp)]
    ax1.step(bin_lefts_ch[:-1], bin_heights_ch, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
    ax1.plot(rangex_ch, result_ch.best_fit, 'r-', label='best fit')
    ax1.title.set_text(title)
    ax1.legend(fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xlabel("Charge (nWb)")
    if(normalization == 0):
        ax1.set_ylabel("Normalized Events")
    elif(normalization == 1):
        ax1.set_ylabel("Normalized Events per min")

    # xmin2 = -5
    # xmax2 = 100
    # binwidth2 = 0.1
    # nbin2 = int ((xmax2 -xmin2)/binwidth2)
    # binsCh2 = np.linspace(xmin2,xmax2,nbin2+1)
    # bin_heights_ch2, bin_lefts_ch2 = np.histogram(df['charge2'][:], bins=binsCh2)
    # bin_heights_ch2 = bin_heights_ch2/max(bin_heights_ch2)
    #
    # ax12.step(bin_lefts_ch2[:-1], bin_heights_ch2, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
    # ax12.set_yscale('log')
    # ax12.set_xlabel("Charge (nWb)")
    # ax12.set_ylabel("Normalized Events")

    ##### FULL CHARGE ######

    #### CHARGE ####

    xmin_full = -10
    xmax_full = 40
    binwidth_full = 0.05
    nbin_full = int ((xmax_full -xmin_full)/binwidth_full)
    binsCh_full = np.linspace(xmin_full,xmax_full,nbin_full+1)
    bin_heights_ch_full, bin_lefts_ch_full = np.histogram(df['charge_full'][:], bins=binsCh_full)
    if(normalization == 0):
        bin_heights_ch_full = bin_heights_ch_full/max(bin_heights_ch_full)
    elif(normalization == 1):
        bin_heights_ch_full = bin_heights_ch_full/duration[i]
    else:
        print('Wrong normalization settings!')
        exit()

    # charge_range_full = 8
    # max_charge = bin_heights_ch.max() # find max charge
    # rangex_ch_max_ar = np.where(bin_heights_ch == max_charge) # find bin with max charge
    # if (len(rangex_ch_max_ar[0]) != 1):
    #     rangex_ch_max = rangex_ch_max_ar[0][0] + xmin/binwidth
    # else:
    #     rangex_ch_max = rangex_ch_max_ar[0] + xmin/binwidth # correct for min bin, which is not 0
    # left_bar = int(rangex_ch_max-charge_range/binwidth)
    # right_bar = int(rangex_ch_max+charge_range/binwidth)
    # rangex_ch = bin_lefts_ch[(bin_lefts_ch>=left_bar*binwidth) & (bin_lefts_ch<right_bar*binwidth)] # create x range for charge fit -> charge_range nWb from max charge left and right
    # rangey_ch = bin_heights_ch[(bin_lefts_ch[:-1]>=left_bar*binwidth) & (bin_lefts_ch[:-1]<right_bar*binwidth)] # create y range for fit
    #
    #
    # params_ch = gmodel.make_params(amp=max_charge, mean=rangex_ch_max*binwidth, sigma=1, bkg = 0)
    # result_ch = gmodel.fit(rangey_ch, params_ch, x=rangex_ch)
    #
    # # results of fit parameter to get the x corresponding to the peak
    # res_param1_ch = result_ch.params['amp'].value
    # res_param2_ch = result_ch.params['mean'].value
    # res_param3_ch = result_ch.params['sigma'].value
    # res_param3_ch = result_ch.params['bkg'].value
    # LO[i] = res_param2_ch
    #
    # print('Mean charge for ' + legNames[i] + ' is: ' + str(res_param2_ch) + 'nWb \n')
    #

    #max_charge_pos = rangex[np.where(rangey == amp)]
    ax15.step(bin_lefts_ch_full[:-1], bin_heights_ch_full, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
    # ax1.plot(rangex_ch, result_ch.best_fit, 'r-', label='best fit')
    ax15.title.set_text(title)
    ax15.legend(fontsize=14)
    ax15.set_yscale('log')
    ax15.set_xlabel("Full Charge (nWb)")
    if(normalization == 0):
        ax15.set_ylabel("Normalized Events")
    elif(normalization == 1):
        ax15.set_ylabel("Normalized Events per min")


    #### AMPLITUDE ####

    xmin_amp = 0
    xmax_amp = 900
    binwidth_amp = 1
    nbin_amp = int ((xmax_amp-xmin_amp)/binwidth_amp)
    bins_amp = np.linspace(xmin_amp,xmax_amp,nbin_amp+1)
    bin_heights_amp, bin_lefts_amp = np.histogram(df['amp'][:], bins=bins_amp)
    if(normalization == 0):
        bin_heights_amp = bin_heights_amp/max(bin_heights_amp)
    elif(normalization == 1):
        bin_heights_amp = bin_heights_amp/duration[i]


    # amp_range = 0.15
    # max_amp = bin_heights_amp.max() # find max amplitude
    # rangex_amp_max_ar = np.where(bin_heights_amp == max_amp) # find array of bin with max charge
    # rangex_amp_max = rangex_amp_max_ar[0] + xmin_amp/binwidth_amp # correct for min bin, which is not 0
    # left_bar_amp = int(rangex_amp_max-amp_range/binwidth_amp)
    # right_bar_amp = int(rangex_amp_max+amp_range/binwidth_amp)
    # rangex_amp = bin_lefts_amp[(bin_lefts_amp>=left_bar_amp*binwidth_amp) & (bin_lefts_amp<right_bar_amp*binwidth_amp)] # create x range for charge fit -> charge_range nWb from max charge left and right
    # rangey_amp = bin_heights_amp[(bin_lefts_amp[:-1]>=left_bar_amp*binwidth_amp) & (bin_lefts_amp[:-1]<right_bar_amp*binwidth_amp)] # create y range for fit
    #
    #
    # params_amp = gmodel.make_params(amp=max_amp, mean=rangex_amp_max*binwidth_amp, sigma=0.1, bkg = 0)
    # result_amp = gmodel.fit(rangey_amp, params_amp, x=rangex_amp)
    #
    # # results of fit parameter to get the x corresponding to the peak
    # res_param1_amp = result_amp.params['amp'].value
    # res_param2_amp = result_amp.params['mean'].value
    # res_param3_amp = result_amp.params['sigma'].value
    # res_param3_amp = result_amp.params['bkg'].value
    # LO_amp[i] = res_param2_amp
    #
    # print('Mean amp for ' + legNames[i] + ' is: ' + str(res_param2_amp) + 'V \n')


    #max_charge_pos = rangex[np.where(rangey == amp)]
    ax2.step(bin_lefts_amp[:-1], bin_heights_amp, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
    # ax2.plot(rangex_amp, result_amp.best_fit, 'r-', label='best fit')
    ax2.set_yscale('log')
    ax2.set_xlabel("Amplitude (mV)")
    if(normalization == 0):
        ax2.set_ylabel("Normalized Events")
    elif(normalization == 1):
        ax2.set_ylabel("Normalized Events per min")

    #### Correlation between ####
    ### amplitude and charge ###

    xmin_ch_hist2d = -1.1
    xmax_ch_hist2d = 40
    binwidth_ch_hist2d = 0.10
    nbins_ch_hist2d = int ((xmax_ch_hist2d-xmin_ch_hist2d)/binwidth_ch_hist2d)
    bins_ch_hist2d = np.linspace(xmin_ch_hist2d,xmax_ch_hist2d,nbins_ch_hist2d+1)
    xmin_amp_hist2d = 0
    xmax_amp_hist2d = 700
    binwidth_amp_hist2d = 1
    nbins_amp_hist2d = int ((xmax_amp_hist2d-xmin_amp_hist2d)/binwidth_amp_hist2d)
    bins_amp_hist2d = np.linspace(xmin_amp_hist2d,xmax_amp_hist2d,nbins_amp_hist2d+1)
    if(N>1):
        ax13[i].hist2d( df['charge'][:], df['amp'][:], [bins_ch_hist2d, bins_amp_hist2d], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
        ax13[i].set_title(legNames[i])
    else:
        ax13.hist2d( df['charge'][:], df['amp'][:], [bins_ch_hist2d, bins_amp_hist2d], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
        ax13.set_title(legNames[i])
    fig13.text(0.5, 0.04, 'Integrated Charge (nWb)', ha='center', va='center')
    fig13.text(0.06, 0.5, 'Amplitude (mV)', ha='center', va='center', rotation='vertical')


    ####################################################
    ##################### TIME #########################
    ####################################################

    if (i != N): #we are interested in the only SiPM measurment just for the charge, to see how many more triggered SPAD we have with the sample compared to no sample, we know that timing is given by the sample. If the time delay peak of the sample looks the same to the only SiPM, it means that we are limited by our system and we cannot measure better timing

    #### TIME DELAY ####
        xmin_dt = -1
        xmax_dt = 5
        binwidth_dt = 0.01
        nbin_dt = int ((xmax_dt-xmin_dt)/binwidth_dt)
        bins_dt = np.linspace(xmin_dt,xmax_dt,nbin_dt+1)
        #bincenters_dt = np.array([0.5*(bins_dt[i]+bins_dt[i+1]) for i in range(len(bins_dt)-1)])
        bin_heights_dt, bin_lefts_dt = np.histogram(df['dt15mV'][:], bins=bins_dt)
        bin_heights_filtered = bin_heights_dt[ (bin_lefts_dt[:-1]>xmin_dt) & (bin_lefts_dt[:-1]<xmin_dt+1)] # getting entries in a 1 ns window before the pulse
        baseline = bin_heights_filtered.mean() # calculating the mean value on the noise -> pedestal to subtract it from the whole distribution
        SNR[i] = max(bin_heights_dt)/baseline # Vojtěch: does it make sense to compare max value and average?
        bin_heights_dt = bin_heights_dt - baseline # subtraction of the pedestal
        bin_heights_dt = bin_heights_dt/max(bin_heights_dt) # normalization

        max_count = np.max(bin_heights_dt)
        bin_max_count = bin_lefts_dt[np.where(bin_heights_dt == max_count)][0]


        mask_gaus = bin_lefts_dt[:-1]<bin_max_count
        mask_monoExp = (bin_lefts_dt[:-1]>bin_max_count+0.0025) & (bin_lefts_dt[:-1]<bin_max_count+1.0)



        # bin_heights_Gaus_dt, bin_lefts_Gaus_dt = np.histogram(dfGausSelec['dt15mV'][:], bins=bins_dt)
        # bin_heights_Gaus_dt = bin_heights_Gaus_dt/max(bin_heights_Gaus_dt)
        # bin_heights_MonoExp_dt, bin_lefts_MonoExp_dt = np.histogram(dfMonoExpSelec['dt15mV'][:], bins=bins_dt)
        # bin_heights_MonoExp_dt = bin_heights_MonoExp_dt/max(bin_heights_MonoExp_dt)
        #
        #
        initial_guess_gaus = [max_count, bin_max_count, 0.1]
        initial_guess_monoExp = [max_count, 1]

        # print(bin_heights_dt[mask_gaus])
        # print(len(bin_heights_dt[:-1][mask_gaus]))
        # print(bin_lefts_dt[mask_gaus])
        # print(len(bin_lefts_dt[mask_gaus]))

        fit_gaus_params, pcov_gaus = curve_fit(Gaus, bin_lefts_dt[:-1][mask_gaus], bin_heights_dt[mask_gaus], p0 = initial_guess_gaus, bounds=([-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_monoExp_params, pcov_monoExp = curve_fit(monoExp, bin_lefts_dt[:-1][mask_monoExp], bin_heights_dt[mask_monoExp], p0 = initial_guess_monoExp, bounds=([-np.inf,-np.inf], [np.inf,np.inf])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_gaus_curve = Gaus(bin_lefts_dt[:-1][mask_gaus], *fit_gaus_params)

        fit_monoExp_curve = monoExp(bin_lefts_dt[:-1][mask_monoExp], *fit_monoExp_params)


        initial_guess_whole = [max_count,fit_gaus_params[1],fit_gaus_params[2],fit_monoExp_params[1]]


        fit_whole_params, pcov_whole = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess_whole, bounds=([-np.inf,-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf])) #could
        # fit_whole_params, pcov_whole = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess_whole, bounds=([-np.inf,fit_gaus_params[1]-math.copysign(1,fit_gaus_params[1])*0.5*fit_gaus_params[1],fit_gaus_params[2]-0.1*fit_gaus_params[2],fit_monoExp_params[1]-0.1*fit_monoExp_params[1]], [np.inf,fit_gaus_params[1]+math.copysign(1,fit_gaus_params[1])*0.5*fit_gaus_params[1],fit_gaus_params[2]+0.1*fit_gaus_params[2],fit_monoExp_params[1]+0.1*fit_monoExp_params[1]])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_whole_curve = convExpGaus(bin_lefts_dt[:-1], *fit_whole_params)

        selected_sample=-1 # -1 shows all plots, other selection shows only 1 selected plot



        if(selected_sample==-1):
            ax3.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
            ax3.plot(bin_lefts_dt[:-1], fit_whole_curve, color=cmapred(0.35))
            ax3.set_xlabel("Time Delay (ns)")
            ax3.title.set_text(title)
            ax3.set_ylabel("Normalized Events")
        elif(i==selected_sample):
            ax3.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
            ax3.plot(bin_lefts_dt[:-1], fit_whole_curve, color=cmapred(0.35))
            ax3.set_xlabel("Time Delay (ns)")
            ax3.title.set_text(title)
            ax3.set_ylabel("Normalized Events")
        x_bef_tw = bin_lefts_dt
        y_bef_tw_histo = bin_heights_dt # calculates FWHM from the histogram width
        y_bef_tw_fit = fit_whole_curve # calculates FWHM from fit

    ############################# NOTE! ##################################
    #### The fit function is not modelling very well the distribution ####
    ### You can struggle with the initialization parameters of the fit ###
    ### OR decide to evaluate the FWHM and FWTM directly from the histo ##
    #################### ( like we are doing now ) #######################

        hmx_bef_tw_histo = half_max_x(x_bef_tw,y_bef_tw_histo)
        fwhm_bef_tw_histo = hmx_bef_tw_histo[1] - hmx_bef_tw_histo[0]
        half_bef_tw_histo = max(y_bef_tw_histo)/2.0
        hmx_bef_tw_fit = half_max_x(x_bef_tw,y_bef_tw_fit)
        fwhm_bef_tw_fit = hmx_bef_tw_fit[1] - hmx_bef_tw_fit[0]
        half_bef_tw_fit = max(y_bef_tw_fit)/2.0
        if(selected_sample==-1):
            ax3.plot(hmx_bef_tw_fit, [half_bef_tw_fit, half_bef_tw_fit], color ='orange')
            ax3.legend(loc='best')
        elif(i==selected_sample):
            ax3.plot(hmx_bef_tw_fit, [half_bef_tw_fit, half_bef_tw_fit], color ='orange')
            ax3.legend(loc='best')
        DTR_bef_tw_histo[i] = fwhm_bef_tw_histo*1000 #DTR in ps
        DTR_bef_tw_fit[i] = fwhm_bef_tw_fit*1000 #DTR in ps


        # tmx_bef_tw_histo = tenth_max_x(x_bef_tw,y_bef_tw_histo)
        # fwtm_bef_tw_histo = tmx_bef_tw_histo[1] - tmx_bef_tw_histo[0]
        # tenth_bef_tw_histo = max(y_bef_tw_histo)/10.0
        # tmx_bef_tw_fit = tenth_max_x(x_bef_tw,y_bef_tw_fit)
        # fwtm_bef_tw_fit = tmx_bef_tw_fit[1] - tmx_bef_tw_fit[0]
        # tenth_bef_tw_fit = max(y_bef_tw_fit)/10.0
        # if(selected_sample==-1):
        #     ax3.plot(tmx_bef_tw_fit, [tenth_bef_tw_fit, tenth_bef_tw_fit], color ='orange')
        # elif(i==selected_sample):
        #     ax3.plot(tmx_bef_tw_fit, [tenth_bef_tw_fit, tenth_bef_tw_fit], color ='orange')
        # DTR_bef_tw_10_histo[i] = fwtm_bef_tw_histo*1000
        # DTR_bef_tw_10_fit[i] = fwtm_bef_tw_fit*1000



    #### RISE-TIME ####
        xmin_rt = 100
        xmax_rt = 1200
        binwidth_rt = 1
        nbin_rt = int ((xmax_rt-xmin_rt)/binwidth_rt)
        bins_rt = np.linspace(xmin_rt,xmax_rt,nbin_rt+1)
        bincenters_rt = np.array([0.5*(bins_rt[i]+bins_rt[i+1]) for i in range(len(bins_rt)-1)])
        bin_heights_rt, bin_lefts_rt = np.histogram(df['rise'][:], bins=bins_rt)
        bin_heights_rt = bin_heights_rt/max(bin_heights_rt)

        ax4.step(bin_lefts_rt[:-1], bin_heights_rt, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
        ax4.set_xlabel("Rise-Time [ps]")
        ax4.set_ylabel("Normalized Events")


    #### Correlation between ####
    ### time delay and charge ###
        xmin_dt1 = 0
        xmax_dt1 = 1.5
        binwidth_dt1 = 0.05
        nbin_dt1 = int ((xmax_dt1-xmin_dt1)/binwidth_dt1)
        bins_dt1 = np.linspace(xmin_dt1,xmax_dt1,nbin_dt1+1)
        xmin1 = -5
        xmax1 = 40
        binwidth1 = 0.1
        nbin1 = int ((xmax1-xmin1)/binwidth1)
        binsCh1 = np.linspace(xmin1,xmax1,nbin1+1)
        if(N>1):
            ax5[i].hist2d( df['charge'][:], df['dt15mV'][:], [binsCh1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax5[i].set_title(legNames[i])
        else:
            ax5.hist2d( df['charge'][:], df['dt15mV'][:], [binsCh1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax5.set_title(legNames[i])
        fig5.text(0.5, 0.04, 'Integrated Charge (nWb)', ha='center', va='center')
        fig5.text(0.06, 0.5, 'Time Delay (ns)', ha='center', va='center', rotation='vertical')

    #### Correlation between ####
    ## time delay and risetime ##
    #### BEFORE TW correction ###
        xmin_rt1 = 200
        xmax_rt1 = 800
        binwidth_rt1 = 5
        nbin_rt1 = int ((xmax_rt1-xmin_rt1)/binwidth_rt1)
        bins_rt1 = np.linspace(xmin_rt1,xmax_rt1,nbin_rt1+1)
        if(N>1):
            ax6[i].hist2d(df['rise'][:], df['dt15mV'][:], [bins_rt1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax6[i].set_title(legNames[i])
        else:
            ax6.hist2d(df['rise'][:], df['dt15mV'][:], [bins_rt1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax6.set_title(legNames[i])
        fig6.text(0.5, 0.04, 'Signal Rise Time [ps]', ha='center', va='center')
        fig6.text(0.06, 0.5, 'Time Delay (ns)', ha='center', va='center', rotation='vertical')

    #### Correlation between ####
    ### rise time and charge ###
        if(N>1):
            ax14[i].hist2d( df['charge'][:], df['rise'][:], [binsCh1, bins_rt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax14[i].set_title(legNames[i])
        else:
            ax14.hist2d( df['charge'][:], df['rise'][:], [binsCh1, bins_rt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax14.set_title(legNames[i])
        fig14.text(0.5, 0.04, 'Integrated Charge (nWb)', ha='center', va='center')
        fig14.text(0.06, 0.5, 'Signal Rise Time [ps]', ha='center', va='center', rotation='vertical')


        ####################################################
        #################### RMS ########################
        ####################################################
        xmin_rms = 0
        xmax_rms = 5
        binwidth_rms = 0.1
        nbin_rms = int ((xmax_rms-xmin_rms)/binwidth_rms)
        bins_rms = np.linspace(xmin_rms,xmax_rms,nbin_rms+1)
        bin_heights_rms, bin_lefts_rms = np.histogram(df['rms'][:], bins=bins_rms)
        bin_heights_rms = bin_heights_rms/max(bin_heights_rms)

        ax11.step(bin_lefts_rms[:-1], bin_heights_rms, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
        ax11.set_yscale('log')
        ax11.set_xlabel("RMS")
        ax11.set_ylabel("Normalized Events")

        ################################################
        #### TIME WALK CORRECTION BASED ON RISETIME ####
        ################################################
        totEntries = len(df.loc[(df['rise']>xmin_rt) & (df['rise']<xmax_rt)])
        #totEntries = len(df.loc[df['rise']<xmax_rt])
        nEntries_rt, nbins_rt = np.histogram(df['rise'][:], bins=bins_rt)
        print('!!!Check TW correction: \n tot events: ' + str(totEntries) + '\n ideal number of events per interval: ' + str(totEntries/Nintervals))
        colorIndex2 = np.linspace(0,1,Nintervals+1)
        #binindex = nbin_rt - 1
        binindex = 0
        binstart = {}
        bincenterstart = {}
        binend = {}
        bincenterend = {}
        binstart[0] = 0

        tw_factor = [0]*Nintervals
        mean_rise = [0]*Nintervals

        for j in range (0,Nintervals):
            Nevents = 0
            while Nevents < totEntries/Nintervals:
                if binindex == nbin_rt-1:
                    break
                Nevents += nEntries_rt[binindex]
                binindex += 1
            print(' events selected in interval ' + str(j) + ': ' + str(Nevents) )
            binend[j] = binindex
            bincenterend[j] = bincenters_rt[binend[j]-1]
            if (j == 0):
                bincenterstart[j] = bincenters_rt[0]
            else:
                bincenterstart[j] = bincenterend[j-1]
            dfRiseSelec = df.loc[df['rise']>bincenterstart[j]]
            dfRiseSelec = dfRiseSelec.loc[dfRiseSelec['rise']<bincenterend[j]]
            #dfRiseSelec = dfRiseSelec/max(dfRiseSelec)
            if(N>1):
                ax7[i].hist(dfRiseSelec['rise'][:], bins=bins_rt, histtype='step', color= cmap(colorIndex2[j]))
                ax7[i].set_title(legNames[i])
            else:
                ax7.hist(dfRiseSelec['rise'][:], bins=bins_rt, histtype='step', color= cmap(colorIndex2[j]))
                ax7.set_title(legNames[i])
            fig7.text(0.5, 0.04, 'Signal Rise Time [ps]', ha='center', va='center')
            fig7.text(0.06, 0.5, 'Events', ha='center', va='center', rotation='vertical')
            fig7.text(0.5, 0.97, 'Time Walk Correction pt1', ha='center', va='center')


            #definition of new dt histograms for each rise-time interval for TW correction
            xmin_dt_tw = -1
            xmax_dt_tw = 2
            binwidthdt = 0.01
            nbin_dt = int ((xmax_dt_tw-xmin_dt_tw)/binwidthdt)
            bins_dt = np.linspace(xmin_dt_tw,xmax_dt_tw,nbin_dt+1)
            bin_heights_dt, bin_lefts_dt = np.histogram(dfRiseSelec['dt15mV'][:], bins=bins_dt)
            bin_heights_filtered = bin_heights_dt[ (bin_lefts_dt[:-1]>xmin_dt_tw) & (bin_lefts_dt[:-1]<xmin_dt_tw+0.5)] # getting entries in a 1 ns window before the pulse
            baseline = bin_heights_filtered.mean() # calculating the mean value on the noise -> pedestal to subtract it from the whole distribution
            SNR[i] = max(bin_heights_dt)/baseline # Vojtěch: does it make sense to compare max value and average?
            bin_heights_dt = bin_heights_dt - baseline # subtraction of the pedestal
            #bin_heights_dt = bin_heights_dt/max(bin_heights_dt) # normalization




                # defining range fit


            ################# NOTE! ##################
            #### If you have problem with the fit ####
            ### You can try to directly use 'mean' ###
            ############ as 'tw_factor' ##############


            max_count = np.max(bin_heights_dt)
            bin_max_count = bin_lefts_dt[np.where(bin_heights_dt == max_count)][0]
            #
            # s = np.std(dfRiseSelec['dt15mV'][:])
            # m = statistics.mean(dfRiseSelec['dt15mV'][:])
            # med = statistics.median(dfRiseSelec['dt15mV'][:])
            #
            # gamma = 3*(m-med)/s
            #
            # #dfRiseSelec['dt15mV'][:].to_excel('out' + str(j) + '.xlsx')
            # print('std = ' + str(s))
            # print('mean = ' + str(m))
            # print('median = ' + str(med))
            # print('skewness = ' + str(gamma))
            #
            #
            #
            # in_mu = m-s*pow((gamma/2),1/3)
            # in_sigma = pow(s,2)*(1-pow((gamma/2),2/3))
            # in_tau = s*pow((gamma/2),1/3)
            #
            # print('mu = ' + str(in_mu))
            # print('sigma^2 = ' + str(in_sigma))
            # print('tau = ' + str(in_tau))
            # initial_guess = [in_mu, sqrt(in_sigma), 10/in_tau]
            # print(initial_guess)
            # fit_whole_params, pcov_sum = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess, bounds=([in_mu-0.2,0.11,0], [in_mu+0.2,1,np.inf]))

            initial_guess = [max_count, bin_max_count, 0.05, 8]
            print(initial_guess)
            fit_whole_params, pcov_sum = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess, bounds=([0,xmin_dt,0,-np.inf], [np.inf,xmax_dt,np.inf,np.inf])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error


            fit_whole_curve = convExpGaus(bin_lefts_dt[:-1], *fit_whole_params)
            mean_fit = bin_lefts_dt[np.argmax(fit_whole_curve)]



            j_selected = -1 # -1 selects all intervals, the rest you can choose
            if((N>1) and (j_selected==-1)):
                ax8[i].step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=cmap(colorIndex2[j]),  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
                ax8[i].plot(bin_lefts_dt[:-1],fit_whole_curve, label='fit', color=cmapred(0.35))
                ax8[i].set_title(legNames[i])
            elif((N>1) and (j==j_selected)):
                ax8[i].step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color= cmap(colorIndex2[j]),  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
                ax8[i].plot(bin_lefts_dt[:-1],fit_whole_curve, label='fit')
                ax8[i].set_title(legNames[i])
            elif((N==1) and (j_selected==-1)):
                ax8.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=cmap(colorIndex2[j]),  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
                ax8.plot(bin_lefts_dt[:-1],fit_whole_curve, label='fit', color=cmapred(0.35))
                ax8.set_title(legNames[i])
            elif((N==1) and (j==j_selected)):
                ax8.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=cmap(colorIndex2[j]),  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
                ax8.plot(bin_lefts_dt[:-1],fit_whole_curve, label='fit', color=cmapred(0.35))
                ax8.set_title(legNames[i])
            elif(j_selected>=Nintervals):
                print('Selected interval is not available!')
                exit()

            fig8.text(0.5, 0.04, 'Time Delay (ns)', ha='center', va='center')
            fig8.text(0.06, 0.5, 'Events', ha='center', va='center', rotation='vertical')
            fig8.text(0.5, 0.97, 'Time Walk Correction pt2', ha='center', va='center')

            i_singleplot = -1
            if(i_singleplot==-1):
                fig12.clf()
                plt.close(fig12)
            elif (i==i_singleplot): # makes a separate plot of the different fits
                ax12.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=cmap(colorIndex2[j]),  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
                ax12.plot(bin_lefts_dt[:-1],fit_whole_curve, label='fit', color=cmapred(0.35))
                fig12.text(0.5, 0.04, 'Time Delay (ns)', ha='center', va='center')
                fig12.text(0.06, 0.5, 'Events', ha='center', va='center', rotation='vertical')
                fig12.text(0.5, 0.95, 'TW Corr. for '+str(legNames[i]), ha='center', va='center')

            tw_factor[j] = mean_fit
            print(tuple(fit_whole_params))
            # tw_factor[j] = res_param2 #scipy.optimize.fmin(lambda x: -convExpGaus(x, res_param1, res_param2, res_param3, res_param4, res_param5), 0, disp=False)
            # using 'scipy.optimize' etc is helpful if you are using a sum of 2Gaus and you have to take the right center

            mean_rise[j] = dfRiseSelec['rise'][:].mean()
            print('=== Factors for interval ' + str(j) + ': mean rise =  ' + str(mean_rise[j]) + ', timewalk factor = ' + str(tw_factor[j]) + '\n')


        mean_rise.insert(0, mean_rise[0]-50) # adding more points to mean rise time
        mean_rise.append(mean_rise[Nintervals]+50) # adding more points to mean rise time

        tw_factorA = np.array(tw_factor)
        #tw_factorA = tw_factorA[:,0]
        tw_factorL = tw_factorA.tolist()
        tw_factorL.insert(0,tw_factorL[0]) # adding the same tw factor as for the first interval to the first inserted mean rise point
        tw_factorL.append(tw_factorL[Nintervals]) # adding the same tw factor as for the last interval to the last inserted mean rise point
        if(N>1):
            ax6[i].scatter(mean_rise, tw_factorL, color = 'red', s = 5) # plotting points on top of the plot
        else:
            ax6.scatter(mean_rise, tw_factorL, color = 'red', s = 5) # plotting points on top of the plot

        spl = CubicSpline(mean_rise, tw_factorL) # calculating cubic spline
        xnew = np.linspace(mean_rise[0], mean_rise[Nintervals+1], num=51)
        if(N>1):
            ax6[i].plot(xnew, spl(xnew), color = 'red') # plotting cubic spline
        else:
            ax6.plot(xnew, spl(xnew), color = 'red') # plotting cubic spline

        #print(df)

        # --> printing values before and after doing some operation on the data
        # can help to debug if you see something strange

        correction = spl(df['rise'][:])
        tocorrect = df['dt15mV'][:]
        correction = np.array(correction)
        df['dt15mV'] = tocorrect - correction

        #print(df)

        #### Correlation between ####
        ## time delay and risetime ##
        #### AFTER TW correction ###

        xmindt1 = -0.3
        xmaxdt1 = 1
        binwidthdt1 = 0.05
        nbin_dt1 = int ((xmaxdt1-xmindt1)/binwidthdt1)
        bins_dt1 = np.linspace(xmindt1,xmaxdt1,nbin_dt1+1)

        if(N>1):
            ax9[i].hist2d(df['rise'][:], df['dt15mV'][:], [bins_rt1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax9[i].set_title(legNames[i])
        else:
            ax9.hist2d(df['rise'][:], df['dt15mV'][:], [bins_rt1, bins_dt1], norm=mpl.colors.LogNorm(), cmap= 'viridis', cmin = 0.9)
            ax9.set_title(legNames[i])
        fig9.text(0.5, 0.04, 'Signal Rise Time [ps]', ha='center', va='center')
        fig9.text(0.06, 0.5, 'Time Delay (ns)', ha='center', va='center', rotation='vertical')
        fig9.text(0.5, 0.97, 'After Time Walk Correction', ha='center', va='center')

        #### TIME DELAY ####
        #### AFTER TW correction ###
        xmindt = -1.5
        xmaxdt = 1.5
        binwidthdt = 0.005
        nbin_dt = int ((xmaxdt-xmindt)/binwidthdt)
        bins_dt = np.linspace(xmindt,xmaxdt,nbin_dt+1)

        #bincenters_dt = np.array([0.5*(bins_dt[i]+bins_dt[i+1]) for i in range(len(bins_dt)-1)])
        bin_heights_dt, bin_lefts_dt = np.histogram(df['dt15mV'][:], bins=bins_dt)
        bin_heights_filtered = bin_heights_dt[(bin_lefts_dt[:-1]>=xmindt) & (bin_lefts_dt[:-1]<xmindt+1)] # getting entries in a 1 ns window before the pulse
        #bin_heights_filtered = bin_heights_dt[ (bincenters_dt>-1) & (bincenters_dt<0)]
        baseline = bin_heights_filtered.mean() # calculating the mean value on the noise -> pedestal to subtract it from the whole distribution
        bin_heights_dt = bin_heights_dt - baseline # pedestal subtraction
        bin_heights_dt = bin_heights_dt/max(bin_heights_dt) # normalization

        nbin_dt = int ((xmaxdt-xmindt+1)/binwidthdt)
        bins_dt = np.linspace(xmindt+1,xmaxdt,nbin_dt+1)
        bin_heights_dt = bin_heights_dt[bin_lefts_dt[:-1]>xmindt+1]
        bin_lefts_dt = bin_lefts_dt[bin_lefts_dt>xmindt+1]

        max_count = np.max(bin_heights_dt)
        bin_max_count = bin_lefts_dt[np.where(bin_heights_dt == max_count)][0]


        mask_gaus = bin_lefts_dt[:-1]<bin_max_count
        mask_monoExp = (bin_lefts_dt[:-1]>bin_max_count+0.0025) & (bin_lefts_dt[:-1]<bin_max_count+1.0)



        # bin_heights_Gaus_dt, bin_lefts_Gaus_dt = np.histogram(dfGausSelec['dt15mV'][:], bins=bins_dt)
        # bin_heights_Gaus_dt = bin_heights_Gaus_dt/max(bin_heights_Gaus_dt)
        # bin_heights_MonoExp_dt, bin_lefts_MonoExp_dt = np.histogram(dfMonoExpSelec['dt15mV'][:], bins=bins_dt)
        # bin_heights_MonoExp_dt = bin_heights_MonoExp_dt/max(bin_heights_MonoExp_dt)
        #
        #
        initial_guess_gaus = [max_count, bin_max_count, 0.1]
        initial_guess_monoExp = [max_count, 1]

        # print(bin_heights_dt[mask_gaus])
        # print(len(bin_heights_dt[:-1][mask_gaus]))
        # print(bin_lefts_dt[mask_gaus])
        # print(len(bin_lefts_dt[mask_gaus]))

        fit_gaus_params, pcov_gaus = curve_fit(Gaus, bin_lefts_dt[:-1][mask_gaus], bin_heights_dt[mask_gaus], p0 = initial_guess_gaus, bounds=([-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_monoExp_params, pcov_monoExp = curve_fit(monoExp, bin_lefts_dt[:-1][mask_monoExp], bin_heights_dt[mask_monoExp], p0 = initial_guess_monoExp, bounds=([-np.inf,-np.inf], [np.inf,np.inf])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_gaus_curve = Gaus(bin_lefts_dt[:-1][mask_gaus], *fit_gaus_params)

        fit_monoExp_curve = monoExp(bin_lefts_dt[:-1][mask_monoExp], *fit_monoExp_params)


        initial_guess_whole = [max_count,fit_gaus_params[1],fit_gaus_params[2],fit_monoExp_params[1]]


        fit_whole_params, pcov_whole = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess_whole, bounds=([-np.inf,-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf,np.inf])) #could
        # fit_whole_params, pcov_whole = curve_fit(convExpGaus, bin_lefts_dt[:-1], bin_heights_dt, p0 = initial_guess_whole, bounds=([-np.inf,fit_gaus_params[1]-math.copysign(1,fit_gaus_params[1])*0.5*fit_gaus_params[1],fit_gaus_params[2]-0.1*fit_gaus_params[2],fit_monoExp_params[1]-0.1*fit_monoExp_params[1]], [np.inf,fit_gaus_params[1]+math.copysign(1,fit_gaus_params[1])*0.5*fit_gaus_params[1],fit_gaus_params[2]+0.1*fit_gaus_params[2],fit_monoExp_params[1]+0.1*fit_monoExp_params[1]])) #could also add sigma (error on x,y)  could define if it’s the absolute or relative error

        fit_whole_curve = convExpGaus(bin_lefts_dt[:-1], *fit_whole_params)


        if(selected_sample==-1):
            ax10.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
            ax10.plot(bin_lefts_dt[:-1], fit_whole_curve, color=cmapred(0.35))
            ax10.set_xlabel("Time Delay (ns)")
            ax10.title.set_text(title)
            ax10.set_ylabel("Normalized Events")
        elif(i==selected_sample):
            ax10.step(bin_lefts_dt[:-1], bin_heights_dt, where = 'post', color=colorIndex[i],  linestyle=line_style[i], label=legNames[i], linewidth=1.5, alpha = alphaIndex[i])
            ax10.plot(bin_lefts_dt[:-1], fit_whole_curve, color=cmapred(0.35))
            ax10.set_xlabel("Time Delay (ns)")
            ax10.title.set_text(title)
            ax10.set_ylabel("Normalized Events")
        x = bin_lefts_dt
        y_histo = bin_heights_dt # calculates FWHM from the histogram width
        y_fit = fit_whole_curve # calculates FWHM from fit

    ############################# NOTE! ##################################
    #### The fit function is not modelling very well the distribution ####
    ### You can struggle with the initialization parameters of the fit ###
    ### OR decide to evaluate the FWHM and FWTM directly from the histo ##
    #################### ( like we are doing now ) #######################

        hmx_histo = half_max_x(x,y_histo)
        fwhm_histo = hmx_histo[1] - hmx_histo[0]
        half_histo = max(y_histo)/2.0
        hmx_fit = half_max_x(x,y_fit)
        fwhm_fit = hmx_fit[1] - hmx_fit[0]
        half_fit = max(y_fit)/2.0
        if(selected_sample==-1):
            ax10.plot(hmx_fit, [half_fit, half_fit], color ='orange')
            ax10.legend(loc='best')
        elif(i==selected_sample):
            ax10.plot(hmx_fit, [half_fit, half_fit], color ='orange')
            ax10.legend(loc='best')
        DTR_histo[i] = fwhm_histo*1000 #DTR in ps
        DTR_fit[i] = fwhm_fit*1000 #DTR in ps


        # tmx_histo = tenth_max_x(x,y_histo)
        # fwtm_histo = tmx_histo[1] - tmx_histo[0]
        # tenth_histo = max(y_histo)/10.0
        # tmx_fit = tenth_max_x(x,y_fit)
        # fwtm_fit = tmx_fit[1] - tmx_fit[0]
        # tenth_fit = max(y_fit)/10.0
        # if(selected_sample==-1):
        #     ax10.plot(tmx_fit, [tenth_fit, tenth_fit], color ='orange')
        # elif(i==selected_sample):
        #     ax10.plot(tmx_fit, [tenth_fit, tenth_fit], color ='orange')
        # DTR_10_histo[i] = fwtm_histo*1000
        # DTR_10_fit[i] = fwtm_fit*1000



        PDE = 0.53 # 0.53 for EJ232, 0.63 for LSO (LSO:Ce:Ca)
        NP = 0.56 # ??? for EJ232,0.56 for LSO at 9 keV
        LTE = 0.6
        SPe = 0.05
        SPe_amp = 0.00265653062
        CT = 1.33
        meanEn = 9

        LSO = 5.33
        LTE_LSO = 0.35
        meanEn_LSO = 15

        ILY[i] = LO[i]/(PDE*NP*LTE*meanEn*SPe*CT)
        ILY_LSO = LSO/(PDE*NP*LTE_LSO*meanEn_LSO*SPe*CT)
        ILY_amp[i] = LO_amp[i]/(PDE*NP*LTE*meanEn*SPe_amp*CT)

print('\n\n\n======================================== DTR results: =========================================')
for i in range (istart, istop):
    print('                                 ' + str(legNames[i]))
    print('         FWHM histo before TW =  ' + "{:.2f}".format(DTR_bef_tw_histo[i]) + ' ps,     FWTM histo before TW =  ' + "{:.2f}".format(DTR_bef_tw_10_histo[i]) + ' ps')
    print('         FWHM histo after TW =   ' + "{:.2f}".format(DTR_histo[i]) + ' ps,     FWTM histo after TW =   ' + "{:.2f}".format(DTR_10_histo[i]) + ' ps')
    print('')
    print('         FWHM fit before TW =    ' + "{:.2f}".format(DTR_bef_tw_fit[i]) + ' ps,     FWTM fit before TW =    ' + "{:.2f}".format(DTR_bef_tw_10_fit[i]) + ' ps')
    print('         FWHM fit after TW =     ' + "{:.2f}".format(DTR_fit[i]) + ' ps,     FWTM fit after TW =     ' + "{:.2f}".format(DTR_10_fit[i]) + ' ps')
    print ('SNR = ' + str(SNR[i]) + ', LO = ' + str(LO[i]) + ' nWb, ILY = ' + str(ILY[i]) + ' ph/keV')
    print('Amp LO = ' + str(LO_amp[i]) + ' V, Amp ILY = ' + str(ILY_amp[i]) + ' ph/keV')
    print('=================================================================================================')



handles, labels = ax1.get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_color()) for h in handles]
ax1.legend(handles=new_handles, labels=labels)
ax2.legend(handles=new_handles, labels=labels)
#ax3.legend(handles=new_handles, labels=labels)
ax4.legend(handles=new_handles, labels=labels)
#ax12.legend(handles=new_handles, labels=labels)

plt.show()

input("Press enter to exit;")
