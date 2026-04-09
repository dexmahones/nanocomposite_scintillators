import pyvisa as visa
from tm_devices import DeviceManager
from tm_devices.drivers import MSO6B
import time
import numpy as np
import pandas as pd
import sys
#python Meas_with_multiple_Thresholds.py 5000 bulkBGO_3x3x20_BCM_HD_37V_symm_SSR 4,6,8,10,12,15,20,30,50 back   
thresholds = list(sys.argv[3].split(','))
rm = visa.ResourceManager()
my_instrument = rm.open_resource(rm.list_resources()[0])
print(my_instrument.query('*IDN?'))
duration = int(sys.argv[1])
Measurements = my_instrument.query('MEASUrement:LIST?').strip().split(',')
intervals = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

for threshold in thresholds:
    if sys.argv[4] == "back":
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
    elif sys.argv[4] == "front":
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
    elif sys.argv[4] == "both":
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
    elif sys.argv[4] == "symmDSR":
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels1:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels1:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels1:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS7:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS13:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLLow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:FALLHigh -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISELow -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISEMid -'+threshold+'E-3')
        my_instrument.write('MEASUrement:MEAS20:REFLevels2:ABSolute:RISEHigh -'+threshold+'E-3')
    print(time.ctime(time.time()))
    print('measuring with',threshold,'mV')
    print('Running for', int(sys.argv[1]), 's')
    
    meas = []
    my_instrument.write('ACQUIRE:STATE OFF')
    time.sleep(.5)
    my_instrument.write('CLEAR')
    time.sleep(.5)
    my_instrument.write('ACQUIRE:STATE ON')
    old_pop = '0'
    time.sleep(.5)
    t_end = time.time() + duration
    t_start = time.time()
    i=0
    while time.time() < t_end:
        new_meas = []
        new_pop = my_instrument.query('MEASUrement:'+Measurements[0]+':SUBGROUP:RESUlts:ALLAcqs:POPUlation? "OUTPUT"')
        if old_pop != new_pop:
            my_instrument.write('ACQUIRE:STATE OFF')
            for Measurement in Measurements: 
                new_meas.append(my_instrument.query('MEASUrement:'+Measurement+':SUBGROUP:RESUlts:CURRentacq:MEAN? "OUTPUT"'))
            my_instrument.write('ACQUIRE:STATE ON')
            meas.append(new_meas)
            old_pop = new_pop
        if i<19 and time.time() > t_start + intervals[i] * duration:
            print(time.ctime(time.time()))
            print(int(intervals[i]*100),"% done")
            print("events:",len(meas))
            i=i+1
    print(time.ctime(time.time()))
    print("100 % done")
    print("events:",len(meas))
    my_instrument.write('ACQUIRE:STATE OFF')
    time.sleep(.5)
    
    data = pd.DataFrame(columns=Measurements,data=np.array(meas).astype(float))
    
    data.to_csv(str(sys.argv[2])+'_'+str(threshold)+'mV.csv', index=False)
    print('saving '+str(sys.argv[2])+'_'+str(threshold)+'mV.csv')
    print('Events:',len(meas))
    time.sleep(.5)