import pyvisa as visa
from tm_devices import DeviceManager
from tm_devices.drivers import MSO6B
import time
import numpy as np
import pandas as pd
import sys
from zaber.serial import BinarySerial, BinaryCommand, BinaryDevice, BinaryReply

invertpos = False

port = BinarySerial("COM3")

def mm_to_step(mm,stepsize=0.000047625):
    step = round(mm/stepsize,0)
    return int(step)

def move_to_pos_mm(mm,port):
    steps = mm_to_step(mm)
    if steps < 533335 and steps > 0:
        port.write(BinaryCommand(0,20,steps))
        time.sleep(7)
        print("moved to position:",mm,"mm")
    elif steps == 0: 
        port.write(BinaryCommand(0,1))
        time.sleep(7)
        print("moved to position:",mm,"mm (home)")
    else:
        print("position out of range. no movement")

move_to_pos_mm(0,port)

#python Meas_with_multiple_DOIpos.py 14400 bulkBGO_3x3x20_BCM_HD_37V_symm_SSR 0,2,4,6,8,10,12,14,16,18,20    
positions = list(sys.argv[3].split(','))
rm = visa.ResourceManager()
my_instrument = rm.open_resource(rm.list_resources()[0])
print(my_instrument.query('*IDN?'))
duration = int(sys.argv[1])
Measurements = my_instrument.query('MEASUrement:LIST?').strip().split(',')
intervals = [0.01,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]

for count in range(len(positions)):
    position = float(positions[count])
    print(time.ctime(time.time()))
    print('measuring at position:',position,'mm')
    print('Running for', int(sys.argv[1]), 's')
    move_to_pos_mm(position,port)
    meas = []
    my_instrument.write('ACQUIRE:STATE OFF')
    my_instrument.write('CLEAR')
    my_instrument.write('ACQUIRE:STATE ON')
    old_pop = '0'
    time.sleep(1)
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
        if i<len(intervals) and time.time() > t_start + intervals[i] * duration:
            print(time.ctime(time.time()))
            print(int(intervals[i]*100),"% done")
            print("events:",len(meas))
            i=i+1
    print(time.ctime(time.time()))
    print("100 % done")
    print("events:",len(meas))
    my_instrument.write('ACQUIRE:STATE OFF')
    
    data = pd.DataFrame(columns=Measurements,data=np.array(meas).astype(float))

    if invertpos:
        DOI = str(positions[-1-count])
    else:
        DOI = str(position)
    data.to_csv(str(sys.argv[2])+'_'+DOI+'mm.csv', index=False)
    print('saving '+str(sys.argv[2])+'_'+DOI+'mm.csv')
    print('Events:',len(meas))

move_to_pos_mm(0,port)
port.close()