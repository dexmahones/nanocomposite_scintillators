# Measurement and calibration for K-type thermocouple
import numpy as np
import matplotlib.pyplot as plt

def temp_from_voltage(uV):
    # For 0 to 500 deg. C range use:
    d0 = 0
    d1 = 5.08355e-2
    d2 = 7.860106e-8
    d3 = -2.503131e-10
    d4 = 8.315270e-14
    d5 = -1.228034e-17
    d6 = 9.804036e-22
    d7 = -4.413030e-26
    d8 = 1.057734e-30
    d9 = -1.0527551e-35
    coeffs = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    # return np.sum([coeffs[i]*uV**i for i in range(len(coeffs))])

    # Effectively  1uV:20°C
    return uV/20

vtest = np.linspace(0,2)*1e3
ttest = temp_from_voltage(vtest)

plt.plot(ttest,vtest)
plt.show()