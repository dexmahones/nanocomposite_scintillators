import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson  # for numerical integration
import matplotlib.pyplot as plt

# Choose photodetector
# photodetector = 'Broadcom NUV-MT SiPM'
# photodetector = 'Hamamatsu R2059 PMT'
photodetector = 'Hamamatsu R7600U-00 PMT' #Same QE as Hamamatsu R11187

# Choose scintillator
# scintillator = 'LSO:Ce:Ca 3x3x3 mm^3'
# scintillator = 'EJ232 3x3x3 mm^3'
# scintillator = 'PVTBBQHf2 3x3x3 mm^3'
# scintillator = 'EJ228'
scintillator = 'EJ262'
# scintillator = 'GAGG'

# Load PMT QE and SiPM PDE data
if (photodetector == 'Broadcom NUV-MT SiPM'):
    data_photodetector = pd.read_csv('Broadcom_NUV-MT_SiPM_PDE.txt', delimiter=',')

elif (photodetector == 'Hamamatsu R2059 PMT'):
    data_photodetector = pd.read_csv('Hamamatsu_R2059_QE.txt', delimiter=',')
elif (photodetector == 'Hamamatsu R7600U-00 PMT'):
    data_photodetector = pd.read_csv('Hamamatsu_R7600U-00_QE.txt', delimiter=',')
else:
    print('Select photodetector!')

# Load emission spectrum of scintillator
if (scintillator == 'LSO:Ce:Ca 3x3x3 mm^3'):
    data_scintillator = pd.read_csv('data/lightYieldTest/LSOexcAt350.txt', delimiter='\t', skiprows=54)
elif (scintillator == 'EJ232 3x3x3 mm^3'):
    data_scintillator = pd.read_csv('data/lightYieldTest/EJ232excAt320.txt', delimiter='\t', skiprows=54)
elif (scintillator == 'PVTBBQHf2 3x3x3 mm^3'):
    data_scintillator = pd.read_csv('PVTBBQHf2_spectrum.txt', delimiter=',')
elif (scintillator == 'EJ228'):
    data_scintillator = pd.read_csv('EJ228_spectrum.txt', delimiter=',')
elif (scintillator == 'EJ262'):
    data_scintillator = pd.read_csv('EJ262_spectrum.txt', delimiter=',')
elif (scintillator == 'GAGG'):
    data_scintillator = pd.read_csv('GAGG_spectrum.txt', delimiter=',')
else:
    print('Select scintillator!')

# Extract values of x and y for both scintillator and photodetector
x_photodetector = data_photodetector['x'].values
y_photodetector = data_photodetector['y'].values

x_scintillator = data_scintillator['x'].values
y_scintillator = 1000*data_scintillator['y'].values # Multiplication by a large number increases the accuracy of the interpolation
                                                    # and does not influence the result as the emission spectra is later normalized to 1!

# Interpolate the data of detector and scintillator, offers to use different binnings
f_detector = interp1d(x_photodetector, y_photodetector, kind='linear', fill_value="extrapolate")
f_scintillator = interp1d(x_scintillator, y_scintillator, kind='linear', fill_value="extrapolate")

# Generate new values with arbitrary binning so that we can match the two curves
binning = 1 # Choose binning
if (photodetector == 'Broadcom NUV-MT SiPM'):
    x_photodetector_new = np.arange(250, 895, binning)
elif (photodetector == 'Hamamatsu R2059 PMT'):
    x_photodetector_new = np.arange(290, 665, binning)
elif (photodetector == 'Hamamatsu R7600U-00 PMT'):
    x_photodetector_new = np.arange(290, 665, binning)
y_photodetector_new = f_detector(x_photodetector_new) # Calculate the interpolation

# Select the range of the measurement, the rest is set as 0
# Otherwise problems could arise from linear interpolation
if (scintillator == 'LSO:Ce:Ca 3x3x3 mm^3'):
    x_scintillator_new = np.arange(370, 500, binning)
elif (scintillator == 'EJ232 3x3x3 mm^3'):
    x_scintillator_new = np.arange(350, 480, binning)
elif (scintillator == 'PVTBBQHf2 3x3x3 mm^3'):
    x_scintillator_new = np.arange(300, 600, binning)
elif (scintillator == 'EJ228'):
    x_scintillator_new = np.arange(350, 500, binning)
elif (scintillator == 'EJ262'):
    x_scintillator_new = np.arange(450, 600, binning)
elif (scintillator == 'GAGG'):
    x_scintillator_new = np.arange(450, 700, binning)
y_scintillator_new = f_scintillator(x_scintillator_new) # Calculate the interpolation

# Select values of x that are in the PMT x range but we did not measure them
x_difference = np.setdiff1d(x_photodetector_new, x_scintillator_new)

# Create a new vector of y values
y_total = np.zeros_like(x_photodetector_new)

for i, x_value in enumerate(x_photodetector_new):
    if x_value in x_difference:
        y_total[i] = 0  # If we did not measure this, set as 0
    else:
        # Find corresponding bin from the emission spectra
        idx = np.where(x_scintillator_new == x_value)[0]
        if len(idx) > 0:  # If this bin exists
            y_total[i] = y_scintillator_new[idx[0]] # Set the value as the one obtained from the measurement


print(f"PMT integral = {simpson(y=y_photodetector_new, x=x_photodetector_new)}") # Check PMT integral (just for fun)

# Normalization of the emission spectra: calculation of the integral by the Simpsons method
integral = simpson(y=y_total,x=x_photodetector_new)

# Renormalize so that the integral of the emission spectra is 1
y_total_renorm = y_total / integral

print(f"Emission spectrum integral after renormalization = {simpson(y=y_total_renorm, x=x_photodetector_new)} (should be 1.0).") # Check that this is around 1

total_QE = binning*np.dot(y_photodetector_new, y_total_renorm) # Calculate the convolution of the photodetector efficiency and scintillator emission spectrum
print(f"Total quantum efficiency for {photodetector} and {scintillator} scintillator is QE = {total_QE} %.")

# Plot the graphs
plt.plot(x_photodetector_new, y_photodetector_new, 'o', label = photodetector)
plt.plot(x_photodetector_new, 3000*y_total_renorm, 'o', label = scintillator)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
