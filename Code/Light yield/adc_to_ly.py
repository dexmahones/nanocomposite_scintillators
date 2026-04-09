def adc_to_photons(S, Cmeas, Ssingle, Csingle, A, QE):
	return S*Cmeas/(Ssingle*Csingle)*pow(10, A/20)*(1/QE)*(1/Energy)

# S = float(input("Value of photopeak in ADC units: \n"))
# Cmeas = float(input("Digitizer charge sensitivity: \n"))
# Ssingle = float(input("Value of photopeak for the single photoélec measurements in ADC units: \n"))
# Csingle = float(input("Single photoélec charge sensitivity: \n"))
# QE = float(input("Quantum efficiency in %: \n"))
# A = float(input("Attenuation in dB: \n"))
# Energy = float(input("Energy in Mev of the photon's source: \n"))

S = 13700 #2330
Cmeas = 160.
Ssingle = 122.85
Csingle = 40.
QE = 8/100.
A = 12.
Energy = .662

ly = adc_to_photons(S, Cmeas, Ssingle, Csingle, A, QE)
print("The Light Yield is {:.2e} or {:.2f} Photons/MeV".format(ly, ly))
