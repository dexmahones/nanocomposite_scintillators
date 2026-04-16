import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tkinter as tk
from tkinter import filedialog
from plot_utils import PlotStyleManager
from scipy.optimize import curve_fit
import os
import seaborn as sns
import re

PSM = PlotStyleManager(cmap_name='turbo', n_colors = 21)

def load_file(file_path = None):
    if not file_path:
        # Initialize tkinter and hide the main window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open file browser specifically for CSV files
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

    # Import the CSV as a pandas DataFrame
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # print("CSV successfully imported!")
            return df
        except Exception as e:
            print(f"Error importing CSV: {e}")
            return None
    else:
        print("No file selected.")
        return None

def gaussian_model(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def n_gaussians(x, *parms):
    amps = parms[::3]
    mus = parms[1::3]
    sigmas = parms[2::3]

    yres = np.zeros_like(x)
    for n in range(len(amps)):
        yres += gaussian_model(x, amps[n], mus[n], sigmas[n])
    return yres

def red_chi_square(y_obs, y_fit, y_err, n_params):
    
    y_obs = np.array(y_obs)
    y_fit = np.array(y_fit)
    y_err = np.array(y_err)
    
    # Degrees of Freedom = (Number of Data Points) - (Number of Parameters)
    dof = len(y_obs) - n_params
    
    if dof <= 0:
        return np.nan
        
    # Standard Chi-Squared formula: sum of squared residuals / error squared
    chi_sq = np.sum(((y_obs - y_fit) / y_err)**2)
    
    # Reduced Chi-Squared
    red_chi_sq = chi_sq / dof
    return red_chi_sq

def fit_n_gaussians(x, y, num_peaks = 2, max_depth = 5, depth = 1):
    sigma_bg = 0.02
    # y_err = np.sqrt(np.abs(y) + sigma_bg**2) # assume Poisson error
    y_err = np.ones_like(x) * sigma_bg
    # Initial Guess (p0)
    # Peak 1: Use the global max
    # Peak 2: A common trick is to guess it's at 80% of the max or shifted
    p0 = [
            np.max(y),
            x[np.argmax(y)],
            float(np.random.uniform(1e-2, max(x)))
        ]

    for n in range(1,num_peaks):
        p0.append(np.max(y)/(n+1))
        p0.append(float(np.random.uniform(min(x),max(x))))
        p0.append(float(np.random.uniform(1e-1, max(x))))
    try:
        bounds = [
            [0.0,min(x),1e-2]*num_peaks,
            [np.inf,max(x),np.inf]*num_peaks
        ]
        popt, pcov = curve_fit(n_gaussians, x, y, p0=p0, bounds = bounds)
        perr = np.sqrt(np.diag(pcov)) # Standard deviation errors

        amps = popt[::3].copy()
        mus = popt[1::3].copy()
        sigmas = popt[2::3].copy()

        u_amps = perr[::3]
        u_mus = perr[1::3]
        u_sigmas = perr[2::3]

        sorted_idx = np.argsort(mus)[::-1]

        amps = amps[sorted_idx]
        mus = mus[sorted_idx]
        sigmas = sigmas[sorted_idx]

        u_amps = u_amps[sorted_idx]
        u_mus = u_mus[sorted_idx]
        u_sigmas = u_sigmas[sorted_idx]

        # Calc xi square per dof
        red_chi_sq = red_chi_square(y,n_gaussians(x, *popt),y_err,len(popt)+1)
        # print(depth,"\t",num_peaks, "\t", red_chi_sq)
        if (red_chi_sq < 0.1 or red_chi_sq > 1.1) and depth < max_depth:
            return fit_n_gaussians(x,y,num_peaks = num_peaks+1, depth = depth+1, max_depth = max_depth)
        
        return {
            "peaks": [
                {
                    "amp": amps[n], "mu": mus[n], "sigma": sigmas[n], "fwhm": 2.355 * sigmas[n], 
                    "u_mu": u_mus[n], "u_sigma":u_sigmas[n], "u_amp":u_amps[n]
                }
                for n in range(len(amps))
            ],
            "fit_y": n_gaussians(x, *popt)
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return {"peaks": [], "fit_y":y}

def find_optimal_spectrum(df, window = 5):
    max_em_peak = 0
    max_em_lambda = 0
    optimal_em = None
    max_ex_peak = 0
    max_ex_lambda = 0
    optimal_ex = None
    samples = {}
    
    # Get the sample names from the very first row
    sample_names = df.columns[::2] # Takes every 2nd column name

    # Loop through the columns in pairs
    for i, name in enumerate(sample_names):
        samples[name] = {}
        # Determine if this is an emission spectrum at fixed excitation or an  excitation spectrum at fixed emission based on filename
        # The convention is:
        #   emission spectrum at fixed excitation   ->      ex{excitation wavelength}   (e.g. SAMPLE_NAME ex310)
        #   excitation spectrum at fixed emission   ->      em{emission wavelength}     (e.g. SAMPLE_NAME em420)

        emission, excitation = None, None
        try:
            emission = float(name.split("em")[-1][:3])
            samples[name]["scantype"] = "ex"
        except:
            try:
                excitation = float(name.split("ex")[-1][:3])
                samples[name]['scantype'] = "em"
            except:
                # print(f"Emission/excitation identification failed for {name}.")
                samples[name]['scantype'] = "unknown"

        col_idx = i * 2
        # Extract the pair (Wavelength and Intensity)
        # Skip the row that says "Wavelength (nm)" and convert to numbers
        pair = df.iloc[1:, col_idx : col_idx + 2].apply(pd.to_numeric, errors='coerce')
    
        # Drop rows where everything is NaN
        pair = pair.dropna(how='all')
        
        if not pair.empty:
            # Store as a dictionary entry
            samples[name].update({
                'wavelength': pair.iloc[:, 0].values,
                'intensity': pair.iloc[:, 1].values - min(pair.iloc[:, 1].values) + 1e-6,
                'excitation': excitation,
                'emission': emission
            })
            weights = np.ones(window)/window
            moving_average = np.convolve(samples[name]['intensity'],weights,mode="same")
            # If emission spectrum, exclude resonance peak at 2x excitation wavelength.
            if excitation:
                samples[name]['intensity'] = samples[name]['intensity'][samples[name]['wavelength']<excitation*2 - 10]
                samples[name]['wavelength'] = samples[name]['wavelength'][samples[name]['wavelength']<excitation*2 - 10]
                if max(moving_average) > max_em_peak:
                    optimal_em = name
                    max_em_peak = max(moving_average)
                    max_em_lambda = samples[name]['wavelength'][np.argmax(moving_average)]
            # Any cuts for excitation spectrum? I guess we just want excitation less than the emission peak?
            elif emission:
                samples[name]['intensity'] = samples[name]['intensity'][samples[name]['wavelength']<emission - 10]
                samples[name]['wavelength'] = samples[name]['wavelength'][samples[name]['wavelength']<emission - 10]
                if max(moving_average) > max_ex_peak:
                    optimal_ex = name
                    max_ex_peak = max(moving_average)
                    max_ex_lambda = samples[name]['wavelength'][np.argmax(moving_average)]
            else:
                print(f"Please specify emission or excitation in sample {name}!")

            # Convert to energy space
            samples[name].update({
                'energy': 1239.84 / samples[name]['wavelength'],
                'energy_corrected_intensity': samples[name]['intensity'] * samples[name]['wavelength']**2,
            })
    print(f"Optimal emission: \t {optimal_em} \t {max_em_lambda}")
    print(f"Optimal excitation: \t {optimal_ex} \t {max_ex_lambda}")
    
    return samples, optimal_em, optimal_ex

def analyse_dataframe(df, plot = True, ax = None, analyse = True, num_peaks = 2, normalize = True, color_idx = 0, new_figure = False, max_peaks = 5,use_optimal = False, convert_to_energy = False):
    
    all_samples, optimal_em, optimal_ex = find_optimal_spectrum(df)
    samples = {key:val for key, val in all_samples.items() if key in [optimal_em,optimal_ex]}
    sample_list = samples.keys()
    # Loop through the columns in pairs
    for i, name in enumerate(sample_list):
        scantype = samples[name]['scantype']

        color_idx +=i
        # Check if there is a color available. Otherwise start repeating colors.
        if color_idx >= len(PSM.custom_colors):
            color_idx = 0

        if convert_to_energy:
            xs, ys = samples[name]['energy'], samples[name]['energy_corrected_intensity']
        else:
            xs, ys = samples[name]['wavelength'], samples[name]['intensity']

        if normalize:
            # ys /= sum(ys) * np.abs(np.diff(samples[name]['energy'])[0])
            ys /= max(ys)

        if analyse:
            domain_width = (max(xs)-min(xs))/2
            xpeak = xs[np.argmax(ys)]
            xmask = (xs>xpeak-domain_width/2)&(xs<xpeak+domain_width/2)
            xfit = xs[xmask]
            yfit = ys[xmask]
            res = fit_n_gaussians(xfit, yfit, num_peaks, max_depth = max_peaks)
            samples[name].update(res)
            try:
                max_amp_idx = np.argmax([peak["amp"] for peak in res["peaks"]])
                x_peak = res["peaks"][max_amp_idx]["mu"]
                # print(f"{name}\t{x_peak:.2f}")
            except:
                pass
        if plot and ax:
            if new_figure:
                fig, ax = plt.subplot_mosaic([[scantype]],figsize = (12,7))
            ax[scantype].plot(xs,ys, ls = ":", color = PSM.custom_colors[color_idx]) # plot raw data
            try: 
                ax[scantype].plot(xfit, samples[name]['fit_y'],label = f"{name}", ls = "-", color = PSM.custom_colors[color_idx], lw = 2, alpha = 0.5) # plot fitted model
                for j,peak in enumerate(samples[name]["peaks"]):
                    ax[scantype].vlines(samples[name]["peaks"][j]['mu'], 0, np.max(ys)*1.75,color = PSM.custom_colors[color_idx]) # plot fitted gaussian
                    ax[scantype].plot(xs,gaussian_model(xs,samples[name]["peaks"][j]['amp'],samples[name]["peaks"][j]['mu'],samples[name]["peaks"][j]['sigma']), color = PSM.custom_colors[color_idx], ls = "--")
                    ax[scantype].text(
                        samples[name]["peaks"][j]['mu'], 
                        np.max(ys)*1.75, 
                        f"${samples[name]["peaks"][j]["mu"]:.2f}\pm{samples[name]["peaks"][j]["u_mu"]:.2f}$nm",
                        rotation = 90,
                        va = 'top',
                        ha = 'right')
            except Exception as e:
                print(e) 
                pass
            if new_figure:
                for peak in res['peaks']:
                    for key, val in peak.items():
                        print(key,"\t",np.round(val,4))
                ax[scantype].grid(True)
                ax[scantype].legend()
                ax[scantype].set_xlabel("Wavelength (nm)")
                ax[scantype].set_ylabel("Intensity (a.u.)")
                plt.show()

            # print(f"Imported: {name} ({len(pair)} data points)")
    return samples

def analyse_folder(plot = True, ax = None, num_peaks = 2, normalize = True, analyse = True, new_figure = False, max_peaks = 5,use_optimal = False, convert_to_energy = False):
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open folder browser
    folder_path = filedialog.askdirectory(
        title="Select folder",
    )

    data = {}
    if folder_path:
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        if plot and not ax:
            fig, ax = plt.subplot_mosaic(
                [["em","ex"]],
                figsize = (12,7)
            )
        
        for j,file in enumerate(files):
            df = load_file(os.path.join(folder_path,file))

            all_data = analyse_dataframe(df, plot=plot, ax=ax, num_peaks = num_peaks, normalize = normalize, color_idx=j, new_figure = new_figure, max_peaks = max_peaks,use_optimal = use_optimal, convert_to_energy = convert_to_energy, analyse = analyse)
            data.update(all_data)
    else:
        print("No folder selcted. Cancelling operation.")

    return data

def tag_reaction_time(data):
    for sample, val in data.items():
        if 'sec' in sample:
            try:
                rt = float(re.findall(r'\d+',sample.split('sec')[0])[0]) # in seconds
                data[sample]["time"] = rt    
            except:
                rt = None
            
    return data

def unpack_peaks(data):
    unpacked_data = []
    for sample, val in data.items():
        for peak in val['peaks']:
            entry = {
                key:data for key,data in val.items() if key not in ['wavelength','intensity','peaks']
            }
            entry.update({"sample":sample})
            entry.update(peak)
            unpacked_data.append(entry)
    return unpacked_data

def wavelength_to_rgb(nm):
    gamma = 0.8
    if 380 <= nm <= 440:
        R, G, B = -(nm - 440) / (440 - 380), 0.0, 1.0
    elif 440 < nm <= 490:
        R, G, B = 0.0, (nm - 440) / (490 - 440), 1.0
    elif 490 < nm <= 510:
        R, G, B = 0.0, 1.0, -(nm - 510) / (510 - 490)
    elif 510 < nm <= 580:
        R, G, B = (nm - 510) / (580 - 510), 1.0, 0.0
    elif 580 < nm <= 645:
        R, G, B = 1.0, -(nm - 645) / (645 - 580), 0.0
    elif 645 < nm <= 750:
        R, G, B = 1.0, 0.0, 0.0
    else:
        R, G, B = 0.0, 0.0, 0.0
        
    # Apply intensity attenuation near limits and gamma correction
    factor = 0.3 + 0.7 * (nm - 380) / (440 - 380) if 380 <= nm < 420 else \
             0.3 + 0.7 * (750 - nm) / (750 - 700) if 700 < nm <= 750 else 1.0
             
    return ((R * factor)**gamma, (G * factor)**gamma, (B * factor)**gamma)

def main(single_file = True, plot = True, num_peaks = 1, max_peaks = 3, normalize = True, new_figure = False, use_optimal = True, convert_to_energy = False):

    # Figure cosmetic class
    PSM = PlotStyleManager(cmap_name='turbo', n_colors = 10)

    # Initialize figure
    if not new_figure:
        fig, ax = plt.subplot_mosaic(
                [["em","ex"]],
                figsize = (12,7)
            )
        ax['ex'].set_title("Excitation Spectrum")
        ax['em'].set_title("Emission Spectrum")
        for a in ['em','ex']:
            ax[a].grid(True)
            ax[a].legend()
            ax[a].set_xlabel("Wavelength (nm)")
            ax[a].set_ylabel("Intensity (a.u.)")
    else:
        ax = None
    
    if single_file:
        df = load_file()
        data = analyse_dataframe(df, plot=plot, ax=ax, num_peaks = num_peaks, normalize = normalize, new_figure=new_figure,max_peaks = max_peaks,use_optimal=use_optimal,convert_to_energy=convert_to_energy)
    else:
        data = analyse_folder(plot = plot, ax = ax, num_peaks = num_peaks, normalize = normalize, new_figure=new_figure, max_peaks = max_peaks,use_optimal=use_optimal,convert_to_energy=convert_to_energy)
    plt.show() # Show spectrum data

    data = tag_reaction_time(data) # Add reaction time info
    data = unpack_peaks(data) # De-nest peak info

    df = pd.DataFrame(data)
    df['scantype'] = None 
    df.loc[pd.isna(df['emission']),'scantype'] = 'emission'
    df.loc[pd.isna(df['excitation']),'scantype'] = 'excitation'

    exdf = df[df['scantype']=='excitation']
    emdf = df[df['scantype']=='emission']

    excitation_cut = (emdf["excitation"]>=260)&(emdf["excitation"]<=360)
    emdf = emdf[excitation_cut].dropna(subset = "time")

    em_mu = []
    exs = []
    u_em_mu = []
    ts = []
    for i,t in enumerate(sorted(pd.unique(emdf["time"]))):
        # Find max peak at time
        dff = emdf[(emdf["time"]==t)]
        dff = dff[dff["amp"]==dff['amp'].max()]

        em_mu.append(dff['mu'].max())
        exs.append(dff['excitation'].max())
        u_em_mu.append(dff['u_mu'].max())
        ts.append(t)
    
    plt.scatter(
        ts,
        em_mu,
        marker = 'o',
        color = "g",
    )
    # plt.errorbar(
    #     emdf["time"],
    #     emdf["mu"],
    #     yerr = emdf["u_mu"],
    #     fmt = 'xk',
    #     capsize = 5,
    # )
    plt.errorbar(
        df["time"],
        df["mu"],
        yerr = df["u_mu"],
        fmt = 'xk',
        capsize = 5,
    )
    # plt.bar(ts,em_lambdas,width = 5,alpha = 0.25, color = "blue")
    # plt.hlines(np.mean(em_mu),min(ts),max(ts),ls = "--", lw = 2, alpha = 0.75, color = 'r')
    # plt.scatter(ts,exs)
    plt.title("CdS emission peak for increasing reaction time.")
    plt.xlabel("Reaction Time (s)")
    plt.ylabel("Wavelength (nm)")
    # plt.ylim(370,400)
    plt.show()

    unique_wavelengths = sorted(df['mu'].unique())
    spectral_palette = [wavelength_to_rgb(w) for w in unique_wavelengths]

    sns.relplot(
        df, x = "time", y = "mu", hue = "mu", size = "u_mu", palette=spectral_palette, legend = False
    )
    plt.show()

# Flags
single_file =       False   # True for one CSV, False for all CSVs in a directory
plot =              True    # True to view all generated plots
num_peaks =         1       # Number of gaussians used in fitting
max_peaks =         1       # Upper limit on number of gaussians in fit
normalize =         True    # True maps intensity between 0 and 1
new_figure =        False   # Plot each scan on a new set of axes
use_optimal =       True    # Only use highest peak intensity scan
convert_to_energy = False   # Convert wavelength (nm) to energy (eV): E = 1239.84 / wavelength

# Run the script
if __name__ == "__main__":
    main(single_file=single_file,plot=plot,num_peaks=num_peaks,max_peaks=max_peaks,normalize=normalize,new_figure=new_figure,use_optimal=use_optimal, convert_to_energy=convert_to_energy)
