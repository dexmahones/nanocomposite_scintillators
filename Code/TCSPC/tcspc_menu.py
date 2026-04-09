import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows
from scipy.stats import norm
import pandas as pd
import pyswarms as ps
import optuna
from pyswarms.single.global_best import GlobalBestPSO
import glob
import os
from plot_utils import set_plot_style
from tcspc_fitting import run_optuna_trial

class DecayAnalyser:
    def __init__(self, signal_path, background_path=None, max_time = None, normalised = False, noise_window = 4, show_plots = False, log_y = True, line_alpha = 0.75, cmap_name = 'jet', n_colors = 2, plot_bg = True):
        self.colors = set_plot_style(cmap_name=cmap_name, n_colors=n_colors)
        self.cmap = cmap_name
        self.n_colors = n_colors
        
        self.line_alpha = line_alpha
        
        self.signal_data = self.read_datafile(signal_path)
        self.bg_data = self.read_datafile(background_path) if background_path else None
        
        self.counts = None
        self.ts = None
        self.bg_counts = None
        self.bg_ts = None
        self.prev_counts = None
        self.prev_ts = None
        
        self.noise_window = noise_window
        self.bg_baseline = None
        self.signal_baseline = None
        self.signal_start = None
        self.bg_start = None
        self.signal_peak_time = None
        self.bg_peak_time = None
        self.max_time = max_time if max_time else max(self.signal_data)
        self.normalised = normalised
        self.show_plots = show_plots
        self.plot_bg = plot_bg
        self.smooth_counts = None
        self.smooth_ts = None
        self.best_fit = None
        self.best_ts = None
        self.best_pos = None
        self.best_fit_score = None
        self.plot_log_yscale = log_y
        self.bin_data()
    
    # === Signal Data ===
    def getSignalData(self):
        return self.signal_data

    def setSignalData(self, val):
        self.signal_data = val
        
    def getSmoothCounts(self):
        return self.smooth_counts

    def setSmoothCounts(self, val):
        self.smooth_counts = val
    
    # === Background Data ===
    def getBGData(self):
        return self.bg_data
    
    def setBGData(self, val):
        self.bg_data = val
    
    # === Counts ===
    def getCounts(self):
        return self.counts
    
    def setCounts(self, val):
        self.setPrevCounts(self.getCounts())
        self.counts = val
    
    def getPrevCounts(self):
        return self.prev_counts
    
    def setPrevCounts(self, val):
        self.prev_counts = val

    def getPrevTS(self):
        return self.prev_ts
    
    def setPrevTS(self, val):
        self.prev_ts = val

    def undoSetCounts(self):
        self.setTS(self.getPrevTS())
        self.setCounts(self.getPrevCounts)
    
    def getNoiseWindow(self):
        return self.noise_window
    
    def setNoiseWindow(self, val):
        self.noise_window = val

    # === Time Stamps ===
    def getTS(self):
        return self.ts
    
    def setTS(self, val):
        self.setPrevTS(self.getTS())
        self.ts = val
    
    def getSmoothTS(self):
        return self.smooth_ts
    
    def setSmoothTS(self, val):
        self.smooth_ts = val
    
    # === Background Counts ===
    def getBGCounts(self):
        return self.bg_counts
    
    def setBGCounts(self, val):
        self.bg_counts = val
    
    # === Background Time Stamps ===
    def getBGTS(self):
        return self.bg_ts
    
    def setBGTS(self, val):
        self.bg_ts = val
    
    # === Max Time ===
    def getMaxTime(self):
        return self.max_time
    
    def setMaxTime(self, val):
        self.max_time = val
    
    def setSignalBaseline(self, val):
        self.signal_baseline = val
        
    def getSignalBaseline(self):
        return self.signal_baseline
    
    def setBGBaseline(self, val):
        self.bg_baseline = val
        
    def getBGBaseline(self):
        return self.bg_baseline
    
    def getSignalStart(self):
        return self.signal_start
    
    def setSignalStart(self, val):
        self.signal_start = val
        
    def getBGStart(self):
        return self.bg_start
    
    def setBGStart(self, val):
        self.bg_start = val
        
    def getSignalPeakTime(self):
        return self.signal_peak_time
    
    def setSignalPeakTime(self, val):
        self.signal_peak_time = val
        
    def getBGPeakTime(self):
        return self.bg_peak_time
    
    def setBGPeakTime(self, val):
        self.bg_peak_time = val
    
    # === Normalisation Flag ===
    def getNormalised(self):
        return self.normalised
    
    def setNormalised(self, val):
        self.normalised = bool(val)
    
    # === Plot Display Flag ===
    def getShowPlots(self):
        return self.show_plots
    
    def setShowPlots(self, val):
        self.show_plots = bool(val)
    
    def getPlotBG(self):
        return self.plot_bg

    def setPlotBG(self,val):
        self.plot_bg = val
        
    def getPlotLogYscale(self):
        return self.plot_log_yscale

    def setPlotLogYScale(self,val):
        self.plot_log_yscale = val
        
    def getCmapName(self):
        return self.cmap

    def setCmapName(self, cmap):
        self.cmap = cmap
        self.colors = set_plot_style(cmap_name=cmap, n_colors=self.n_colors)
        
        
    def normalise_data(self):
        self.setNormalised(True)
        if self.counts is not None:
            self.setCounts(self.counts/(sum(self.counts)*(self.ts[1]-self.ts[0])))
        if self.bg_counts is not None:
            self.setBGCounts(self.bg_counts/(sum(self.bg_counts)*(self.bg_ts[1]-self.bg_ts[0])))
        if self.show_plots:
            self.plot_data()
        print("Data normalised.")
        
    def read_datafile(self, file_path):
        if not file_path:
            return None
        with open(file_path, "r") as infile:
            events = np.array(infile.read().split()).astype(float)
        print(f"Successfully read file {file_path}")
        return events
    
    def plot_data(self, save_path = None):
        if self.bg_counts is not None and self.plot_bg:
            plt.scatter(self.bg_ts, self.bg_counts, label="Background", alpha = self.line_alpha, color = self.colors[-2])
        plt.scatter(self.ts, self.counts, label="Signal", alpha = self.line_alpha, color = self.colors[-3])
            
        if self.signal_start is not None:
            plt.vlines(self.signal_start,0,max(self.counts),color = "r", ls = "--")
            
        if self.signal_baseline is not None:
            plt.hlines(self.signal_baseline,min(self.ts),max(self.ts),color = "r", ls = "--")
            
        if self.bg_start is not None and self.plot_bg:
            plt.vlines(self.bg_start,0,max(self.counts),color = "k", ls = "--", alpha = 1)
            
        if self.bg_baseline is not None and self.plot_bg:
            plt.hlines(self.bg_baseline,min(self.ts),max(self.ts),color = "k", ls = "--", alpha = 1)
            
        if self.bg_peak_time is not None and self.plot_bg:
            plt.vlines(self.bg_peak_time,0,max(self.counts),color = "k", ls = "--", alpha = 1)
            
        if self.signal_peak_time is not None:
            plt.vlines(self.signal_peak_time,0,max(self.counts),color = "r", ls = "--", alpha = 1)
            
        if self.smooth_counts is not None:
            plt.plot(self.smooth_ts, self.smooth_counts, label = "Moving Ave.", alpha = 0.4, lw = 1)
        
        if self.best_fit is not None:
            plt.plot(self.best_ts, self.best_fit, label = "Best Fit.",ls = "--", alpha = 1, lw = 2, color = "r")
            
        if self.plot_log_yscale:
            plt.yscale("log")
            plt.ylim(1e-6,max(self.counts)*2)
        
        if self.best_pos is not None:
            result = ""
            result+=("td: "+"    ".join(np.round(self.best_pos[0::3],4).astype(str))+"\n")
            result+=("tr: "+"    ".join(np.round(self.best_pos[1::3],4).astype(str))+"\n")
            result+=("P:  "+"    ".join(np.round(self.best_pos[2::3],4).astype(str))+"\n")
            
            plt.text(self.ts[len(self.ts)//2], max(self.counts)/2, result, va = 'center', ha = 'center')
        
        plt.legend()
        plt.xlabel("Delay time (ns)")
        plt.ylabel("Norm. counts (a.u)")
        
        if save_path:
            plt.savefig(save_path, format="svg")
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.show()
        
    def bin_data(self):
        events = self.signal_data
        max_time = self.max_time
        
        raw_count_data, bins = np.histogram(events, bins=int(max(events)))
        raw_ts = (1/2 + bins[:-1]) / (max(events) - min(events)) * max_time
        counts = raw_count_data.astype(float)
        
        self.setCounts(counts)
        self.setTS(raw_ts)
        

        if self.bg_data is not None:
            bg_count_data, _ = np.histogram(self.bg_data, bins=bins)
            self.setBGCounts(bg_count_data.astype(float))
            self.setBGTS(raw_ts)
        
        if self.normalised:
            self.normalise_data()
            
        if self.show_plots:
            self.plot_data()
            
        print("Data binning complete.")

    def apply_cut(self, min_t, max_t):
        mask = (self.ts > min_t) & (self.ts < max_t)
        self.setCounts(self.counts[mask])
        self.setTS(self.ts[mask])
        
        if self.bg_counts is not None:
            mask = (self.bg_ts > min_t) & (self.bg_ts < max_t)
            self.setBGCounts(self.bg_counts[mask])
            self.setBGTS(self.bg_ts[mask])
            
        if self.smooth_counts is not None:
            mask = (self.smooth_ts > min_t) & (self.smooth_ts < max_t)
            self.setSmoothCounts(self.smooth_counts[mask])
            self.setSmoothTS(self.smooth_ts[mask])
        
        if self.show_plots:
            self.plot_data()
        
        print(f"Time cut applied: ({min_t}, {max_t})")
    
    def find_signal_start(self,counts,ts):
        
        noise_window = self.noise_window
        
        stdevs = np.zeros_like(counts)
        means = np.zeros_like(counts)
        for i in range(noise_window//2,counts.shape[0]-noise_window//2):
            sample = counts[i-noise_window//2:i+noise_window//2]
            stdevs[i+noise_window//2] = np.std(sample)
            means[i+noise_window//2] = np.mean(sample)
    
        gradient = np.diff(stdevs)
        gradient[:noise_window] = 0
        gradient[-noise_window:] = 0
        idt = np.argmax(gradient)
        
        t0 = ts[idt-noise_window]
        
        idt1 = np.argmax(counts)
        t1 = ts[idt1] # If we want to include the pulse peak_time time
    
        # t0 = np.mean([t0,t1])
        
        baseline = np.mean(counts[:idt-noise_window+1])
        
        return t0, t1, baseline
    
    def find_baselines(self):
        t0,t1,baseline = self.find_signal_start(self.counts,self.ts)
        
        self.setSignalBaseline(baseline)
        self.setSignalStart(t0)
        self.setSignalPeakTime(t1)
        
        if self.bg_counts is not None:
            t0,t1,baseline = self.find_signal_start(self.bg_counts,self.bg_ts)
            self.setBGBaseline(baseline)
            self.setBGStart(t0)
            self.setBGPeakTime(t1)
            
        if self.show_plots:
            self.plot_data()
    
    def baseline_correction(self):
        if self.signal_baseline is not None:
            counts = self.counts - self.signal_baseline
            #counts[counts<0] = 0
            self.setCounts(counts)
            self.setSignalBaseline(0)
        else:
            self.find_baselines()
            return self.baseline_correction()
        
        if self.bg_baseline is not None:
            counts = self.bg_counts - self.bg_baseline
            counts[counts<0] = 0
            self.setBGCounts(counts)
            self.setBGBaseline(0)
            
        if self.show_plots:
            self.plot_data()
        
        print("Baseline correction complete")
            
    def start_time_correction(self, by = "start"):
        if self.signal_start is not None:
            if by == "start":
                ts = self.ts - self.signal_start
                self.setSignalPeakTime(self.signal_peak_time - self.signal_start)    
                self.setSignalStart(0)
            else:
                ts = self.ts - self.signal_peak_time
                self.setSignalStart(self.signal_start - self.signal_peak_time)
                self.setSignalPeakTime(0)
                
            self.setTS(ts)
        else:
            self.find_baselines()
            return self.start_time_correction()
        
        if self.bg_start is not None:
            if by == "start":
                ts = self.bg_ts - self.bg_start
                self.setBGPeakTime(self.bg_peak_time - self.bg_start)
                self.setBGStart(0)
            else:
                ts = self.bg_ts - self.bg_peak_time
                self.setBGStart(self.bg_start - self.bg_peak_time)
                self.setBGPeakTime(0)
                
            self.setBGTS(ts)
            
        if self.bg_counts is not None:
            self.apply_cut(max([self.ts[0],self.bg_ts[0]]),min([self.ts[-1],self.bg_ts[-1]]))
            
        if self.show_plots:
            self.plot_data()
            
        print("Start time correction complete.")
        
    def subtract_background(self):
        if self.bg_counts is None:
            print("No background data to subtract.")
            return
        endpoint = min([len(self.counts), len(self.bg_counts)])
        counts = self.counts[:endpoint] - self.bg_counts[:endpoint]
        counts[counts < 0] = 0
        self.setCounts(counts)
        self.setTS(self.ts[:endpoint])
        self.setBGCounts(self.bg_counts[:endpoint])
        self.setBGTS(self.bg_ts[:endpoint])
        
        if self.show_plots:
            self.plot_data()
            
        print("Background subtraction complete.")
            
    def reset_state(self):
        self.setShowPlots(False)
        self.log_y_scale = False
        self.setNormalised(False)
        self.setSmoothCounts(None)
        self.setSmoothTS(None)
        self.best_fit = None
        self.best_ts = None
        self.bin_data()
        print("State reset.")
        
    def calc_moving_average(self, window_size = 11):
        self.setSmoothCounts(np.convolve(self.counts, np.ones(window_size)/window_size, mode='valid'))
        self.setSmoothTS(np.convolve(self.ts, np.ones(window_size)/window_size, mode='valid'))
        
        if self.show_plots:
            self.plot_data
        
        print("Moving average calculation complete.")
    
    def fit_data(self, kwargs):     
        kwargs["baseline"] = self.signal_baseline
        obj_val, pos, y_best = run_optuna_trial(self.counts, self.ts, 0, **kwargs)
        
        self.best_fit_score = obj_val
        self.best_fit = y_best[0]
        self.best_pos = pos
        self.best_ts = self.ts

def handle_settings_menu(da):
    def print_flags():
        print("\n--- Current Flags ---")
        print(f"show_plots     : {da.getShowPlots()}")
        print(f"normalised     : {da.getNormalised()}")
        print(f"plot_bg        : {da.getPlotBG()}")
        print(f"log_y_scale    : {da.getPlotLogYscale()}")
        print(f"noise_window   : {da.getNoiseWindow()}")
        print(f"max_time       : {da.getMaxTime()}")
        print(f"cmap      : {da.getCmapName()}")

    while True:
        print_flags()
        print("\nSet a flag or value. Format: <key> <value>")
        print("Available keys: plot_bg, show_plots, normalised, log_y, noise_window, max_time,cmap")
        print("Type 'b' to go back.")
        cmd = input(">>> ").strip().lower()

        if cmd == "b":
            break

        parts = cmd.split()
        if len(parts) != 2:
            print("Invalid input. Use format: key value")
            continue

        key, val = parts
        try:
            if key == "plot_bg":
                da.setPlotBG(val.lower() in ["1", "true", "yes", "y"])
            elif key == "show_plots":
                da.setShowPlots(val.lower() in ["1", "true", "yes", "y"])
            elif key == "normalised":
                da.setNormalised(val.lower() in ["1", "true", "yes", "y"])
            elif key == "log_y_scale":
                da.setPlotLogYScale(val.lower() in ["1", "true", "yes", "y"])
            elif key == "noise_window":
                da.setNoiseWindow(int(val))
            elif key == "max_time":
                da.setMaxTime(float(val))
            elif key == "cmap":
                da.setCmapName(val)
                print(f"✅ Colormap updated to '{val}'")
            else:
                print("Unknown key.")
        except Exception as e:
            print(f"Error updating setting: {e}")

def run_default_analysis(da, use_defaults = True):
    print("\n=== Running Default Analysis Sequence (b c s x w a f) ===")
    
    # Use defaults?
    try:
        use_defaults = str(input("Use default values?")).lower() in ["1","y","yes","true"]
    except:
        print("Invalid input. Not using defaults")
    
    if use_defaults:
        nw = 11
    else:        
        # --- 1. Prompt for noise window ---
        try:
            nw = int(input("Noise window size: "))
            da.setNoiseWindow(nw)
        except:
            print("Invalid input. Using default noise window = 10")
            da.setNoiseWindow(10)

    # --- 2. Baseline find & correction ---
    da.find_baselines()
    da.baseline_correction()
    da.normalise_data()
    
    if use_defaults:
        correct_by = "start"
    else:
        # --- 3. Start time correction ---
        correct_by = input("Correct start time by 'start' or 'peak': ").strip().lower()
        if correct_by not in ["start", "peak"]:
            print("Invalid input. Defaulting to 'start'")
            correct_by = "start"
     
    da.start_time_correction(by=correct_by)
    # --- 4. Subtract background ---
    da.subtract_background()
    
    if use_defaults:
        t_min, t_max = -10, 60
    else:
        # --- 5. Time window cut ---
        try:
            t_min, t_max = map(float, input("Time window (t_min t_max): ").split())
        except:
            print("Invalid input. Using default window (-10, 60)")
            t_min, t_max = -10, 60
            
    da.apply_cut(t_min, t_max)
    da.normalise_data()
    
    if use_defaults:
        ws = 20
    else:
        # --- 6. Moving average ---
        try:
            ws = int(input("Moving average window size: "))
        except:
            print("Invalid input. Using default = 20")
            ws = 20
    da.calc_moving_average(window_size=ws)
        
    kwargs = {
        "td_min": 0.1,
        "td_max": 200,
        "tr_min": 1e-3,
        "tr_max": 0.50,
        "iterations": 500,
        "n_trials": 50,
        "n_exponentials": 2,
        "trial_sampling": 20,
        "obj_val_type": "mse"
    }
    
    choose_parms_and_run_fit(da, kwargs)
        

def choose_parms_and_run_fit(da, kwargs):
    while True:
        param = input("Set fit param (key val), 's' to start fit, 'p' to print, 'b' to back: ").strip()
        if param == 's':
            da.find_baselines()
            da.fit_data(kwargs)
            da.plot_data()
            break
        elif param == 'p':
            for key, val in kwargs.items():
                print(key,"\t: ",val)
        elif param == 'b':
            break
        else:
            try:
                key, val = param.split()
                if key in kwargs:
                    if key in ['iterations', 'n_trials', 'n_exponentials', 'trial_sampling']:
                        kwargs[key] = int(val)
                    elif key == "obj_val_type":
                        if val in ['mse','log_mse','chi_squared']:
                            kwargs[key] = val
                        else:
                            print("Invalid objective value type. Choose one of:\n")
                            for t in ['mse','log_mse','chi_squared']:
                                print(t)
                    else:
                        kwargs[key] = float(val)
                                
                else:
                    print("Unknown parameter.")
            except:
                print("Invalid input format.")
                
def prompt_for_files():
    print("\n=== TCSPC File Selection ===")
    
    signal_file = input("Enter path to SIGNAL file (required): ").strip()
    while not os.path.isfile(signal_file):
        print("Invalid file path. Try again.")
        signal_file = input("Enter path to SIGNAL file (required): ").strip()
    
    background_file = input("Enter path to BACKGROUND file (optional): ").strip()
    if background_file == "":
        background_file = None
    elif not os.path.isfile(background_file):
        print("Background file not found. Continuing without background.")
        background_file = None
    
    max_time = 0
    while max_time == 0:
        try:
            max_time = float(input("Enter max time in ns during acquisition: ").strip())
            if max_time == 0:
                print("Must be positive")
        except:
            print("Must be float in ns")
    
    da = DecayAnalyser(
        signal_file, 
        background_file, 
        cmap_name="copper",
        max_time=max_time,
        normalised=False,
        noise_window=10,
        line_alpha=0.25,
        n_colors=5,
        log_y=False,
        show_plots=False
    )
    
    return da

def save_plot_svg(da):
    filename = input("Filename to save as (without extension): ").strip()
    if not filename:
        print("⚠️ No filename provided. Skipping.")
        return

    try:
        da.plot_data(save_path=f"{filename}.svg")
        print(f"✅ Plot saved as {filename}.svg")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")
        
def show_main_menu():
    print("\nDecay Analyser Menu:")
    print("[d] Run default analysis sequence (b c s x w a f)")
    print("[b] Find baselines")
    print("[c] Apply baseline correction")
    print("[s] Start time correction")
    print("[x] Subtract background")
    print("[a] Moving average")
    print("[w] Apply time window cut")
    print("[f] Fit decay data")
    print("[r] Reset state")
    print("[p] Plot current state")
    print("[v] Save current plot as .svg")
    print("[g] Settings")
    print("[m] Menu")
    print("[q] Quit")
    print("[i] Input new files")

def interactive_decay_analysis():
    da = prompt_for_files()
    kwargs = {
        "td_min": 0.1,
        "td_max": 1000,
        "tr_min": 1e-3,
        "tr_max": 0.10,
        "iterations": 500,
        "n_trials": 50,
        "n_exponentials": 2,
        "trial_sampling": 20,
        "obj_val_type": "mse"
    }
    show_main_menu()
    while True:
        
        choice = input("Select an option (or key shortcut): ").strip().lower()

        if choice == "b":
            try:
                nw = int(input("Enter noise window size: "))
                da.setNoiseWindow(nw)
                da.find_baselines()
                da.plot_data()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "v":
            save_plot_svg(da)
    
        elif choice == "i":
            da = prompt_for_files()
            print("Files loaded and analyser reset.")
            
        elif choice == "d":
            run_default_analysis(da)
            
        elif choice == "c":
            try:
                da.baseline_correction()
                da.normalise_data()
                da.plot_data()
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "s":
            try:
                correct_by = input("Correct by 'start' or 'peak': ").strip()
                da.start_time_correction(by=correct_by)
                da.plot_data()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "x":
            da.subtract_background()
            da.plot_data()

        elif choice == "a":
            try:
                window_size = int(input("Enter moving average window size: "))
                da.calc_moving_average(window_size=window_size)
                da.plot_data()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "w":
            try:
                t_min, t_max = map(float, input("Enter t_min and t_max (space-separated): ").split())
                da.apply_cut(t_min, t_max)
                da.normalise_data()
                da.plot_data()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "f":
            choose_parms_and_run_fit(da, kwargs)
        
        elif choice == "r":
            da.reset_state()
            print("State reset.")
        
        elif choice == "p":
            da.plot_data()
        
        elif choice == "q":
            print("Exiting...")
            break
        elif choice == "g":
            handle_settings_menu(da)
        elif choice == "m":
            show_main_menu()
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    interactive_decay_analysis()