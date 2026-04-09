import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows
from scipy.stats import norm
import pandas as pd
import pyswarms as ps
import optuna
from pyswarms.single.global_best import GlobalBestPSO
import glob
from plot_utils import set_plot_style
from tcspc_fitting import run_optuna_trial

class DecayAnalyser:
    def __init__(self, signal_path, background_path=None, max_time = None, normalised = False, noise_window = 4, show_plots = False, log_y = True, line_alpha = 0.75, cmap_name = 'jet', n_colors = 2):
        self.colors = set_plot_style(cmap_name=cmap_name, n_colors=n_colors)
        
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
        self.plot_bg = True
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
        return self.plot_bg

    def setPlotLogYScale(self,val):
        self.plot_bg = val
        
    def normalise_data(self):
        self.setNormalised(True)
        if self.counts is not None:
            self.setCounts(self.counts/(sum(self.counts)*(self.ts[1]-self.ts[0])))
        if self.bg_counts is not None:
            self.setBGCounts(self.bg_counts/(sum(self.bg_counts)*(self.bg_ts[1]-self.bg_ts[0])))
        if self.show_plots:
            self.plot_data()
        
    def read_datafile(self, file_path):
        if not file_path:
            return None
        with open(file_path, "r") as infile:
            events = np.array(infile.read().split()).astype(float)
        return events
    
    def plot_data(self):
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
            plt.ylim(1e-4,max(self.counts))
        plt.legend()
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

    def apply_cut(self, min_t, max_t):
        mask = (self.ts >= min_t) & (self.ts <= max_t)
        self.setCounts(self.counts[mask])
        self.setTS(self.ts[mask])
        
        if self.bg_counts is not None:
            mask = (self.bg_ts >= min_t) & (self.bg_ts <= max_t)
            self.setBGCounts(self.bg_counts[mask])
            self.setBGTS(self.bg_ts[mask])
            
        if self.smooth_counts is not None:
            mask = (self.smooth_ts >= min_t) & (self.smooth_ts <= max_t)
            self.setSmoothCounts(self.smooth_counts[mask])
            self.setSmoothTS(self.smooth_ts[mask])
        
        if self.show_plots:
            self.plot_data()
    
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
            
            
        self.apply_cut(max([self.ts[0],self.bg_ts[0]]),min([self.ts[-1],self.bg_ts[-1]]))
            
        if self.show_plots:
            self.plot_data()
        
    def subtract_background(self):
        if self.bg_counts is None:
            print("No background data to subtract.")
            return
        counts = self.counts - self.bg_counts
        counts[counts < 0] = 0
        self.setCounts(counts)
        
        if self.show_plots:
            self.plot_data()
            
    def reset_state(self):
        self.setShowPlots(True)
        self.setNormalised(False)
        self.bin_data()
        
    def calc_moving_average(self, window_size = 11):
        self.setSmoothCounts(np.convolve(self.counts, np.ones(window_size)/window_size, mode='valid'))
        self.setSmoothTS(np.convolve(self.ts, np.ones(window_size)/window_size, mode='valid'))
        
        if self.show_plots:
            self.plot_data
    
    def fit_data(self, kwargs):     
        kwargs["baseline"] = self.signal_baseline
        obj_val, pos, y_best = run_optuna_trial(self.counts, self.ts, 0, **kwargs)
        
        self.best_fit_score = obj_val
        self.best_fit = y_best[0]
        self.best_pos = pos
        self.best_ts = self.ts
        
if __name__ == "__main__":
    
    da = DecayAnalyser(
    "TCSPC/CdS_RT_2_3A_500kHz_200ns_no_filter.txt", 
    "TCSPC/CdS_RT_background.txt", 
    cmap_name = "bone",
    max_time = 200,
    normalised = False,
    noise_window = 10,
    line_alpha = 0.25,
    n_colors=5,
    log_y = False,
    show_plots = False
    )
    
    done = "n"
    while done!="y":
        # try:
        nw = int(input("Noise window size for rising pulse trigger: "))

        da.setNoiseWindow(nw)

        da.find_baselines()

        da.plot_data()
        done = input("Apply baseline correction and move on? (y/n)")
        # except:
            # print("Please type a integer")

        if done == "y":
            # Apply baseline correction
            try:
                da.baseline_correction()
                da.normalise_data()
                da.plot_data()
            except:
                    done = "n"
                    print("BG-signal size mismatch. Try a different window size.")

    # Start time correction
    while True:
        done = "n"
        try:
            correct_start_by = input("Correct start time by start or peak? (start/peak)")
            da.start_time_correction(by = correct_start_by)
            da.subtract_background()
            da.start_time_correction(by = correct_start_by)
            da.normalise_data()
            da.plot_data()
            done = input("Move to next step? (y/n)")
        except:
            print("Invalid input\n")
        
        if done[0].lower() == "y":
            break
    
    # Moving average window size
    while True:
        done = "n"
        try:
            ws = int(input("Moving average window size: "))
        except:
            print("Invalid input\n")
        
        try:
            da.calc_moving_average(window_size=ws)
            da.plot_data()
            done = input("Move to next step? (y/n)")
        except:
            print("Window size invalid")
        
        if done[0].lower() == "y":
            break
        else:
            da.undoSetCounts()
        
    # Select time window
    while True:
        done = "n"
        t_min = 0 
        t_max = 0
        
        try:
            t_min, t_max = (int(i) for i in input("Time window (t_min t_max): ").split(" "))
            print(t_min,t_max)
        except:
            print("Invalid input")
            
        if t_min >= t_max:
            print(f"Invalid input: {t_min},{t_max}")
        else:
            try:
                da.apply_cut(t_min,t_max)
                da.normalise_data()
                da.plot_data()
                done = input("Move to next step? (y/n)")
            except:
                print(f"Invalid input: {t_min},{t_max}")
            
            if done[0].lower() == "y":
                break
            else:
                da.undoSetCounts()
                da.plot_data()
    
    # Fit data to exponentials
    kwargs = {
            "td_min":0.1,
            "td_max":1000,
            "tr_min":1e-3,
            "tr_max":0.10,
            "iterations":500,
            "n_trials":50,
            "n_exponentials":2,
            "trial_sampling":20,
            "obj_val_type":"mse"
        }
    while True:
        pair = input("Set parameter (parm val). Type s to stop, p for list of parameters. ")
        if pair == "s":
            break
        elif pair == "p":
            print(kwargs.items())
            print("\n".join(["\t".join(list(key_val)) for key_val in zip(kwargs.items())]))
                  
        elif len(pair.split(" ")) > 1:
            key, val = pair.split(" ")
            if key in kwargs.keys():
                kwargs[key] = float(val)
        else:
            print("Invalid input")        
    
    da.setPlotBG(False)
    da.find_baselines()
    da.fit_data()
    da.plot_data()