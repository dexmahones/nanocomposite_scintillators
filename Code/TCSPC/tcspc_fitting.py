import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows
from scipy.stats import norm
import pandas as pd
import pyswarms as ps
import optuna
from pyswarms.single.global_best import GlobalBestPSO
import glob

# Single exponential component in scintillation sum
def scintillation_component(t, td, tr, rho):
    t_safe_td = np.clip(t / td, 0, 700)  # 700 is safe; exp(-700) ≈ 0
    t_safe_tr = np.clip(t / tr, 0, 700)  # Could probably be handled better...

    sig = (np.exp(-t_safe_td)-np.exp(-t_safe_tr))/(td-tr)*rho
    return sig

# Sum up exponential components
def scintillation_pulse(parms, t):
    # input parms as td1, tr1, rho1, td2, tr2, rho2 etc...

    tds = parms[:,0::3]
    trs = parms[:,1::3]
    rhos = parms[:,2::3]

    f = np.zeros((tds.shape[0],len(t)))
    t = f.copy() + t

    for i in range(tds.shape[1]):
        td = tds[:,i]
        tr = trs[:,i]
        rho = rhos[:,i]
        f+=scintillation_component(t,td[:,np.newaxis],tr[:,np.newaxis],rho[:,np.newaxis])
    f[f<=0] = 0
    return f

# Instrument response function stand-in
def irf(t, fwhm):
    irf_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    irf = norm.pdf(t, loc=0, scale=irf_sigma)
    irf /= sum(irf)

    return irf

# Finding time offset
def time_offset(counts, ts, noise_window = 11, plot = False, criterion = "grad"):    
    stdevs = np.zeros_like(counts)
    means = np.zeros_like(counts)
    for i in range(noise_window//2,counts.shape[1]-noise_window//2):
        sample = counts[:,i-noise_window//2:i+noise_window//2]
        stdevs[:,i+noise_window//2] = np.std(sample, axis = 1)
        means[:,i+noise_window//2] = np.mean(sample, axis = 1)

    gradient = np.diff(stdevs, axis = 1)
    gradient[:,:noise_window] = 0
    gradient[:,-noise_window:] = 0
    idc,idt = np.where(gradient==np.max(gradient, axis = 1)[:,np.newaxis])
    
    t0 = ts[idt-noise_window//2]

    idt1 = np.argmax(counts, axis = 1)
    t1 = ts[idt1] # If we want to include the pulse peak time

    # t0 = np.mean([t0,t1])
    
    baselines = [np.mean(row[:i+1]) for row, i in zip(counts, idt-20)]

    baseline = np.array(baselines)[:,np.newaxis]

    if plot:
        fig, axes = plt.subplots(2,1, figsize = (10,10))
        
        axes[0].plot(ts,counts.T, "k", lw = 1, alpha = 0.5)

        axes[1].plot(ts[1:],np.diff(counts).T, "k", lw = 1, alpha = 0.5)
        axes[0].hlines([baseline],0,ts[-1], "b",ls = "--",lw = 1, alpha = 0.95)
        axes[1].hlines([baseline],0,ts[-1], "b",ls = "--",lw = 1, alpha = 0.95)
        axes[0].vlines(ts[idt],0,np.max(counts.flatten()), 'r',ls = "--", lw = 1, alpha = 0.95)
        axes[1].vlines(ts[idt],0,np.max(np.diff(counts).flatten()), 'r',ls = "--", lw = 1,alpha = 0.95)
        plt.show()

    # decay_counts[0] = 0 # For some reason set start and end to zero...
    # decay_counts[-1] = 0

    return t0, baseline

# Cost value for particle swarm optimization
def obj_func(parms, t_data, y_data, irf, baseline, obj_val_type):
        dt = t_data[1]-t_data[0]
        rhos = parms[:,2::3]
        sum_rhos = np.sum(rhos, axis = 1)
        parms[:,2::3] = np.divide(parms[:,2::3],sum_rhos[:,np.newaxis], out = np.ones_like(rhos)/rhos.shape[0], where = (sum_rhos[:,np.newaxis]!=0))

        y_fit = scintillation_pulse(parms,t_data)
        y_fit += baseline
        # y_fit = convolve(y_fit,irf[np.newaxis,:], mode = 'same') 
        y_fit /= (np.sum(y_fit,axis = 1)[:,np.newaxis]*dt)
        # y_fit[y_fit<=0] = baseline
        
        # y_fit *= total_counts

        if obj_val_type == "mse":
                return np.mean((y_data-y_fit)**2,axis = 1)
        elif obj_val_type == "log_mse":
                return np.mean(-1/np.log((y_data-y_fit)**2),axis = 1)
        elif obj_val_type == "chi_squared":
                sigmas = np.sqrt(np.abs(y_data))
                val = np.divide((y_data-y_fit)**2,np.sqrt(sigmas),  out=np.zeros_like(y_fit, dtype=float), where = (sigmas!=0))
                # val = np.divide((y_data-y_fit)**2,sigmas,  out=np.ones_like(y_fit, dtype=float), where = (sigmas!=0))
                return np.sum(val, axis = 1)
        else:
                return None

# Particle swarm optimization
def fit_model(t_data, y_data, irf, n_exponentials = 3, n_particles = 10, iterations = 1000, options = {}, bounds = (), baseline = 1e-3, obj_val_type = "mse", plot = False, verbose = False):
        dt = t_data[1]-t_data[0]
        optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=3*n_exponentials, options=options, bounds=bounds)
        obj_val, pos = optimizer.optimize(obj_func, iterations, t_data = t_data, y_data = y_data, irf = irf, baseline = baseline, obj_val_type = obj_val_type, verbose=verbose)

        if verbose:
                print()
                print("td: \t","\t".join(np.round(pos[0::3],2).astype(str)))
                print("tr: \t","\t".join(np.round(pos[1::3],2).astype(str)))
                print("P: \t","\t".join(np.round(pos[2::3],2).astype(str)))
                print()

        y_best = scintillation_pulse(pos[np.newaxis,:],t_data) 
        y_best += baseline
        # y_best = convolve(y_best,irf[np.newaxis,:], mode = 'same')
        y_best /= (np.sum(y_best,axis = 1)[:,np.newaxis]*dt)
        # y_best[y_best<=0] = baseline
        
        # y_best *= total_counts

        return obj_val, pos, y_best

# Improve with hyperparameter optimization
# Possibly implement hyperbandit pruning
# Also implement early stopping in the PSO
def create_optuna_objective(y_fit_data, t_data, instr_resp,iterations, baseline = 0, td_min = 0, td_max = 0, tr_min = 1e-3, tr_max = 0, rho_min = 0, rho_max = 1, n_exponentials = 2, obj_val_type =  "chi_squared"):
    def optuna_objective(trial):
        # Suggest values for c1, c2, w, and swarm size
        c1 = trial.suggest_float("c1", 0.01, 1.)
        c2 = trial.suggest_float("c2", 0.01, 1.)
        w = trial.suggest_float("w", 0.01, 1.)
        n_particles = trial.suggest_int("n_particles", 5, 50)

        # If bounds not specified, include them in hyperparameters
        tr_max_trial, td_min_trial, td_max_trial = (0,0,0)
        if tr_max == 0:
            tr_max_trial = trial.suggest_float("tr_max",tr_min+1e-2,tr_min*10+0.1)
        else:
            tr_max_trial = tr_max

        if td_min == 0:
            td_min_trial = trial.suggest_float("td_min",tr_max+1,tr_max+10)
        else:
            td_min_trial = td_min

        if td_max == 0:
            td_max_trial = trial.suggest_float("td_max",td_min+1,(td_min+1)*100)
        else:
            td_max_trial = td_max

        # In case bounds are to be included in hyperparameter optimization
        x_max = np.tile([td_max_trial,tr_max_trial,rho_max],n_exponentials)
        x_min = np.tile([td_min_trial,tr_min,rho_min],n_exponentials)

        bounds = (x_min, x_max)

        # n_particles = 1
        # n_exponentials = 2
        options = {'c1': c1, 'c2': c2, 'w': w}

        # Run model fitting
        obj_val, pos, y_best =  fit_model(
            t_data,
            y_fit_data,
            instr_resp,
            n_exponentials = n_exponentials, 
            n_particles = n_particles,
            iterations = iterations,
            options = options,
            bounds = bounds,
            baseline = baseline,
            obj_val_type = obj_val_type,
            verbose = False)

        return obj_val
    return optuna_objective

def run_optuna_trial(count_data,ts,instr_resp,baseline = 0, n_exponentials = 2, iterations = 1000, n_trials = 20, trial_scint_constants = False, td_min = 0.1, td_max = 200, tr_min = 1e-3, tr_max = 0.01,trial_sampling = 5, obj_val_type =  "chi_squared"):
    # Sparse data for fast searching
    y_sparse = count_data[0::trial_sampling].copy()
    t_sparse = ts[0::trial_sampling].copy()

    y_sparse /= (sum(y_sparse)*(t_sparse[1]-t_sparse[0]))

    rho_max = 1
    rho_min = 0

    if trial_scint_constants:
        objective = create_optuna_objective(
            t_data=t_sparse,
            y_fit_data=y_sparse,
            instr_resp=instr_resp,
            iterations=iterations,
            n_exponentials = n_exponentials,
            baseline=baseline,
            td_min=0, td_max=0,
            tr_min=tr_min, tr_max=0,
            rho_min=0.0, rho_max=1.0,
            obj_val_type = obj_val_type
        )
    else:
            objective = create_optuna_objective(
            t_data=t_sparse,
            y_fit_data=y_sparse,
            instr_resp=instr_resp,
            iterations=iterations,
            baseline=baseline,
            n_exponentials = n_exponentials,
            td_min=td_min, td_max=td_max,
            tr_min=tr_min, tr_max=tr_max,
            rho_min=0.0, rho_max=1.0,
            obj_val_type = obj_val_type
        )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best score:", study.best_value)
    print("Best parameters:", study.best_params)

    best_options = study.best_params
    best_n_particles = best_options.pop("n_particles")

    try:
        best_td_min = best_options.pop("td_min")
    except:
        best_td_min = td_min

    try:
        best_td_max = best_options.pop("td_max")
    except:
        best_td_max = td_max

    try:
        best_tr_max = best_options.pop("td_min")
    except:
        best_tr_max = tr_max

    x_max = np.tile([best_td_max,best_tr_max,rho_max],n_exponentials)
    x_min = np.tile([best_td_min,tr_min,rho_min],n_exponentials)

    bounds = (x_min, x_max)

    obj_val, pos, y_best =  fit_model(
        ts, count_data,
        instr_resp,
        n_exponentials = n_exponentials, 
        n_particles = best_n_particles,
        iterations = iterations,
        options = best_options,
        bounds = bounds,
        baseline = baseline,
        obj_val_type = obj_val_type,
        verbose = True)
    
    return obj_val, pos, y_best