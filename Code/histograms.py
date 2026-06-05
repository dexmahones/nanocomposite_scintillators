import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from plot_utils import PlotStyleManager
import json
import re
import os
import seaborn as sns
import plotly 
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

PSM = PlotStyleManager(n_colors = 4, cmap_name= "copper")

def input_metadata(file_path = None):
    # Define the list of metadata fields required
    fields = [
        "sipm", "duration", "scintillator", "measurement", 
        "bias", "trigger", "channel", "gating",
        "coolingV", "coolingA",
        "label"
    ]
    
    metadata = {}
    print(f"--- {file_path} ---")
    
    # Iterate through fields and prompt the user for input
    for field in fields:
        user_input = input(f"Enter {field}: ")
        metadata[field] = user_input

    # Save the dictionary to a JSON file with indentation for readability
    try:
        with open(file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        print(f"\nMetadata successfully saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def load_metadata(meta_path):
    with open(meta_path,"r") as infile:
        meta = json.load(infile)
    return meta

def load_file(file_path = None, normalize = True):
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
            meta_path = file_path.replace(".csv",".json")
            if not os.path.exists(meta_path):
                input_metadata(meta_path)
            meta_data = load_metadata(meta_path)
            if normalize:
                try:
                    df[df.columns[0]] *= 1e9
                    duration = float(re.findall(r"-?\d+\.?\d*", meta_data['duration'])[0])
                    num_bins = len(df[df.columns[0]])
                    xrange = df[df.columns[0]].max() - df[df.columns[0]].min()
                    df[df.columns[1]] = df[df.columns[1]] / duration * num_bins / xrange
                    # (df[df.columns[1]].sum() * xrange/num_bins)
                except:
                    print("Could not normalize")
            return df, file_path, meta_data
        except Exception as e:
            print(f"Error importing CSV: {e}")
            return None, file_path, None
    else:
        print("No file selected.")
        return None

def load_spectrum_folder(normalize = True, num_bins = 220, min_x = 0, max_x = 4000):
    dfs = []

    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open folder browser
    folder_path = filedialog.askdirectory(
        title="Select folder",
    )
    if folder_path:
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        for j,file in enumerate(files):
            df,_,meta_data = load_file(os.path.join(folder_path,file),normalize = normalize)
            df = df[(df[df.columns[0]]<=max_x)&(df[df.columns[0]]>=min_x)]
            
            new_bins = np.linspace(df[df.columns[0]].min(), df[df.columns[0]].max(), num_bins-1)
            # Map x values to the new bins
            # df['new_bins'] = pd.cut(df[df.columns[0]], bins=new_bins).apply(lambda x: x.left)
            
            # Group by the new bins and sum the y values
            # rebinned_df = df.groupby('new_bins', observed=False)[df.columns[1]].sum().reset_index()
            df["label"] = meta_data["label"]
            dfs.append(df)
    data = pd.concat(dfs)
    

    return data, folder_path

def filter_x(df, xmin = None, xmax = None):
    if xmin:
        df = df[df.iloc[:,0]>=xmin]
    if xmax:
        df = df[df.iloc[:,0]<=xmax]
    return df

def show_histogram(df, title = None, xlabel = None, ylabel = None, backend = "plotly", folder_path = ""):
    cols = df.columns

    if backend == "seaborn":
        sns.relplot(
            df,
            x = cols[0],
            y = cols[1],
            hue="label",
            kind="line", 
            drawstyle="steps-mid"
        )
        xlabel = xlabel if xlabel else cols[0]
        ylabel = ylabel if ylabel else cols[1]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid("on")
        # plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    elif backend == "plotly":
        df = df.rename(columns = {
            df.columns[0]:xlabel,
            df.columns[1]:ylabel
        })
        fig = px.line(df, x=xlabel, y=ylabel, color="label", template = "simple_white", line_shape="hvh")
        fig.write_html(os.path.join(folder_path,"spectra.html"), include_plotlyjs="cdn")
        fig.show()

if __name__ == "__main__":
    # df,dir = load_file()
    df,dir = load_spectrum_folder(normalize = False)
    # df = filter_x(df, xmin = 1e-9, xmax = 2e-8)
    title = " ".join(dir.split("/")[::-1][:2][::-1]).split(".")[0]

    show_histogram(df, xlabel = "Charge (V*s)", ylabel = "Counts", title = title, folder_path=dir, backend = 'seaborn')