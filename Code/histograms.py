import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from plot_utils import PlotStyleManager
import json
import os
import seaborn as sns
import plotly 
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

PSM = PlotStyleManager(n_colors = 5)

def input_metadata(file_path = None):
    # Define the list of metadata fields required
    fields = [
        "sipm", "duration", "scintillator", "measurement", 
        "bias", "trigger", "channel", "gating",
        "coolingV", "coolingA"
    ]
    
    metadata = {}
    print("--- Metadata Entry ---")
    
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
            if normalize:
                df[df.columns[1]] /= df[df.columns[1]].max()
            # print("CSV successfully imported!")
            meta_path = file_path.replace(".csv",".json")
            if not os.path.exists(meta_path):
                input_metadata(meta_path)
            return df, file_path
        except Exception as e:
            print(f"Error importing CSV: {e}")
            return None, file_path
    else:
        print("No file selected.")
        return None

def load_spectrum_folder(normalize = False):
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
            df,_ = load_file(os.path.join(folder_path,file),normalize = False)
            df["label"] = file.split(".")[0]
            dfs.append(df)
    return pd.concat(dfs), folder_path

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
        plt.show()
        
    elif backend == "plotly":
        fig = px.line(df, x=cols[0], y=cols[1], color="label", template = "simple_white", line_shape="hvh")
        fig.write_html(os.path.join(folder_path,"spectra.html"), include_plotlyjs="cdn")
        fig.show()

if __name__ == "__main__":
    # df,dir = load_file()
    df,dir = load_spectrum_folder(normalize = False)

    # df = filter_x(df, xmin = 1e-9, xmax = 2e-8)
    title = " ".join(dir.split("/")[::-1][:2][::-1]).split(".")[0]

    show_histogram(df, xlabel = "Charge (V*s)", ylabel = "Counts (a.u)", title = title, folder_path=dir)