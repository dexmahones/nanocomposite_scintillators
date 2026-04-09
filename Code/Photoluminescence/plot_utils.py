import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

class PlotStyleManager:
    def __init__(self, cmap_name='viridis', n_colors=5, alpha_default=0.8):
        self.cmap_name = cmap_name
        self.n_colors = n_colors
        self.alpha_default = alpha_default
        self.cmap = None

        # Default rcParams
        self.figsize = (7, 7)
        self.dpi = 100
        self.grid = False
        self.font_size = 12
        self.label_size = 12
        self.title_size = 14
        self.legend_fontsize = 10
        self.tick_labelsize = 10
        self.linewidth = 1
        self.markersize = 6
        self.savefig_dpi = 300
        self.savefig_bbox = "tight"

        self.custom_colors = []
        self.apply_style()

    def apply_style(self):
        cmap = plt.get_cmap(self.cmap_name)
        cmap.set_bad(color='white')
        self.cmap = cmap.copy()

        self.custom_colors = [cmap(i) for i in np.linspace(0, 1, self.n_colors)]

        plt.rcParams.update({
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            "axes.grid": self.grid,
            "font.size": self.font_size,
            "axes.labelsize": self.label_size,
            "axes.titlesize": self.title_size,
            "legend.fontsize": self.legend_fontsize,
            "xtick.labelsize": self.tick_labelsize,
            "ytick.labelsize": self.tick_labelsize,
            "lines.linewidth": self.linewidth,
            "lines.markersize": self.markersize,
            "axes.prop_cycle": cycler('color', self.custom_colors),
            "savefig.dpi": self.savefig_dpi,
            "savefig.bbox": self.savefig_bbox,
        })

    # Setter methods
    def set_cmap(self, cmap_name):
        self.cmap_name = cmap_name
        self.apply_style()

    def get_cmap(self):
        return self.cmap
    
    def set_n_colors(self, n_colors):
        self.n_colors = n_colors
        self.apply_style()

    def set_alpha(self, alpha_default):
        self.alpha_default = alpha_default  # Stored for external use

    def set_linewidth(self, val):
        self.linewidth = val
        self.apply_style()

    def set_markersize(self, val):
        self.markersize = val
        self.apply_style()

    def set_figsize(self, width, height):
        self.figsize = (width, height)
        self.apply_style()

    def set_dpi(self, dpi):
        self.dpi = dpi
        self.apply_style()

    def set_grid(self, show_grid):
        self.grid = show_grid
        self.apply_style()

    def set_font_size(self, size):
        self.font_size = size
        self.apply_style()

    def set_label_size(self, size):
        self.label_size = size
        self.apply_style()

    def set_title_size(self, size):
        self.title_size = size
        self.apply_style()

    def set_legend_fontsize(self, size):
        self.legend_fontsize = size
        self.apply_style()

    def set_tick_labelsize(self, size):
        self.tick_labelsize = size
        self.apply_style()

    def set_savefig_dpi(self, dpi):
        self.savefig_dpi = dpi
        self.apply_style()

    def set_savefig_bbox(self, bbox):
        self.savefig_bbox = bbox
        self.apply_style()

    def get_colors(self):
        return self.custom_colors
