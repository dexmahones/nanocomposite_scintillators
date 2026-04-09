import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QLineEdit, QLabel, QHBoxLayout, QFormLayout, QCheckBox, QScrollArea, QFrame, QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class CSVPairplotViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Pairplot Viewer")
        self.df = None
        self.csv_filepaths = []

        layout = QHBoxLayout()
        filelayout = QVBoxLayout()
        self.select_folder_button = QPushButton("Select Folder")

        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["CSV Files"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Renaming + Selection Section
        self.rename_form = QFormLayout()
        self.rename_widgets = []
        self.rename_frame = QFrame()
        self.grid_lyt = QVBoxLayout()
        self.header_lyt = QHBoxLayout()
        self.headers = [QLabel("Measurement"),QLabel("Alas"),QLabel("Cuts"),QLabel("Selection")]
        for label in self.headers:
            self.header_lyt.addWidget(label)

        self.grid_lyt.addLayout(self.header_lyt)
        self.grid_lyt.addLayout(self.rename_form)
        self.rename_frame.setLayout(self.grid_lyt)
        self.rename_frame.setVisible(False)

        self.pairplot = QPushButton("Pairplot")
        self.pairplot.clicked.connect(self.plot_pairplot)
        self.pairplot.setVisible(False)
        self.histplots = QPushButton("Histograms")
        self.histplots.setVisible(False)
        self.histplots.clicked.connect(self.plot_histograms)

        plot_btns = QHBoxLayout()
        plot_btns.addWidget(self.histplots)
        plot_btns.addWidget(self.pairplot)

        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        matplotlib_widget = QWidget()
        matplotlib_layout = QVBoxLayout()
        tools = QHBoxLayout()
        self.num_bins = QLineEdit("Num bins")
        self.logx = QCheckBox("Log X")
        self.logy = QCheckBox("Log Y")
        self.num_events = QLineEdit("Num Events")
        self.normalize = QCheckBox("Normalise")

        self.logx.clicked.connect(self.plot_histograms)
        self.logy.clicked.connect(self.plot_histograms)
        self.normalize.clicked.connect(self.plot_histograms)

        self.cut_toggle = QCheckBox("Cut")
        self.cut_min = QLineEdit("Min")
        self.cut_max = QLineEdit("Max")
        self.num_bins.editingFinished.connect(self.plot_histograms)
        self.num_events .editingFinished.connect(self.plot_histograms)
        self.cut_toggle.clicked.connect(self.plot_histograms)
        self.cut_min.editingFinished.connect(self.plot_histograms)
        self.cut_max.editingFinished.connect(self.plot_histograms)

        tools.addWidget(self.toolbar)
        tools.addWidget(self.logx)
        tools.addWidget(self.logy)
        tools.addWidget(self.normalize)
        tools.addWidget(self.num_bins)
        tools.addWidget(self.num_events)

        matplotlib_layout.addLayout(tools)
        matplotlib_layout.addWidget(self.canvas)
        matplotlib_widget.setLayout(matplotlib_layout)

        # Assemble left panel
        filelayout.addWidget(self.select_folder_button)
        filelayout.addWidget(self.table)
        filelayout.addWidget(self.rename_frame)
        filelayout.addLayout(plot_btns)
        layout.addLayout(filelayout, stretch=1)
        layout.addWidget(matplotlib_widget, stretch=3)
        self.setLayout(layout)

        # Connect events
        self.select_folder_button.clicked.connect(self.select_folder)
        self.table.cellClicked.connect(self.load_selected_csv)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.populate_csv_table(folder_path)

    def populate_csv_table(self, folder_path):
        self.csv_filepaths.clear()
        self.table.setRowCount(0)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    full_path = os.path.join(root, file)
                    size_kb = os.path.getsize(full_path) / 1024
                    if size_kb <= 100000:
                        self.csv_filepaths.append(full_path)
                        row = self.table.rowCount()
                        self.table.insertRow(row)
                        self.table.setItem(row, 0, QTableWidgetItem(file))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        if not self.csv_filepaths:
            QMessageBox.information(self, "No CSVs", "No CSV files <10MB found.")

    def load_selected_csv(self, row, _):
        filepath = self.csv_filepaths[row]
        try:
            self.df = pd.read_csv(filepath)
            if self.df.shape[1] < 2:
                raise ValueError("CSV must have at least two columns.")

            self.populate_rename_form()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load or plot CSV:\n{e}")

    def populate_rename_form(self):
        # Clear old widgets
        while self.rename_form.rowCount():
            self.rename_form.removeRow(0)
        self.rename_widgets = []
        
        for col in self.df.columns:
            checkbox = QCheckBox()
            checkbox.setChecked(True)

            lineedit = QLineEdit(col)
            minedit = QLineEdit("Min")
            maxedit = QLineEdit("1")
            minselect = QLineEdit("Min")
            maxselect = QLineEdit("Max")

            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(lineedit)
            row_layout.addWidget(minedit)
            row_layout.addWidget(maxedit)
            row_layout.addWidget(minselect)
            row_layout.addWidget(maxselect)
            row_widget.setLayout(row_layout)

            self.rename_form.addRow(QLabel(col), row_widget)
            self.rename_widgets.append((checkbox, lineedit, minedit, maxedit, minselect, maxselect))
        
        self.rename_frame.setVisible(True)
        self.pairplot.setVisible(True)
        self.histplots.setVisible(True)

    def apply_renames(self):
        if self.df is None:
            print("No data")
            return None

        # Determine selected columns and new names
        selected_cols = ["Selected"]
        new_names = []
        self.plot_df = self.df.copy()
        self.plot_df["Selected"] = True
        for (checkbox, lineedit, minedit, maxedit, minselect, maxselect), orig_name in zip(self.rename_widgets, self.df.columns):
            if checkbox.isChecked():
                try:
                    minval = float(minedit.text())
                except:
                    minval = self.plot_df[orig_name].min()

                try:
                    maxval = float(maxedit.text())
                except:
                    maxval = self.plot_df[orig_name].max()

                self.plot_df = self.plot_df[(self.plot_df[orig_name]>=minval) & (self.plot_df[orig_name]<=maxval)]

                try:
                    minsel = float(minselect.text())
                except:
                    minsel = self.plot_df[orig_name].min()

                try:
                    maxsel = float(maxselect.text())
                except:
                    maxsel = self.plot_df[orig_name].max()

                self.plot_df["Selected"] = (self.plot_df[orig_name]>=minsel) & (self.plot_df[orig_name]<=maxsel) & self.plot_df["Selected"]

                selected_cols.append(orig_name)
                new_names.append(lineedit.text())

        if len(set(new_names)) != len(new_names):
            QMessageBox.warning(self, "Warning", "Column names must be unique.")
            return self.plot_df

        try:
            self.plot_df = self.plot_df[selected_cols]
            self.plot_df.columns = ["Selected"]+new_names

            if self.cut_toggle.isChecked():
                try:
                    minval = float(minedit.text())
                except:
                    minval = self.plot_df[orig_name].min()

                try:
                    maxval = float(maxedit.text())
                except:
                    maxval = self.plot_df[orig_name].max()

            self.plot_df
            return self.plot_df
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename or plot:\n{e}")
            return self.plot_df
    
    def plot_histograms(self):
        
        self.apply_renames()
        if self.plot_df is None:
            return
        
        selected_columns = [col for col in self.plot_df.columns if col!="Selected"]

        plt.close("all")
        n = len(selected_columns)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        self.figure.clear()
        axes = self.figure.subplots(rows, cols).flatten()
        try:
            num_bins = int(self.num_bins.text())
        except:
            num_bins = 200

        try:
            num_events = int(self.num_events.text())
        except:
            num_events = len(self.plot_df)
        print(len(self.plot_df))
        for i, col in enumerate(selected_columns):
            try:
                axes[i].hist(self.plot_df[col].dropna().iloc[0:num_events], bins=num_bins, density = self.normalize.isChecked())
                axes[i].set_title(col)
                if self.logx.isChecked():
                    axes[i].set_xscale("log")
                if self.logy.isChecked():
                    axes[i].set_yscale("log")
            except Exception as e:
                print(i,str(e))

        for ax in axes[n:]:
            ax.set_visible(False)
        
        self.canvas.figure = self.figure
        self.canvas.draw()
    
    def plot_pairplot(self):
        if hasattr(self, "pairplot_fig") and self.pairplot is not None:
            plt.close(self.pairplot_fig.figure)

        self.plot_df = self.apply_renames()
        selected_columns = [col for col in self.plot_df.columns if col!="Selected"]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("Generating pairplot...")
        self.canvas.draw()
        
        try:
            num_events = int(self.num_events.text())
        except:
            num_events = len(self.plot_df)

        try:
            sns_fig = sns.pairplot(self.plot_df.iloc[0:num_events], vars = selected_columns, hue = "Selected", diag_kind = 'hist',plot_kws={"s": 10, "alpha": 0.25}, corner = True)
            # sns_fig.figure.tight_layout()
            self.pairplot_fig = sns_fig
            self.canvas.figure = sns_fig.figure
            # self.pairplot_fig.map_lower(sns.kdeplot, levels=4, color=".2")
            self.canvas.draw()
        except ValueError as e:
            reply = QMessageBox.warning(
                self, 
                "Seaborn Error", 
                f"Could not generate pairplot:\n{e}\n\nDataset has {len(self.plot_df)} entries. Downsample and try again?",
                QMessageBox.Yes | QMessageBox.No,      # buttons
                QMessageBox.No                        # default button
            )

            if reply == QMessageBox.Yes:
                self.plot_df = self.plot_df.sample(frac=0.5, random_state=42)
                self.plot_pairplot()
            else:
                ax.set_title("Plot generation cancelled.")
        except Exception as e:
            QMessageBox.critical(self, "Seaborn Error", f"Could not generate pairplot:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = CSVPairplotViewer()
    viewer.showMaximized()
    sys.exit(app.exec_())
