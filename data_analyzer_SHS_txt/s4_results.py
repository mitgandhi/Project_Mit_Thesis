import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QFileDialog, QLabel,
                             QGroupBox, QGridLayout, QCheckBox, QMessageBox, QTextEdit,
                             QListWidget, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer


class DataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.df = None
        self.filename = None
        self.last_modified_time = None
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Default color cycle
        self.setup_ui()

        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_file_updates)
        self.refresh_timer.start(1000)  # Check every second

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create horizontal layout for left panel and right panel
        h_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # File selection with filename display
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        self.btn_file = QPushButton("Select File")
        self.btn_file.clicked.connect(self.load_file)
        file_layout.addWidget(self.btn_file, 0, 0)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label, 0, 1)

        # Add auto-refresh checkbox
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        file_layout.addWidget(self.auto_refresh_cb, 1, 0)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Data Information
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # Axis selection
        axis_group = QGroupBox("Plot Controls")
        axis_layout = QGridLayout()

        # X-axis selection
        axis_layout.addWidget(QLabel("X-Axis:"), 0, 0)
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(200)
        axis_layout.addWidget(self.x_combo, 0, 1)

        # Y-axis selection list
        axis_layout.addWidget(QLabel("Y-Axes:"), 1, 0)
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.y_list.setMinimumHeight(100)
        axis_layout.addWidget(self.y_list, 1, 1)

        # Plot customization
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        axis_layout.addWidget(self.grid_cb, 2, 0)

        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        axis_layout.addWidget(self.legend_cb, 2, 1)

        axis_group.setLayout(axis_layout)
        left_layout.addWidget(axis_group)

        # Plot button
        self.plot_btn = QPushButton("Create Plot")
        self.plot_btn.clicked.connect(self.create_plot)
        self.plot_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(self.plot_btn)

        # Add stretch to push everything up
        left_layout.addStretch()

        h_layout.addWidget(left_panel, stretch=1)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        h_layout.addWidget(right_panel, stretch=2)

        layout.addLayout(h_layout)

    def check_file_updates(self):
        if not self.auto_refresh_cb.isChecked() or self.filename is None:
            return

        try:
            current_mtime = os.path.getmtime(self.filename)
            if self.last_modified_time is None or current_mtime > self.last_modified_time:
                self.last_modified_time = current_mtime
                self.reload_file()
                self.create_plot()
        except Exception as e:
            print(f"Error checking file updates: {str(e)}")

    def reload_file(self):
        if self.filename:
            df = self.read_data_file(self.filename)
            if df is not None:
                self.df = df
                self.update_info()

    def read_data_file(self, filename):
        try:
            # First try to read as standard CSV
            df = pd.read_csv(filename)

            # If that fails, try reading with different delimiters
            if len(df.columns) == 1:
                for delimiter in ['\t', ' ', ';']:
                    try:
                        df = pd.read_csv(filename, delimiter=delimiter)
                        if len(df.columns) > 1:
                            break
                    except:
                        continue

            # Clean the data
            df = df.dropna(axis=1, how='all')  # Remove empty columns
            df = df.dropna(how='all')  # Remove empty rows

            return df

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error reading file: {str(e)}")
            return None

    def load_file(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, "Select File",
                                                      "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)")
            if not filename:
                return

            # Read the file
            df = self.read_data_file(filename)
            if df is None:
                return

            # Store the DataFrame and filename
            self.df = df
            self.filename = filename
            self.last_modified_time = os.path.getmtime(filename)

            # Update file label
            self.file_label.setText(os.path.basename(filename))
            self.file_label.setStyleSheet("color: black; font-style: normal;")

            # Update combo boxes and list
            self.x_combo.clear()
            self.y_list.clear()
            self.x_combo.addItems(df.columns)
            self.y_list.addItems(df.columns)

            # Update data information display
            self.update_info()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading file: {str(e)}")

    def update_info(self):
        if self.df is not None:
            info_text = f"File: {os.path.basename(self.filename)}\n"
            info_text += f"Rows: {len(self.df)}, Columns: {len(self.df.columns)}\n"
            info_text += f"Last Updated: {time.ctime(self.last_modified_time)}"
            self.info_text.setText(info_text)

    def create_plot(self):
        try:
            if self.df is None:
                QMessageBox.warning(self, "Error", "Please load a file first")
                return

            # Get selected columns
            x_col = self.x_combo.currentText()
            y_cols = [item.text() for item in self.y_list.selectedItems()]

            if not y_cols:
                QMessageBox.warning(self, "Error", "Please select at least one Y-axis column")
                return

            # Clear previous plot
            self.ax.clear()

            # Plot each selected Y column
            for i, y_col in enumerate(y_cols):
                color = self.colors[i % len(self.colors)]
                self.ax.plot(self.df[x_col], self.df[y_col], color=color, label=y_col)

            # Customize plot
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel(' / '.join(y_cols))
            self.ax.set_title(f'Data Analysis: Multiple Variables vs {x_col}')

            if self.grid_cb.isChecked():
                self.ax.grid(True)

            if self.legend_cb.isChecked():
                self.ax.legend()

            # Update canvas
            self.canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                                f"Error creating plot: {str(e)}\nTry selecting different columns or check if the data is valid.")


def main():
    app = QApplication(sys.argv)
    window = DataAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()