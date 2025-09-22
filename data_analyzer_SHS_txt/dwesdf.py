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
        self.setGeometry(100, 100, 1400, 800)

        # Initialize variables
        self.df1 = None
        self.df2 = None
        self.filename1 = None
        self.filename2 = None
        self.last_modified_time1 = None
        self.last_modified_time2 = None
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
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

        # File 1 controls
        self.btn_file1 = QPushButton("Select File 1")
        self.btn_file1.clicked.connect(lambda: self.load_file(1))
        file_layout.addWidget(self.btn_file1, 0, 0)

        self.file1_label = QLabel("No file selected")
        self.file1_label.setStyleSheet("color: gray; font-style: italic;")
        self.file1_label.setWordWrap(True)
        file_layout.addWidget(self.file1_label, 0, 1)

        # File 2 controls
        self.btn_file2 = QPushButton("Select File 2")
        self.btn_file2.clicked.connect(lambda: self.load_file(2))
        file_layout.addWidget(self.btn_file2, 1, 0)

        self.file2_label = QLabel("No file selected")
        self.file2_label.setStyleSheet("color: gray; font-style: italic;")
        self.file2_label.setWordWrap(True)
        file_layout.addWidget(self.file2_label, 1, 1)

        # Add auto-refresh checkbox
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        file_layout.addWidget(self.auto_refresh_cb, 2, 0, 1, 2)

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

        # Y1-axis selection list (File 1)
        y1_label = QLabel("Y1 (File 1):")
        y1_label.setStyleSheet("font-weight: bold;")
        axis_layout.addWidget(y1_label, 1, 0)
        self.y1_list = QListWidget()
        self.y1_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.y1_list.setMinimumHeight(100)
        axis_layout.addWidget(self.y1_list, 1, 1)

        # Y2-axis selection list (File 2)
        y2_label = QLabel("Y2 (File 2):")
        y2_label.setStyleSheet("font-weight: bold;")
        axis_layout.addWidget(y2_label, 2, 0)
        self.y2_list = QListWidget()
        self.y2_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.y2_list.setMinimumHeight(100)
        axis_layout.addWidget(self.y2_list, 2, 1)

        # Plot customization
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        axis_layout.addWidget(self.grid_cb, 3, 0)

        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        axis_layout.addWidget(self.legend_cb, 3, 1)

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
        if not self.auto_refresh_cb.isChecked():
            return

        try:
            if self.filename1:
                current_mtime = os.path.getmtime(self.filename1)
                if self.last_modified_time1 is None or current_mtime > self.last_modified_time1:
                    self.last_modified_time1 = current_mtime
                    self.reload_file(1)
                    self.create_plot()

            if self.filename2:
                current_mtime = os.path.getmtime(self.filename2)
                if self.last_modified_time2 is None or current_mtime > self.last_modified_time2:
                    self.last_modified_time2 = current_mtime
                    self.reload_file(2)
                    self.create_plot()

        except Exception as e:
            print(f"Error checking file updates: {str(e)}")

    def reload_file(self, file_num):
        filename = self.filename1 if file_num == 1 else self.filename2
        if filename:
            df = self.read_data_file(filename)
            if df is not None:
                if file_num == 1:
                    self.df1 = df
                else:
                    self.df2 = df
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

    def load_file(self, file_num):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, f"Select File {file_num}",
                                                      "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)")
            if not filename:
                return

            # Read the file
            df = self.read_data_file(filename)
            if df is None:
                return

            # Store the DataFrame and filename
            if file_num == 1:
                self.df1 = df
                self.filename1 = filename
                self.last_modified_time1 = os.path.getmtime(filename)
                # Update file 1 label
                self.file1_label.setText(os.path.basename(filename))
                self.file1_label.setStyleSheet("color: black; font-style: normal;")
                # Update combo boxes for file 1
                self.x_combo.clear()
                self.y1_list.clear()
                self.x_combo.addItems(df.columns)
                self.y1_list.addItems(df.columns)
            else:
                self.df2 = df
                self.filename2 = filename
                self.last_modified_time2 = os.path.getmtime(filename)
                # Update file 2 label
                self.file2_label.setText(os.path.basename(filename))
                self.file2_label.setStyleSheet("color: black; font-style: normal;")
                # Update list for file 2
                self.y2_list.clear()
                self.y2_list.addItems(df.columns)

            # Update data information display
            self.update_info()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading file: {str(e)}")

    def update_info(self):
        info_text = ""

        if self.df1 is not None:
            info_text += f"File 1 ({os.path.basename(self.filename1)}):\n"
            info_text += f"Rows: {len(self.df1)}, Columns: {len(self.df1.columns)}\n"
            if self.last_modified_time1:
                info_text += f"Last Updated: {time.ctime(self.last_modified_time1)}\n\n"

        if self.df2 is not None:
            info_text += f"File 2 ({os.path.basename(self.filename2)}):\n"
            info_text += f"Rows: {len(self.df2)}, Columns: {len(self.df2.columns)}\n"
            if self.last_modified_time2:
                info_text += f"Last Updated: {time.ctime(self.last_modified_time2)}"

        self.info_text.setText(info_text)

    def create_plot(self):
        try:
            if self.df1 is None:
                QMessageBox.warning(self, "Error", "Please load at least one file")
                return

            # Get selected columns
            x_col = self.x_combo.currentText()
            y1_cols = [item.text() for item in self.y1_list.selectedItems()]

            # Clear previous plot
            self.ax.clear()

            # Plot data from file 1
            color_index = 0
            file1_name = os.path.basename(self.filename1)
            for y1_col in y1_cols:
                color = self.colors[color_index % len(self.colors)]
                self.ax.plot(self.df1[x_col], self.df1[y1_col],
                             color=color, linestyle='-',
                             label=f'{file1_name} - {y1_col}')
                color_index += 1

            # Plot data from file 2 if available
            if self.df2 is not None:
                y2_cols = [item.text() for item in self.y2_list.selectedItems()]
                file2_name = os.path.basename(self.filename2)
                for y2_col in y2_cols:
                    color = self.colors[color_index % len(self.colors)]
                    self.ax.plot(self.df2[x_col], self.df2[y2_col],
                                 color=color, linestyle='--',
                                 label=f'{file2_name} - {y2_col}')
                    color_index += 1

            # Customize plot
            self.ax.set_xlabel(x_col)
            self.ax.set_ylabel('Values')
            self.ax.set_title('Data Analysis')

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