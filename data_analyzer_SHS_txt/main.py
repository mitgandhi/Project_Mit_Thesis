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


class TripleDataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triple File Data Analyzer")
        self.setGeometry(100, 100, 1600, 800)

        # Initialize variables for all three files
        self.dfs = [None, None, None]  # List to store DataFrames
        self.filenames = [None, None, None]  # List to store filenames
        self.last_modified_times = [None, None, None]  # List to store modification times
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.linestyles = ['-', '--', '-.']  # Different line styles for each file
        self.markers = ['o', 'x', 's']  # Different markers for each file

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

        # File selection for all files
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        # Create file controls for all three files
        self.file_buttons = []
        self.file_labels = []
        for i in range(3):
            btn = QPushButton(f"Select File {i + 1}")
            btn.clicked.connect(lambda checked, x=i: self.load_file(x))
            file_layout.addWidget(btn, i, 0)
            self.file_buttons.append(btn)

            label = QLabel("No file selected")
            label.setStyleSheet("color: gray; font-style: italic;")
            label.setWordWrap(True)
            file_layout.addWidget(label, i, 1)
            self.file_labels.append(label)

        # Add auto-refresh checkbox
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        file_layout.addWidget(self.auto_refresh_cb, 3, 0)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Data Information
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # Common X-axis selection
        x_axis_group = QGroupBox("X-Axis Selection (from File 1)")
        x_axis_layout = QGridLayout()
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(200)
        x_axis_layout.addWidget(self.x_combo, 0, 0)
        x_axis_group.setLayout(x_axis_layout)
        left_layout.addWidget(x_axis_group)

        # Y-axis selections for all files
        y_axis_group = QGroupBox("Y-Axis Selection")
        y_axis_layout = QGridLayout()

        self.y_lists = []
        for i in range(3):
            y_axis_layout.addWidget(QLabel(f"File {i + 1} Y-Axes:"), i * 2, 0)
            y_list = QListWidget()
            y_list.setSelectionMode(QAbstractItemView.MultiSelection)
            y_list.setMinimumHeight(100)
            y_axis_layout.addWidget(y_list, i * 2 + 1, 0)
            self.y_lists.append(y_list)

        y_axis_group.setLayout(y_axis_layout)
        left_layout.addWidget(y_axis_group)

        # Plot customization
        custom_group = QGroupBox("Plot Customization")
        custom_layout = QGridLayout()

        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        custom_layout.addWidget(self.grid_cb, 0, 0)

        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        custom_layout.addWidget(self.legend_cb, 0, 1)

        custom_group.setLayout(custom_layout)
        left_layout.addWidget(custom_group)

        # Plot button
        self.plot_btn = QPushButton("Create Comparison Plot")
        self.plot_btn.clicked.connect(self.create_comparison_plot)
        self.plot_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(self.plot_btn)

        # Add stretch to push everything up
        left_layout.addStretch()

        h_layout.addWidget(left_panel, stretch=1)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        h_layout.addWidget(right_panel, stretch=2)

        layout.addLayout(h_layout)

    def get_third_last_folder(self, filepath):
        if filepath:
            # Split the path into components
            path_parts = os.path.normpath(filepath).split(os.sep)
            # If path has at least 2 components, return second-to-last
            if len(path_parts) >= 4:
                return path_parts[-4]
        return "Unknown"

    def check_file_updates(self):
        if not self.auto_refresh_cb.isChecked():
            return

        try:
            updated = False
            # Store current selections before refresh
            current_x = self.x_combo.currentText()
            current_y = []
            for y_list in self.y_lists:
                current_y.append([item.text() for item in y_list.selectedItems()])

            # Check all files
            for i in range(3):
                if self.filenames[i]:
                    current_mtime = os.path.getmtime(self.filenames[i])
                    if self.last_modified_times[i] is None or current_mtime > self.last_modified_times[i]:
                        self.last_modified_times[i] = current_mtime
                        self.reload_file(i)
                        updated = True

            if updated:
                # Restore X-axis selection
                index = self.x_combo.findText(current_x)
                if index >= 0:
                    self.x_combo.setCurrentIndex(index)

                # Restore Y-axis selections
                for i, y_list in enumerate(self.y_lists):
                    for j in range(y_list.count()):
                        item = y_list.item(j)
                        if item.text() in current_y[i]:
                            item.setSelected(True)

                # Update plot with restored selections
                self.create_comparison_plot()

        except Exception as e:
            print(f"Error checking file updates: {str(e)}")

    def reload_file(self, file_num):
        if self.filenames[file_num]:
            df = self.read_data_file(self.filenames[file_num])
            if df is not None:
                self.dfs[file_num] = df
                if file_num == 0:  # Update X-axis options only for File 1
                    self.x_combo.clear()
                    self.x_combo.addItems(df.columns)
                self.update_info()
                self.update_column_lists(file_num)

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
            filename, _ = QFileDialog.getOpenFileName(self, f"Select File {file_num + 1}",
                                                      "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)")
            if not filename:
                return

            # Read the file
            df = self.read_data_file(filename)
            if df is None:
                return

            # Store the DataFrame and filename
            self.dfs[file_num] = df
            self.filenames[file_num] = filename
            self.last_modified_times[file_num] = os.path.getmtime(filename)
            self.file_labels[file_num].setText(os.path.basename(filename))
            self.file_labels[file_num].setStyleSheet("color: black; font-style: normal;")

            # Update X-axis options if this is File 1
            if file_num == 0:
                self.x_combo.clear()
                self.x_combo.addItems(df.columns)

            # Update column lists and info
            self.update_column_lists(file_num)
            self.update_info()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading file: {str(e)}")

    def update_column_lists(self, file_num):
        if self.dfs[file_num] is not None:
            self.y_lists[file_num].clear()
            self.y_lists[file_num].addItems(self.dfs[file_num].columns)

    def update_info(self):
        info_text = ""
        for i in range(3):
            if self.dfs[i] is not None:
                info_text += f"File {i + 1}: {os.path.basename(self.filenames[i])}\n"
                info_text += f"Rows: {len(self.dfs[i])}, Columns: {len(self.dfs[i].columns)}\n"
                info_text += f"Last Updated: {time.ctime(self.last_modified_times[i])}\n\n"

        self.info_text.setText(info_text)

    def create_comparison_plot(self):
        try:
            if self.dfs[0] is None:  # Check if File 1 is loaded
                QMessageBox.warning(self, "Error", "Please load File 1 first (required for X-axis)")
                return

            # Get X-axis column from File 1
            x_col = self.x_combo.currentText()
            if not x_col:
                QMessageBox.warning(self, "Error", "Please select an X-axis column")
                return

            # Clear previous plot
            self.ax.clear()

            # Track all Y columns for color assignment
            all_y_cols = []
            color_index = 0

            # Plot data from all files
            for file_num in range(3):
                if self.dfs[file_num] is not None:
                    folder_name = self.get_third_last_folder(self.filenames[file_num])
                    y_cols = [item.text() for item in self.y_lists[file_num].selectedItems()]

                    if file_num == 0:
                        x_values = self.dfs[0][x_col]
                    else:
                        # Match lengths with File 1
                        min_length = min(len(self.dfs[0]), len(self.dfs[file_num]))
                        x_values = self.dfs[0][x_col].iloc[:min_length]

                    for y_col in y_cols:
                        if y_col in self.dfs[file_num].columns:
                            color = self.colors[color_index % len(self.colors)]
                            if file_num == 0:
                                y_values = self.dfs[file_num][y_col]
                            else:
                                y_values = self.dfs[file_num][y_col].iloc[:min_length]

                            self.ax.plot(x_values, y_values, color=color,
                                         label=f"{folder_name} - {y_col}",
                                         linestyle='-'
                                         )

                            all_y_cols.append(y_col)
                            color_index += 1

            # Customize plot
            self.ax.set_xlabel(x_col)
            if all_y_cols:
                self.ax.set_ylabel(' / '.join(all_y_cols))
            self.ax.set_title(f'Data Comparison between Files (X-axis: {x_col})')

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
    window = TripleDataAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()