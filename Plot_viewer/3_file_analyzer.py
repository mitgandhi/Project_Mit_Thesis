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
                             QListWidget, QAbstractItemView, QScrollArea, QSpinBox)
from PyQt5.QtCore import Qt, QTimer


class DynamicDataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Multi-File Data Analyzer")
        self.setGeometry(100, 100, 1600, 800)

        # Initialize variables for dynamic file handling
        self.dfs = []  # List to store DataFrames
        self.filenames = []  # List to store filenames
        self.last_modified_times = []  # List to store modification times
        self.file_buttons = []  # List to store file buttons
        self.file_labels = []  # List to store file labels
        self.y_lists = []  # List to store Y-axis selection widgets
        self.remove_buttons = []  # List to store remove buttons

        # Color and style settings
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.linestyles = ['-', '--', '-.', ':']
        self.markers = ['o', 'x', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D']

        # Start with 3 files by default
        self.num_files = 3
        self.initialize_file_storage()

        self.setup_ui()

        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_file_updates)
        self.refresh_timer.start(1000)  # Check every second

    def initialize_file_storage(self):
        """Initialize storage for the specified number of files"""
        # Extend lists if needed
        while len(self.dfs) < self.num_files:
            self.dfs.append(None)
            self.filenames.append(None)
            self.last_modified_times.append(None)

        # Truncate lists if needed
        if len(self.dfs) > self.num_files:
            self.dfs = self.dfs[:self.num_files]
            self.filenames = self.filenames[:self.num_files]
            self.last_modified_times = self.last_modified_times[:self.num_files]

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create horizontal layout for left panel and right panel
        h_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # File count control
        count_group = QGroupBox("File Count Control")
        count_layout = QHBoxLayout()

        count_layout.addWidget(QLabel("Number of Files:"))
        self.file_count_spin = QSpinBox()
        self.file_count_spin.setMinimum(1)
        self.file_count_spin.setMaximum(50)
        self.file_count_spin.setValue(self.num_files)
        self.file_count_spin.valueChanged.connect(self.update_file_count)
        count_layout.addWidget(self.file_count_spin)

        self.add_file_btn = QPushButton("Add File")
        self.add_file_btn.clicked.connect(self.add_file)
        count_layout.addWidget(self.add_file_btn)

        count_group.setLayout(count_layout)
        left_layout.addWidget(count_group)

        # Create scrollable area for file controls
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)

        # File selection group
        self.file_group = QGroupBox("File Selection")
        self.file_layout = QGridLayout()
        self.file_group.setLayout(self.file_layout)
        self.scroll_layout.addWidget(self.file_group)

        # Auto-refresh checkbox
        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        self.scroll_layout.addWidget(self.auto_refresh_cb)

        # Data Information
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        self.scroll_layout.addWidget(info_group)

        # X-axis selection group
        x_axis_group = QGroupBox("X-Axis Selection (from File 1)")
        x_axis_layout = QGridLayout()
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(200)
        x_axis_layout.addWidget(self.x_combo, 0, 0)
        x_axis_group.setLayout(x_axis_layout)
        self.scroll_layout.addWidget(x_axis_group)

        # Y-axis selections group
        self.y_axis_group = QGroupBox("Y-Axis Selection")
        self.y_axis_layout = QGridLayout()
        self.y_axis_group.setLayout(self.y_axis_layout)
        self.scroll_layout.addWidget(self.y_axis_group)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)

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

        # Initialize file controls
        self.create_file_controls()

    def update_file_count(self, count):
        """Update the number of files based on spinbox value"""
        self.num_files = count
        self.initialize_file_storage()
        self.create_file_controls()

    def add_file(self):
        """Add one more file to the analyzer"""
        self.num_files += 1
        self.file_count_spin.setValue(self.num_files)
        self.initialize_file_storage()
        self.create_file_controls()

    def remove_file(self, file_num):
        """Remove a specific file from the analyzer"""
        if self.num_files <= 1:
            QMessageBox.warning(self, "Warning", "Must have at least one file!")
            return

        # Remove data for this file
        if file_num < len(self.dfs):
            self.dfs.pop(file_num)
            self.filenames.pop(file_num)
            self.last_modified_times.pop(file_num)

        self.num_files -= 1
        self.file_count_spin.setValue(self.num_files)
        self.create_file_controls()
        self.update_info()

    def create_file_controls(self):
        """Create UI controls for all files"""
        # Clear existing controls
        self.clear_file_controls()

        # Initialize storage
        self.initialize_file_storage()

        # Create new controls
        self.file_buttons = []
        self.file_labels = []
        self.remove_buttons = []
        self.y_lists = []

        # File selection controls
        for i in range(self.num_files):
            # File selection button
            btn = QPushButton(f"Select File {i + 1}")
            btn.clicked.connect(lambda checked, x=i: self.load_file(x))
            self.file_layout.addWidget(btn, i, 0)
            self.file_buttons.append(btn)

            # File label
            label = QLabel("No file selected")
            label.setStyleSheet("color: gray; font-style: italic;")
            label.setWordWrap(True)
            self.file_layout.addWidget(label, i, 1)
            self.file_labels.append(label)

            # Remove button (only if more than 1 file)
            if self.num_files > 1:
                remove_btn = QPushButton("Remove")
                remove_btn.setMaximumWidth(60)
                remove_btn.clicked.connect(lambda checked, x=i: self.remove_file(x))
                self.file_layout.addWidget(remove_btn, i, 2)
                self.remove_buttons.append(remove_btn)

        # Y-axis selection controls
        for i in range(self.num_files):
            # Label
            self.y_axis_layout.addWidget(QLabel(f"File {i + 1} Y-Axes:"), i * 2, 0)

            # List widget
            y_list = QListWidget()
            y_list.setSelectionMode(QAbstractItemView.MultiSelection)
            y_list.setMinimumHeight(80)
            y_list.setMaximumHeight(120)
            self.y_axis_layout.addWidget(y_list, i * 2 + 1, 0)
            self.y_lists.append(y_list)

        # Update existing file displays
        for i in range(min(len(self.filenames), self.num_files)):
            if self.filenames[i]:
                self.file_labels[i].setText(os.path.basename(self.filenames[i]))
                self.file_labels[i].setStyleSheet("color: black; font-style: normal;")
                if self.dfs[i] is not None:
                    self.update_column_lists(i)

        self.update_info()

    def clear_file_controls(self):
        """Clear all existing file control widgets"""
        # Clear file layout
        while self.file_layout.count():
            child = self.file_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear y-axis layout
        while self.y_axis_layout.count():
            child = self.y_axis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_third_last_folder(self, filepath):
        """Extract third-to-last folder name from filepath"""
        if filepath:
            path_parts = os.path.normpath(filepath).split(os.sep)
            if len(path_parts) >= 4:
                return path_parts[-4]
        return "Unknown"

    def check_file_updates(self):
        """Check if any files have been updated and refresh if needed"""
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
            for i in range(len(self.filenames)):
                if self.filenames[i] and os.path.exists(self.filenames[i]):
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
                    if i < len(current_y):
                        for j in range(y_list.count()):
                            item = y_list.item(j)
                            if item.text() in current_y[i]:
                                item.setSelected(True)

                # Update plot with restored selections
                self.create_comparison_plot()

        except Exception as e:
            print(f"Error checking file updates: {str(e)}")

    def reload_file(self, file_num):
        """Reload a specific file"""
        if file_num < len(self.filenames) and self.filenames[file_num]:
            df = self.read_data_file(self.filenames[file_num])
            if df is not None:
                self.dfs[file_num] = df
                if file_num == 0:  # Update X-axis options only for File 1
                    self.x_combo.clear()
                    self.x_combo.addItems(df.columns)
                self.update_info()
                self.update_column_lists(file_num)

    def read_data_file(self, filename):
        """Read data file with multiple delimiter attempts"""
        try:
            # First try to read as standard CSV
            df = pd.read_csv(filename)

            # If that fails, try reading with different delimiters
            if len(df.columns) == 1:
                for delimiter in ['\t', ' ', ';', '|']:
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
        """Load a file for the specified file number"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, f"Select File {file_num + 1}",
                "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
            )
            if not filename:
                return

            # Read the file
            df = self.read_data_file(filename)
            if df is None:
                return

            # Ensure we have enough storage space
            while len(self.dfs) <= file_num:
                self.dfs.append(None)
                self.filenames.append(None)
                self.last_modified_times.append(None)

            # Store the DataFrame and filename
            self.dfs[file_num] = df
            self.filenames[file_num] = filename
            self.last_modified_times[file_num] = os.path.getmtime(filename)

            if file_num < len(self.file_labels):
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
        """Update the column list for a specific file"""
        if file_num < len(self.dfs) and file_num < len(self.y_lists) and self.dfs[file_num] is not None:
            self.y_lists[file_num].clear()
            self.y_lists[file_num].addItems(self.dfs[file_num].columns)

    def update_info(self):
        """Update the information display"""
        info_text = ""
        for i in range(min(len(self.dfs), self.num_files)):
            if self.dfs[i] is not None:
                filename = self.filenames[i] if i < len(self.filenames) else "Unknown"
                info_text += f"File {i + 1}: {os.path.basename(filename) if filename else 'Unknown'}\n"
                info_text += f"Rows: {len(self.dfs[i])}, Columns: {len(self.dfs[i].columns)}\n"
                if i < len(self.last_modified_times) and self.last_modified_times[i]:
                    info_text += f"Last Updated: {time.ctime(self.last_modified_times[i])}\n"
                info_text += "\n"

        self.info_text.setText(info_text)

    def create_comparison_plot(self):
        """Create comparison plot with data from all loaded files"""
        try:
            # Check if File 1 is loaded (required for X-axis)
            if not self.dfs or self.dfs[0] is None:
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
            style_index = 0

            # Plot data from all files
            for file_num in range(min(len(self.dfs), len(self.y_lists))):
                if self.dfs[file_num] is not None and file_num < len(self.y_lists):
                    folder_name = self.get_third_last_folder(
                        self.filenames[file_num] if file_num < len(self.filenames) else None
                    )
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
                            linestyle = self.linestyles[style_index % len(self.linestyles)]
                            marker = self.markers[file_num % len(self.markers)]

                            if file_num == 0:
                                y_values = self.dfs[file_num][y_col]
                            else:
                                y_values = self.dfs[file_num][y_col].iloc[:min_length]

                            self.ax.plot(x_values, y_values,
                                         color=color,
                                         linestyle=linestyle,
                                         marker=marker,
                                         markersize=3,
                                         label=f"F{file_num + 1} {folder_name} - {y_col}",
                                         linewidth=1.5)

                            all_y_cols.append(y_col)
                            color_index += 1

                    style_index += 1

            # Customize plot
            self.ax.set_xlabel(x_col)
            if all_y_cols:
                unique_y_cols = list(set(all_y_cols))
                self.ax.set_ylabel(' / '.join(unique_y_cols[:5]))  # Limit ylabel length

            self.ax.set_title(f'Multi-File Data Comparison (X-axis: {x_col})')

            if self.grid_cb.isChecked():
                self.ax.grid(True, alpha=0.3)

            if self.legend_cb.isChecked():
                # Legend below plot in multiple columns
                legend = self.ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,
                                        frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
                legend.get_frame().set_facecolor('white')

                # Adjust bottom margin to accommodate legend without shrinking plot
                self.figure.subplots_adjust(bottom=0.2)
            else:
                # Reset to default margins when legend is hidden
                self.figure.subplots_adjust(bottom=0.1)

            # Update canvas
            self.canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error",
                                f"Error creating plot: {str(e)}\n"
                                f"Try selecting different columns or check if the data is valid.")


def main():
    app = QApplication(sys.argv)
    window = DynamicDataAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()