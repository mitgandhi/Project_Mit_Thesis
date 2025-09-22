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


class SixFileDataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Six File Data Analyzer")
        self.setGeometry(100, 100, 1600, 800)

        # Initialize variables for all six files
        self.dfs = [None] * 6  # List to store DataFrames
        self.filenames = [None] * 6  # List to store filenames
        self.last_modified_times = [None] * 6  # List to store modification times
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
        self.linestyles = ['-', '--', '-.', ':', '-']  # Different line styles for each file
        self.markers = ['o', 'x', 's', 'd', '^', '*']  # Different markers for each file

        self.setup_ui()

        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_file_updates)
        self.refresh_timer.start(1000)  # Check every second

    def check_file_updates(self):
        if not self.auto_refresh_cb.isChecked():
            return

        try:
            updated = False
            for i in range(6):
                if self.filenames[i]:
                    current_mtime = os.path.getmtime(self.filenames[i])
                    if self.last_modified_times[i] is None or current_mtime > self.last_modified_times[i]:
                        self.last_modified_times[i] = current_mtime
                        self.reload_file(i)
                        updated = True

            if updated:
                self.create_comparison_plot()
        except Exception as e:
            print(f"Error checking file updates: {str(e)}")

    def reload_file(self, file_num):
        if self.filenames[file_num]:
            df = pd.read_csv(self.filenames[file_num])
            if df is not None:
                self.dfs[file_num] = df
                if file_num == 0:
                    self.x_combo.clear()
                    self.x_combo.addItems(df.columns)
                self.update_column_lists(file_num)
                self.update_info()

    def update_column_lists(self, file_num):
        if self.dfs[file_num] is not None:
            self.y_lists[file_num].clear()
            self.y_lists[file_num].addItems(self.dfs[file_num].columns)

    def update_info(self):
        info_text = ""
        for i in range(6):
            if self.dfs[i] is not None:
                info_text += f"File {i + 1}: {os.path.basename(self.filenames[i])}\n"
                info_text += f"Rows: {len(self.dfs[i])}, Columns: {len(self.dfs[i].columns)}\n"
                info_text += f"Last Updated: {time.ctime(self.last_modified_times[i])}\n\n"
        self.info_text.setText(info_text)

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

        self.file_buttons = []
        self.file_labels = []
        for i in range(6):
            btn = QPushButton(f"Select File {i + 1}")
            btn.clicked.connect(lambda checked, x=i: self.load_file(x))
            file_layout.addWidget(btn, i, 0)
            self.file_buttons.append(btn)

            label = QLabel("No file selected")
            label.setStyleSheet("color: gray; font-style: italic;")
            label.setWordWrap(True)
            file_layout.addWidget(label, i, 1)
            self.file_labels.append(label)

        self.auto_refresh_cb = QCheckBox("Auto Refresh")
        self.auto_refresh_cb.setChecked(True)
        file_layout.addWidget(self.auto_refresh_cb, 6, 0)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        h_layout.addWidget(left_panel, stretch=1)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        h_layout.addWidget(right_panel, stretch=2)
        layout.addLayout(h_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SixFileDataAnalyzer()
    window.show()
    sys.exit(app.exec_())
