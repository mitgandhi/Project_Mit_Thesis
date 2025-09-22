import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QVBoxLayout, QWidget, QLabel, QMessageBox)


class DataExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Column Extractor')
        self.setGeometry(100, 100, 400, 200)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create and add widgets
        self.status_label = QLabel('No file selected')
        browse_button = QPushButton('Browse Text File')
        extract_button = QPushButton('Extract and Save Column')

        layout.addWidget(self.status_label)
        layout.addWidget(browse_button)
        layout.addWidget(extract_button)

        # Connect buttons to functions
        browse_button.clicked.connect(self.browse_file)
        extract_button.clicked.connect(self.extract_and_save)

        self.input_file = None

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_name:
            self.input_file = file_name
            self.status_label.setText(f'Selected file: {file_name}')

    def extract_and_save(self):
        if not self.input_file:
            QMessageBox.warning(self, 'Warning', 'Please select an input file first!')
            return

        try:
            # Read the text file
            df = pd.read_csv(self.input_file, delimiter='\s+', header=None)

            # Extract column 1 (index 1)
            column_1 = df[1]

            # Get save file name
            save_file, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV File",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )

            if save_file:
                # Add .csv extension if not present
                if not save_file.endswith('.csv'):
                    save_file += '.csv'

                # Save to CSV
                column_1.to_csv(save_file, index=False, header=['Column1'])
                QMessageBox.information(self, 'Success', 'Column successfully extracted and saved!')

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')


def main():
    app = QApplication(sys.argv)
    window = DataExtractorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()