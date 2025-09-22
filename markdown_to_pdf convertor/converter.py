import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
                             QCheckBox, QFileDialog, QMessageBox, QProgressBar,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt
import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, LEGAL
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from pathlib import Path
import datetime
import re


class MarkdownConverterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Markdown to PDF Converter")
        self.setMinimumSize(600, 400)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        file_layout.addWidget(QLabel("Input Markdown File:"), 0, 0)
        self.input_path = QLineEdit()
        file_layout.addWidget(self.input_path, 0, 1)
        browse_input = QPushButton("Browse")
        browse_input.clicked.connect(self.browse_input_file)
        file_layout.addWidget(browse_input, 0, 2)

        file_layout.addWidget(QLabel("Output PDF File:"), 1, 0)
        self.output_path = QLineEdit()
        file_layout.addWidget(self.output_path, 1, 1)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.browse_output_file)
        file_layout.addWidget(browse_output, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Document Settings Group
        settings_group = QGroupBox("Document Settings")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Page Size:"), 0, 0)
        self.page_size = QComboBox()
        self.page_size.addItems(['A4', 'Letter', 'Legal'])
        settings_layout.addWidget(self.page_size, 0, 1)

        settings_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.font_size = QComboBox()
        self.font_size.addItems(['10', '11', '12', '14'])
        settings_layout.addWidget(self.font_size, 1, 1)

        self.include_header = QCheckBox("Include Header/Footer")
        self.include_header.setChecked(True)
        settings_layout.addWidget(self.include_header, 2, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Convert Button
        convert_button = QPushButton("Convert to PDF")
        convert_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        convert_button.clicked.connect(self.convert_file)
        layout.addWidget(convert_button)

        # Status Label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def browse_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Markdown File",
            "",
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)"
        )
        if file_name:
            self.input_path.setText(file_name)
            if not self.output_path.text():
                default_output = str(Path(file_name).with_suffix('.pdf'))
                self.output_path.setText(default_output)

    def browse_output_file(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF As",
            "",
            "PDF Files (*.pdf);;All Files (*.*)"
        )
        if file_name:
            self.output_path.setText(file_name)

    def convert_file(self):
        if not self.input_path.text() or not self.output_path.text():
            QMessageBox.warning(self, "Warning", "Please select both input and output files.")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(20)
            self.status_label.setText("Reading input file...")

            # Read markdown content
            with open(self.input_path.text(), 'r', encoding='utf-8') as f:
                md_content = f.read()

            self.progress.setValue(40)
            self.status_label.setText("Converting markdown...")

            # Convert markdown to HTML
            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

            self.progress.setValue(60)
            self.status_label.setText("Generating PDF...")

            # Create PDF
            self.create_pdf(html_content)

            self.progress.setValue(100)
            self.status_label.setText("Conversion completed successfully!")
            QMessageBox.information(self, "Success", "PDF generated successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            self.status_label.setText("Conversion failed.")
        finally:
            self.progress.setVisible(False)

    def create_pdf(self, html_content):
        # Set up page size
        page_sizes = {'A4': A4, 'Letter': LETTER, 'Legal': LEGAL}
        page_size = page_sizes.get(self.page_size.currentText(), A4)

        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_path.text(),
            pagesize=page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        heading_style = styles['Heading1']

        # Update font sizes
        base_size = int(self.font_size.currentText())
        normal_style.fontSize = base_size
        heading_style.fontSize = base_size + 4

        # Create story (content)
        story = []

        # Add header if enabled
        if self.include_header.isChecked():
            header = Paragraph(
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                normal_style
            )
            story.append(header)
            story.append(Spacer(1, 12))

        # Process HTML content
        sections = re.split(r'(<h[1-6].*?</h[1-6]>)', html_content)

        for section in sections:
            if section.startswith('<h1'):
                # Handle headers
                text = re.sub('<[^<]+?>', '', section)
                story.append(Paragraph(text, heading_style))
                story.append(Spacer(1, 12))
            else:
                # Handle regular paragraphs
                text = re.sub('<[^<]+?>', '', section)
                if text.strip():
                    story.append(Paragraph(text, normal_style))
                    story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)


def main():
    app = QApplication(sys.argv)
    window = MarkdownConverterWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()