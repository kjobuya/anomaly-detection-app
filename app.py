from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor
from PyQt5.uic import loadUi

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        loadUi(r'assets\start_window.ui', self)  # Load the .ui file
        
        self.reset()
        
        self.num_widgets = self.stackedWidget.count()
        self.current_page = 0
        self.stackedWidget.setCurrentIndex(self.current_page)
        
        # Button events
        self.nextButton.clicked.connect(self.next)
        self.backButton.clicked.connect(self.back)
        self.addNominalPathButton.clicked.connect(self.addNominalPath)
        self.addDefectPathButton.clicked.connect(self.addDefectPath)
        
        # ComboBox events
        self.nominalComboBox.currentIndexChanged.connect(self.on_nominalComboBox_changed)
        self.defectComboBox.currentIndexChanged.connect(self.on_defectComboBox_changed)
        
        self.show()
        
    def reset(self):
        self.nominal_path = None
        self.defect_path = None
        
        self.nominalComboBox.clear()
        self.defectComboBox.clear()
        
        
    # event handlers functions    
    def next(self):
        if self.current_page != self.num_widgets - 1:
            self.current_page += 1
            self.stackedWidget.setCurrentIndex(self.current_page)
        
    def back(self):
        if self.current_page != 0:
            self.current_page -= 1
            self.stackedWidget.setCurrentIndex(self.current_page)
            
    def addNominalPath(self):
        self.nominal_path = QFileDialog.getExistingDirectory(self, 'Select Nominal Directory')
        if self.nominal_path:
            self.nominalComboBox.addItem(self.nominal_path)
            self.nominalComboBox.setCurrentIndex(self.nominalComboBox.count() - 1)
        
    def addDefectPath(self):
        self.defect_path = QFileDialog.getExistingDirectory(self, 'Select Defect Directory')
        if self.defect_path:
            self.defectComboBox.addItem(self.nominal_path)
            self.defectComboBox.setCurrentIndex(self.defectComboBox.count() - 1)
            
    def on_nominalComboBox_changed(self):
        self.nominal_path = self.nominalComboBox.currentText()
        
    def on_defectComboBox_changed(self):
        self.defect_path = self.defectComboBox.currentText()