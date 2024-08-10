from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor
from PyQt5.uic import loadUi

from utilities import *

import os
import cv2

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.pages_dict = {
            "home": 0,
            "folder paths": 1,
            "nominal samples": 2,
        }
        
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
        self.nextNominalImageButton.clicked.connect(self.nextNominalImage)
        self.prevNominalImageButton.clicked.connect(self.prevNominalImage)
        
        # ComboBox events
        self.nominalComboBox.currentIndexChanged.connect(self.on_nominalComboBox_changed)
        self.defectComboBox.currentIndexChanged.connect(self.on_defectComboBox_changed)
        
        self.show()
        
    def reset(self):
        self.nominal_path = None
        self.defect_path = None
        self.displayed_nominal_img_idx = 0
        self.nominal_images = []
        self.defect_images = []
        
        self.nominalComboBox.clear()
        self.defectComboBox.clear()
        
    def view_change(self):
        self.stackedWidget.setCurrentIndex(self.current_page)
        
    def load_images(self, images_paths):
        images = []
        for image_path in images_paths:
            image = cv2.imread(image_path)
            images.append(image)
            
        return images
    
    def display_img(self, image, label:QLabel):                      
        disp_img = ConvertCVArray2QPixmap(image)
        label.setPixmap(disp_img)
        
    def display_nominal_img(self):
        self.display_img(self.nominal_images[self.displayed_nominal_img_idx], self.nominalImageDisplayLabel)
               
    # event handlers functions    
    def next(self):
        if self.current_page == self.pages_dict["folder paths"]:
            # need to load images from paths
            if self.nominal_path and self.defect_path:
                nominal_images_paths = [os.path.join(self.nominal_path, file) for file in os.listdir(self.nominal_path)]
                defect_images_paths = [os.path.join(self.defect_path, file) for file in os.listdir(self.defect_path)]
                
                self.nominal_images = self.load_images(nominal_images_paths)
                self.defect_images = self.load_images(defect_images_paths)
                
                self.display_nominal_img()
                
                      
        if self.current_page != self.num_widgets - 1:
            self.current_page += 1
            self.view_change()
        
    def back(self):
        if self.current_page != 0:
            self.current_page -= 1
            self.view_change()
            
    def addNominalPath(self):
        self.nominal_path = QFileDialog.getExistingDirectory(self, 'Select Nominal Directory')
        if self.nominal_path:
            self.nominalComboBox.addItem(self.nominal_path)
            self.nominalComboBox.setCurrentIndex(self.nominalComboBox.count() - 1)
        
    def addDefectPath(self):
        self.defect_path = QFileDialog.getExistingDirectory(self, 'Select Defect Directory')
        if self.defect_path:
            self.defectComboBox.addItem(self.defect_path)
            self.defectComboBox.setCurrentIndex(self.defectComboBox.count() - 1)
            
    def on_nominalComboBox_changed(self):
        self.nominal_path = self.nominalComboBox.currentText()
        
    def on_defectComboBox_changed(self):
        self.defect_path = self.defectComboBox.currentText()
        
    def nextNominalImage(self):
        if self.displayed_nominal_img_idx < len(self.nominal_images) - 1:
            self.displayed_nominal_img_idx += 1
            self.display_nominal_img()
            
    def prevNominalImage(self):
        if self.displayed_nominal_img_idx > 0:
            self.displayed_nominal_img_idx -= 1
            self.display_nominal_img()