from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from utilities import *

import os
import cv2
import time

class MyApp(QtWidgets.QMainWindow):
    
    loading_progress_signal = pyqtSignal(int) # used to communicate with loading dialog box
    
    def __init__(self):
        super().__init__()
        
        self.pages_dict = {
            "home": 0,
            "folder paths": 1,
            "nominal samples": 2,
            "defect samples": 3
        }
        
        loadUi(r'assets\start_window.ui', self)  # Load the .ui file
        
        self.reset()
        
        self.num_widgets = self.stackedWidget.count()
        self.stackedWidget.setCurrentIndex(self.current_page)
        
        # Button events
        self.nextButton.clicked.connect(self.next)
        self.backButton.clicked.connect(self.back)
        self.addNominalPathButton.clicked.connect(self.addNominalPath)
        self.addDefectPathButton.clicked.connect(self.addDefectPath)
        self.nextNominalImageButton.clicked.connect(self.nextNominalImage)
        self.prevNominalImageButton.clicked.connect(self.prevNominalImage)
        self.nextDefectImageButton.clicked.connect(self.nextDefectImage)
        self.prevDefectImageButton.clicked.connect(self.prevDefectImage)
        
        # ComboBox events
        self.nominalComboBox.currentIndexChanged.connect(self.on_nominalComboBox_changed)
        self.defectComboBox.currentIndexChanged.connect(self.on_defectComboBox_changed)
        
        self.show()
        
    def reset(self):
        """
        Resets the application state to its initial values.
        
        Parameters:
            None
        
        Returns:
            None
        """
        
        self.current_page = 0
        
        self.nominal_path = None
        self.defect_path = None
        
        self.no_of_nominal_images = 0
        self.no_of_defect_images = 0
        self.total_images_loaded = 0
        
        self.displayed_nominal_img_idx = 0
        self.displayed_defect_img_idx = 0
        
        self.nominal_images = []
        self.defect_images = []
        
        # self.nominal_images_mutex = QMutex()
        # self.defect_images_mutex = QMutex()
        self.total_images_mutex = QMutex()
        
        self.nominalComboBox.clear()
        self.defectComboBox.clear()
        
    def view_change(self):
        """
    	Changes the current view of the application by setting the current index of the stacked widget.
    	
    	Parameters:
    		None
    	
    	Returns:
    		None
    	"""
          
        self.stackedWidget.setCurrentIndex(self.current_page)
        
    def load_images(self, images_paths, dst, mutex):
        """
        Loads a list of images from the provided paths.

        Parameters:
            images_paths (list): A list of paths to the images to be loaded.

        Returns:
            list: A list of loaded images.
        """
                   
        for image_path in images_paths:
            image = cv2.imread(image_path)
            mutex.lock()
            try:
                dst.append(image)
                self.total_images_loaded += 1
                self.loading_progress_signal.emit(self.total_images_loaded)
                # print(self.total_images_loaded)
            finally:
                mutex.unlock()
            
        return dst
    
    def display_img(self, image, label:QLabel):                      
        """
        Display an image on a QLabel.

        Args:
            image (numpy.ndarray): The image to be displayed.
            label (QtWidgets.QLabel): The QLabel on which the image will be displayed.

        Returns:
            None
        """
        disp_img = ConvertCVArray2QPixmap(image)
        label.setPixmap(disp_img)
        
    def display_nominal_img(self):
        self.display_img(self.nominal_images[self.displayed_nominal_img_idx], self.nominalImageDisplayLabel)
        
    def display_defect_img(self):
        self.display_img(self.defect_images[self.displayed_defect_img_idx], self.defectImageDisplayLabel)
               
    # EVENT HANDLERS 
    def next(self):
        if self.current_page+1 == self.pages_dict["nominal samples"]:
            # need to load images from paths
            if self.nominal_path and self.defect_path:
                nominal_images_paths = [os.path.join(self.nominal_path, file) for file in os.listdir(self.nominal_path)]
                defect_images_paths = [os.path.join(self.defect_path, file) for file in os.listdir(self.defect_path)]
                
                self.no_of_nominal_images = len(nominal_images_paths)
                self.no_of_defect_images = len(defect_images_paths)
                
                self.nominal_loader_thread = WorkerThread(1, self.load_images, nominal_images_paths, self.nominal_images, mutex=self.total_images_mutex)
                self.nominal_loader_thread.start()

                self.defect_loader_thread = WorkerThread(2, self.load_images, defect_images_paths, self.defect_images, mutex=self.total_images_mutex)
                self.defect_loader_thread.start()
                
                dialog_box = LoadingDialog(self, "Loading images", self.no_of_nominal_images+self.no_of_defect_images)
                dialog_box.exec()
                
                self.nominal_loader_thread.wait()
                self.defect_loader_thread.wait()
                self.display_nominal_img()
                
        if self.current_page+1 == self.pages_dict["defect samples"]:
                self.display_defect_img()
                
                      
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
            
    def nextDefectImage(self):
        if self.displayed_defect_img_idx < len(self.defect_images) - 1:
            self.displayed_defect_img_idx += 1
            self.display_defect_img()
            
    def prevDefectImage(self):
        if self.displayed_defect_img_idx > 0:
            self.displayed_defect_img_idx -= 1
            self.display_defect_img()
            
class LoadingDialog(QDialog):
    def __init__(self, parent, label_text: str, completion_count: int):
        super().__init__()
                
        loadUi(r'assets\loading_dialog.ui', self)  # Load the .ui file
        
        self.setWindowTitle(label_text)
        
        self.loadingScreenLabel.setText("Loading...")
        
        self.progressBar.setValue(0)
        self.progressBar.setRange(0, completion_count)
        
        parent.loading_progress_signal.connect(self.update_progress_bar)
                
    def update_progress_bar(self, number):      
        self.progressBar.setValue(number)
        
        if number == self.progressBar.maximum():
            self.accept()
                                    
class WorkerThread(QThread):
    def __init__(self, thread_id, func, *args, **kwargs):
        super().__init__()
        self.thread_id = thread_id
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)
