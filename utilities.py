import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap

def ConvertQPixmap2CVArray (QPixmapObject):
    """ Convert a QPixmap object to a type opencv can work with"""
    
    # Convert QPixmap to OpenCV format (numpy array)
    image_np = QPixmapObject.toImage().convertToFormat(QImage.Format_RGB888)
    
    width = image_np.width()
    height = image_np.height()
    buffer = image_np.bits()
    buffer.setsize(height * width * 3)  # Set the buffer size according to image dimensions and channels
    img_arr = np.frombuffer(buffer, np.uint8).reshape((height, width, 3))
    
    image_np = cv2.cvtColor(img_arr,  cv2.COLOR_RGB2BGR)
      
    return image_np
    
def ConvertCVArray2QPixmap (CVArray):
    """ Convert opencv array into QPixmap object """
    
    # Convert the NumPy array to an 8-bit unsigned integer (uint8) data type
    numpy_array = CVArray.astype(np.uint8)
    
    height, width, channels = CVArray.shape 
    bytes_per_line = channels * width
    
    # qimg = QImage(CVArray.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qimg = QImage(numpy_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    
    return pixmap