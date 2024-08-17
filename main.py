import sys

from PyQt5.QtWidgets import QApplication

from app import *

def main():
    app = QApplication(sys.argv)
    window = MyApp()
    app.exec()
       
if __name__ == "__main__":
    main()