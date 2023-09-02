import sys
from PyQt6.QtWidgets import QApplication, QDialog
from ui_imagedialog  import Ui_Dialog





app = QApplication(sys.argv)
window = QDialog()
ui = Ui_Dialog()
ui.setupUi(window)

window.show()
sys.exit(app.exec())