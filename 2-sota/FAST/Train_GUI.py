import sys
from csbdeep.utils import normalize
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, \
    QWidget, QTextEdit, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from torch.utils.data import DataLoader

from datasets.dataset import ReadDatasets
from models.Unet_Lite import Unet_Lite
from utils.config import *

from train_in_gui import goTraining


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class TrainingThread(QThread):
    finished = pyqtSignal()
    imageReady = pyqtSignal(np.ndarray, str)

    def __init__(self,parameters, parent = None):
        super(TrainingThread, self).__init__(parent)
        self.is_running = True
        self.parameters = parameters
    def run(self):
        goTraining(self)
        self.finished.emit()

    def stop(self):

        self.is_running = False
        print("\nTraining stopped")
        self.quit()


class TrainingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.redirect_stdout()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Training GUI")
        self.setGeometry(100, 100, 700, 600)

        self.setStyleSheet("""
            QMainWindow {
                background-color: black;
            }
            QLabel, QPushButton, QTextEdit {
                color: white;
                background-color: black;
            }
            QPushButton {
                background-color: #333;  
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555;  
            }
            QTextEdit {
                border: 1px solid #555;
                padding: 5px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        left_layout = QVBoxLayout()

        path_and_buttons_layout = QVBoxLayout()
        self.config_path_label = QLabel("No Train Folder Path selected")

        self.config_path_label.setWordWrap(True)
        path_and_buttons_layout.addWidget(self.config_path_label)

        self.select_folder_btn = QPushButton('Select Train Folder Path')
        path_and_buttons_layout.addWidget(self.select_folder_btn)

        self.start_training_btn = QPushButton('Start Training')
        path_and_buttons_layout.addWidget(self.start_training_btn)

        self.stop_training_btn = QPushButton('Stop Training')
        path_and_buttons_layout.addWidget(self.stop_training_btn)

        left_layout.addLayout(path_and_buttons_layout)

        self.logo_label = QLabel()
        logo_pixmap = QPixmap(r'./FAST_logo.png')
        self.logo_label.setPixmap(logo_pixmap)
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.logo_label)

        self.log_text_edit = QTextEdit()
        # self.log_text_edit.setStyleSheet("color: white;")
        self.log_text_edit.setFixedSize(600, 550)
        left_layout.addWidget(self.log_text_edit)

        self.raw_text_label = QLabel("Raw images:")
        # self.raw_text_label.setStyleSheet("color: white;")
        self.denoised_text_label = QLabel("Denoised images:")
        # self.denoised_text_label.setStyleSheet("color: white;")
        right_layout = QVBoxLayout()
        self.image_label_raw = QLabel()
        self.image_label_denoised = QLabel()

        for label in [self.image_label_raw, self.image_label_denoised]:
            label.setFixedSize(400, 400)
            label.setAlignment(QtCore.Qt.AlignCenter)
            # label.setStyleSheet("border: 1px solid white;")
            black_image = QPixmap(400, 400)
            black_image.fill(QtCore.Qt.black)
            label.setPixmap(black_image)

        right_layout.addWidget(self.raw_text_label)
        right_layout.addWidget(self.image_label_raw)
        right_layout.addWidget(self.denoised_text_label)
        right_layout.addWidget(self.image_label_denoised)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.select_folder_btn.clicked.connect(self.select_folder)
        self.start_training_btn.clicked.connect(self.start_training)
        self.stop_training_btn.clicked.connect(self.stop_training)

        self.model_init()
        # self.central_widget.setStyleSheet("background-color: black;")
        self.show()

    def update_image(self, image_array, figure = 'raw'):

        # image_array = 255 * (image_array - np.min(image_array) / (np.max(image_array) - np.min(image_array)))
        image_array = normalize(image_array, pmin = 3, pmax = 99.8, clip = True) * 255.0
        image_array = image_array.astype(np.uint8)


        image = QImage(image_array.data, image_array.shape[0], image_array.shape[1],
                       QImage.Format_Grayscale8)


        pixmap = QPixmap.fromImage(image)
        if figure == 'raw':
            self.image_label = self.image_label_raw
        else:
            self.image_label = self.image_label_denoised
        scaled_pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)


        self.image_label.setPixmap(scaled_pixmap)

    def model_init(self):

        config_file = "./params.json"
        self.args = json2args(config_file)
        self.model = Unet_Lite(in_channels = self.args.miniBatch_size,
                               out_channels = self.args.miniBatch_size,
                               f_maps = [64, 64, 64],
                               num_groups = 32,
                               final_sigmoid = True).cuda()
        self.model.train()
        print("model has been initialized")

    def redirect_stdout(self):
        sys.stdout = EmittingStream(textWritten = self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten = self.normalOutputWritten)

    def normalOutputWritten(self, text):
        cursor = self.log_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log_text_edit.setTextCursor(cursor)
        self.log_text_edit.ensureCursorVisible()

    def select_folder(self):
        self.start_training_btn.setEnabled(False)
        train_path = QFileDialog.getExistingDirectory(self, "Select Train Folder Path")
        if train_path:
            self.args.train_folder = train_path

            self.config_path_label.setText(f"Train Folder: {train_path}")
            print('train set')
            train_set = ReadDatasets(dataPath = self.args.train_folder,
                                     dataType = self.args.data_type,
                                     dataExtension = self.args.data_extension,
                                     mode = 'train',
                                     denoising_strategy = self.args.denoising_strategy)
            print('val set')
            if not self.args.withGT:
                self.args.val_folder = self.args.train_folder
            val_set = ReadDatasets(dataPath = self.args.val_folder,
                                   dataType = self.args.data_type,
                                   dataExtension = self.args.data_extension,
                                   mode = 'val',
                                   denoising_strategy = self.args.denoising_strategy)
            train_loader = DataLoader(dataset = train_set, batch_size = self.args.batch_size,
                                      drop_last = True, num_workers = self.args.num_workers, pin_memory = True)
            val_loader = DataLoader(dataset = val_set, batch_size = self.args.batch_size,
                                    drop_last = True, num_workers = self.args.num_workers, pin_memory = True)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.start_training_btn.setEnabled(True)
    def start_training(self):
        if hasattr(self, 'args'):
            self.start_training_btn.setEnabled(False)  # Disable the button to prevent multiple clicks
            self.training_thread = TrainingThread(self)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.imageReady.connect(self.update_image)
            self.training_thread.start()
        else:
            QMessageBox.warning(self, "Configuration Required",
                                "Please load a configuration file before starting the training.")

    def stop_training(self):
        if hasattr(self, 'training_thread'):
            self.training_thread.stop()
            self.start_training_btn.setEnabled(True)
            self.stop_training_btn.setEnabled(True)

    def on_training_finished(self):
        self.start_training_btn.setEnabled(True)  # Re-enable the button after training is finished


def main():
    app = QApplication(sys.argv)
    ex = TrainingGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
