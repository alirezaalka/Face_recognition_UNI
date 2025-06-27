import sys
import cv2
import numpy as np
import os
import threading
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QInputDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QSize, pyqtSignal

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    confidence = output[1][index] / k
    return output[0][index], confidence

class FaceRecognitionGUI(QMainWindow):
    prompt_name_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        logging.debug("Initializing FaceRecognitionGUI")
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { 
                color: #ffffff; 
                font-family: Segoe UI; 
                font-size: 14px; 
                padding: 5px; 
            }
            QPushButton { 
                background-color: #4CAF50; 
                color: #ffffff; 
                font-family: Segoe UI; 
                font-size: 12px; 
                padding: 10px; 
                border: none; 
                border-radius: 5px; 
                min-width: 120px; 
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
            #video_label { 
                background-color: #1e1e1e; 
                border: 1px solid #3a3a3a; 
                border-radius: 5px; 
            }
            #result_label { 
                background-color: #1e1e1e; 
                border-radius: 5px; 
                padding: 10px; 
            }
            #status_label { 
                color: #cccccc; 
                font-size: 12px; 
            }
        """)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.video_label = QLabel(self.central_widget)
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label, stretch=1)

        self.result_label = QLabel("Waiting for faces...", self.central_widget)
        self.result_label.setObjectName("result_label")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        self.detect_button = QPushButton("Detect Face", self.central_widget)
        self.detect_button.clicked.connect(self.detect_face)
        self.button_layout.addWidget(self.detect_button)

        self.live_button = QPushButton("Live Detection", self.central_widget)
        self.live_button.clicked.connect(self.toggle_live_detection)
        self.button_layout.addWidget(self.live_button)

        self.register_button = QPushButton("Register New Face", self.central_widget)
        self.register_button.clicked.connect(self.start_register_face)
        self.button_layout.addWidget(self.register_button)

        self.pause_button = QPushButton("Pause Video", self.central_widget)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)

        self.about_button = QPushButton("About Me", self.central_widget)
        self.about_button.clicked.connect(self.show_about)
        self.button_layout.addWidget(self.about_button)

        self.status_label = QLabel("Status: Idle", self.central_widget)
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        for button in [self.detect_button, self.live_button, self.register_button, self.pause_button, self.about_button]:
            anim = QPropertyAnimation(button, b"size")
            anim.setDuration(100)
            anim.setStartValue(button.size())
            anim.setEndValue(QSize(button.width() + 10, button.height() + 5))
            button.enterEvent = lambda event, b=button, a=anim: a.start()
            button.leaveEvent = lambda event, b=button, a=anim: a.setDirection(QPropertyAnimation.Direction.Backward) or a.start()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open webcam")
            self.status_label.setText("Status: Camera not detected")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        if self.face_cascade.empty():
            logging.error("Failed to load haarcascade_frontalface_alt.xml")
            self.status_label.setText("Status: Haar cascade file not found")
        self.dataset_path = "./face_dataset/"
        self.face_data = []
        self.labels = []
        self.class_id = 0
        self.names = {}
        self.load_dataset()

        self.live_running = False
        self.paused = False
        self.registering = False
        self.frame = None
        self.display_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.registered_data = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

        self.recognition_thread = None
        self.recognition_running = False
        self.register_thread = None
        self.register_count = 0

        self.prompt_name_signal.connect(self.prompt_name)
        logging.debug("Initialization completed")

    def load_dataset(self):
        self.face_data = []
        self.labels = []
        self.class_id = 0
        self.names = {}
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        for fx in os.listdir(self.dataset_path):
            if fx.endswith('.npy'):
                self.names[self.class_id] = fx[:-4]
                data_item = np.load(os.path.join(self.dataset_path, fx))
                self.face_data.append(data_item)
                target = self.class_id * np.ones((data_item.shape[0],))
                self.class_id += 1
                self.labels.append(target)

        if self.face_data:
            self.face_dataset = np.concatenate(self.face_data, axis=0)
            self.face_labels = np.concatenate(self.labels, axis=0).reshape((-1, 1))
            self.trainset = np.concatenate((self.face_dataset, self.face_labels), axis=1)
        else:
            self.trainset = None
            self.status_label.setText("Status: No dataset found in ./face_dataset/")

    def update_video(self):
        if self.paused or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame = frame.copy()
                if self.registering:
                    frame = self.draw_register_prompt(frame)
                elif self.live_running and self.results:
                    frame = self.draw_faces(frame, self.results)
                self.display_frame = frame.copy()

            frame_rgb = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))
        else:
            self.status_label.setText("Status: Failed to capture frame")

    def draw_faces(self, frame, results):
        for (x, y, w, h, name, confidence) in results:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame

    def draw_register_prompt(self, frame):
        cv2.putText(frame, f"Capturing face {self.register_count}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w