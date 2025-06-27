import sys
import cv2
import numpy as np
import os
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

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
    return output[0][index]

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: #ffffff; font-family: Segoe UI; font-size: 14px; }
            QPushButton { 
                background-color: #4CAF50; 
                color: #ffffff; 
                font-family: Segoe UI; 
                font-size: 12px; 
                padding: 10px; 
                border: none; 
                border-radius: 5px; 
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3a3a3a;")
        self.main_layout.addWidget(self.video_label, stretch=1)

        self.result_label = QLabel("Waiting for face...", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        self.detect_button = QPushButton("Detect Face", self)
        self.detect_button.clicked.connect(self.detect_face)
        self.button_layout.addWidget(self.detect_button)

        self.live_button = QPushButton("Live Detection", self)
        self.live_button.clicked.connect(self.toggle_live_detection)
        self.button_layout.addWidget(self.live_button)

        self.about_button = QPushButton("About Me", self)
        self.about_button.clicked.connect(self.show_about)
        self.button_layout.addWidget(self.about_button)

        self.status_label = QLabel("Status: Idle", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.dataset_path = "./face_dataset/"
        self.face_data = []
        self.labels = []
        self.class_id = 0
        self.names = {}

        for fx in os.listdir(self.dataset_path):
            if fx.endswith('.npy'):
                self.names[self.class_id] = fx[:-4]
                data_item = np.load(self.dataset_path + fx)
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

        self.live_running = False
        self.frame = None
        self.lock = threading.Lock()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

    def update_video(self):
        if not self.cap.isOpened():
            self.status_label.setText("Status: Camera not detected")
            return

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))

            if self.live_running and self.trainset is not None:
                result = self.run_face_recognition(self.frame)
                self.result_label.setText(f"Person: {result}")
        else:
            self.status_label.setText("Status: Failed to capture frame")

    def run_face_recognition(self, frame):
        if frame is None or self.trainset is None:
            return "Unknown"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for face in faces:
            x, y, w, h = face
            offset = 5
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            if face_section.size == 0:
                continue
            face_section = cv2.resize(face_section, (100, 100))
            out = knn(self.trainset, face_section.flatten())
            return self.names.get(int(out), "Unknown")
        
        return "No face detected"

    def detect_face(self):
        if self.trainset is None:
            self.status_label.setText("Status: No dataset available")
            return
        
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        
        if frame is not None:
            result = self.run_face_recognition(frame)
            self.result_label.setText(f"Person: {result}")
            self.status_label.setText("Status: Single detection completed")
        else:
            self.status_label.setText("Status: No frame captured")

    def toggle_live_detection(self):
        if self.trainset is None:
            self.status_label.setText("Status: No dataset available")
            return
        
        self.live_running = not self.live_running
        if self.live_running:
            self.live_button.setText("Stop Live Detection")
            self.status_label.setText("Status: Live detection running")
        else:
            self.live_button.setText("Live Detection")
            self.status_label.setText("Status: Live detection stopped")

    def show_about(self):
        self.result_label.setText("About: Face Recognition App v1.0\nDeveloped by Arpit Dhamija")
        self.status_label.setText("Status: About page displayed")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())