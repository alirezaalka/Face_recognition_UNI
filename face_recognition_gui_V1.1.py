import sys
import cv2
import numpy as np
import os
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QInputDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QSize

########## KNN CODE ############
def distance(v1, v2):
    # Euclidean
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
    confidence = output[1][index] / k  # Calculate confidence as fraction of k
    return output[0][index], confidence
################################

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        # Apply modern dark theme stylesheet with enhanced styling
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

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Video feed label
        self.video_label = QLabel(self)
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label, stretch=1)

        # Result label for recognition output
        self.result_label = QLabel("Waiting for faces...", self)
        self.result_label.setObjectName("result_label")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        # Button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        # Buttons
        self.detect_button = QPushButton("Detect Face", self)
        self.detect_button.clicked.connect(self.detect_face)
        self.button_layout.addWidget(self.detect_button)

        self.live_button = QPushButton("Live Detection", self)
        self.live_button.clicked.connect(self.toggle_live_detection)
        self.button_layout.addWidget(self.live_button)

        self.register_button = QPushButton("Register New Face", self)
        self.register_button.clicked.connect(self.start_register_face)
        self.button_layout.addWidget(self.register_button)

        self.pause_button = QPushButton("Pause Video", self)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)

        self.about_button = QPushButton("About Me", self)
        self.about_button.clicked.connect(self.show_about)
        self.button_layout.addWidget(self.about_button)

        # Status label
        self.status_label = QLabel("Status: Idle", self)
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Button animations
        for button in [self.detect_button, self.live_button, self.register_button, self.pause_button, self.about_button]:
            anim = QPropertyAnimation(button, b"size")
            anim.setDuration(100)
            anim.setStartValue(button.size())
            anim.setEndValue(QSize(button.width() + 10, button.height() + 5))
            button.enterEvent = lambda event, b=button, a=anim: a.start()
            button.leaveEvent = lambda event, b=button, a=anim: a.setDirection(QPropertyAnimation.Direction.Backward) or a.start()

        # Initialize OpenCV and dataset
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.dataset_path = "./face_dataset/"
        self.face_data = []
        self.labels = []
        self.class_id = 0
        self.names = {}
        self.load_dataset()

        # State variables
        self.live_running = False
        self.paused = False
        self.registering = False
        self.frame = None
        self.display_frame = None
        self.results = []
        self.lock = threading.Lock()

        # Timer for video feed update
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

        # Thread for face recognition and registration
        self.recognition_thread = None
        self.recognition_running = False
        self.register_thread = None
        self.register_count = 0
        self.register_data = []

    def load_dataset(self):
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

            # Convert frame to RGB and display
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def run_face_recognition(self, frame):
        if frame is None or self.trainset is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []

        for face in faces:
            x, y, w, h = face
            offset = 5
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            if face_section.size == 0:
                continue
            face_section = cv2.resize(face_section, (100, 100))
            label, confidence = knn(self.trainset, face_section.flatten())
            name = self.names.get(int(label), "Unknown")
            results.append((x, y, w, h, name, confidence))
        
        return results

    def recognition_loop(self):
        while self.recognition_running and self.cap.isOpened():
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
            if frame is not None:
                results = self.run_face_recognition(frame)
                with self.lock:
                    self.results = results
                num_faces = len(results)
                names = ", ".join([f"{name} ({conf:.2f})" for _, _, _, _, name, conf in results]) if results else "No faces detected"
                self.result_label.setText(f"Faces: {names}")
                self.status_label.setText(f"Status: Detected {num_faces} face(s)")
            else:
                self.result_label.setText("Faces: No frame captured")
                self.status_label.setText("Status: No frame captured")

    def register_face(self):
        self.register_count = 0
        self.register_data = []
        self.registering = True
        self.status_label.setText("Status: Capturing face data... Look at the camera.")

        while self.register_count < 10 and self.registering and self.cap.isOpened():
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:  # Ensure single face for registration
                x, y, w, h = faces[0]
                offset = 5
                face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
                if face_section.size != 0:
                    face_section = cv2.resize(face_section, (100, 100))
                    self.register_data.append(face_section.flatten())
                    self.register_count += 1
                    self.status_label.setText(f"Status: Captured {self.register_count}/10 faces")
            # Short delay to avoid overloading
            cv2.waitKey(100)

        self.registering = False
        if self.register_count == 10:
            name, ok = QInputDialog.getText(self, "Enter Name", "Enter the name for this face:")
            if ok and name.strip():
                if not os.path.exists(self.dataset_path):
                    os.makedirs(self.dataset_path)
                np.save(os.path.join(self.dataset_path, f"{name.strip()}.npy"), np.array(self.register_data))
                self.load_dataset()  # Reload dataset
                self.status_label.setText(f"Status: Registered {name} successfully")
                self.result_label.setText("Waiting for faces...")
            else:
                self.status_label.setText("Status: Registration cancelled")
        else:
            self.status_label.setText("Status: Registration failed - insufficient face data")

    def start_register_face(self):
        if self.registering:
            self.status_label.setText("Status: Already registering a face")
            return
        if self.paused:
            self.toggle_pause()  # Resume video if paused
        if self.live_running:
            self.toggle_live_detection()  # Stop live detection
        self.register_thread = threading.Thread(target=self.register_face, daemon=True)
        self.register_thread.start()

    def detect_face(self):
        if self.trainset is None:
            self.status_label.setText("Status: No dataset available")
            return

        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        
        if frame is not None:
            results = self.run_face_recognition(frame)
            num_faces = len(results)
            names = ", ".join([f"{name} ({conf:.2f})" for _, _, _, _, name, conf in results]) if results else "No faces detected"
            self.result_label.setText(f"Faces: {names}")
            self.status_label.setText(f"Status: Single detection completed ({num_faces} face(s))")
            with self.lock:
                self.display_frame = self.draw_faces(frame.copy(), results)
        else:
            self.status_label.setText("Status: No frame captured")

    def toggle_live_detection(self):
        if self.trainset is None:
            self.status_label.setText("Status: No dataset available")
            return

        if self.registering:
            self.status_label.setText("Status: Cannot start live detection while registering")
            return

        self.live_running = not self.live_running
        if self.live_running:
            self.live_button.setText("Stop Live Detection")
            self.recognition_running = True
            self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.recognition_thread.start()
            self.status_label.setText("Status: Live detection running")
        else:
            self.live_button.setText("Live Detection")
            self.recognition_running = False
            if self.recognition_thread:
                self.recognition_thread.join()
            self.result_label.setText("Waiting for faces...")
            self.status_label.setText("Status: Live detection stopped")
            with self.lock:
                self.results = []

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume Video" if self.paused else "Pause Video")
        self.status_label.setText(f"Status: {'Paused' if self.paused else 'Video resumed'}")
        if self.paused:
            self.live_running = False
            self.recognition_running = False
            self.registering = False
            self.live_button.setText("Live Detection")
            if self.recognition_thread:
                self.recognition_thread.join()
            if self.register_thread:
                self.register_thread.join()
            self.result_label.setText("Waiting for faces...")
            self.results = []

    def show_about(self):
        self.result_label.setText("About: Face Recognition App v1.2\nDeveloped by Arpit Dhamija")
        self.status_label.setText("Status: About page displayed")

    def closeEvent(self, event):
        self.recognition_running = False
        self.registering = False
        if self.recognition_thread:
            self.recognition_thread.join()
        if self.register_thread:
            self.register_thread.join()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())