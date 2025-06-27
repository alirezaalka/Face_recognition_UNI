import sys
import cv2
import numpy as np
import os
import threading
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QInputDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QSize, pyqtSignal

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Signal to trigger name prompt on the main thread
    prompt_name_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        logging.debug("Initializing FaceRecognitionGUI")
        try:
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
            self.central_widget = QWidget(self)
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(20, 20, 20, 20)

            # Video feed label
            self.video_label = QLabel(self.central_widget)
            self.video_label.setObjectName("video_label")
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setMinimumSize(640, 480)
            self.main_layout.addWidget(self.video_label, stretch=1)

            # Result label for recognition output
            self.result_label = QLabel("Waiting for faces...", self.central_widget)
            self.result_label.setObjectName("result_label")
            self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_layout.addWidget(self.result_label)

            # Button layout
            self.button_layout = QHBoxLayout()
            self.button_layout.setSpacing(10)
            self.main_layout.addLayout(self.button_layout)

            # Buttons
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

            # Status label
            self.status_label = QLabel("Status: Idle", self.central_widget)
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
            logging.debug("Loading OpenCV and dataset")
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

            # State variables
            self.live_running = False
            self.paused = False
            self.registering = False
            self.frame = None
            self.display_frame = None
            self.results = []
            self.lock = threading.Lock()
            self.registered_data = None  # Store data for name prompt

            # Timer for video feed update
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_video)
            self.timer.start(30)

            # Thread for face recognition and registration
            self.recognition_thread = None
            self.recognition_running = False
            self.register_thread = None
            self.register_count = 0

            # Connect signal for name prompt
            self.prompt_name_signal.connect(self.prompt_name)
            logging.debug("Initialization completed")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def load_dataset(self):
        logging.debug("Loading dataset")
        try:
            self.face_data = []
            self.labels = []
            self.class_id = 0
            self.names = {}
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)
                logging.info(f"Created dataset directory: {self.dataset_path}")
            for fx in os.listdir(self.dataset_path):
                if fx.endswith('.npy'):
                    try:
                        self.names[self.class_id] = fx[:-4]
                        data_item = np.load(os.path.join(self.dataset_path, fx))
                        self.face_data.append(data_item)
                        target = self.class_id * np.ones((data_item.shape[0],))
                        self.class_id += 1
                        self.labels.append(target)
                    except Exception as e:
                        logging.error(f"Failed to load {fx}: {str(e)}")
                        continue
            
            if self.face_data:
                self.face_dataset = np.concatenate(self.face_data, axis=0)
                self.face_labels = np.concatenate(self.labels, axis=0).reshape((-1, 1))
                self.trainset = np.concatenate((self.face_dataset, self.face_labels), axis=1)
                logging.debug("Dataset loaded successfully")
            else:
                self.trainset = None
                self.status_label.setText("Status: No dataset found in ./face_dataset/")
                logging.warning("No dataset found in ./face_dataset/")
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            self.status_label.setText("Status: Error loading dataset")
            self.trainset = None

    def update_video(self):
        try:
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
                logging.error("Failed to capture frame")
        except Exception as e:
            logging.error(f"Error in update_video: {str(e)}")
            self.status_label.setText("Status: Error updating video")

    def draw_faces(self, frame, results):
        try:
            for (x, y, w, h, name, confidence) in results:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            return frame
        except Exception as e:
            logging.error(f"Error in draw_faces: {str(e)}")
            return frame

    def draw_register_prompt(self, frame):
        try:
            cv2.putText(frame, f"Capturing face {self.register_count}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return frame
        except Exception as e:
            logging.error(f"Error in draw_register_prompt: {str(e)}")
            return frame

    def run_face_recognition(self, frame):
        try:
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
        except Exception as e:
            logging.error(f"Error in run_face_recognition: {str(e)}")
            return []

    def recognition_loop(self):
        try:
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
        except Exception as e:
            logging.error(f"Error in recognition_loop: {str(e)}")

    def register_face(self):
        try:
            logging.debug("Starting face registration process")
            self.register_count = 0
            self.registered_data = []
            self.registering = True
            self.status_label.setText("Status: Capturing face data... Look at the camera.")

            while self.register_count < 10 and self.registering and self.cap.isOpened():
                with self.lock:
                    frame = self.frame.copy() if self.frame is not None else None
                if frame is None:
                    logging.warning("No frame available during registration")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 1:  # Ensure single face for registration
                    x, y, w, h = faces[0]
                    offset = 5
                    face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
                    if face_section.size != 0:
                        face_section = cv2.resize(face_section, (100, 100))
                        self.registered_data.append(face_section.flatten())
                        self.register_count += 1
                        self.status_label.setText(f"Status: Captured {self.register_count}/10 faces")
                        logging.debug(f"Captured face {self.register_count}/10")
                    else:
                        logging.warning("Empty face section detected")
                elif len(faces) == 0:
                    logging.debug("No face detected in frame")
                else:
                    logging.debug(f"Multiple faces detected: {len(faces)}")
                # Short delay to avoid overloading
                cv2.waitKey(100)

            self.registering = False
            logging.debug(f"Registration loop ended: {self.register_count} faces captured")
            if self.register_count == 10:
                logging.debug("Emitting prompt_name_signal")
                self.prompt_name_signal.emit()
            else:
                self.status_label.setText("Status: Registration failed - insufficient face data")
                logging.warning(f"Registration failed: {self.register_count} faces captured")
        except Exception as e:
            self.registering = False
            self.status_label.setText("Status: Registration failed")
            logging.error(f"Error in register_face: {str(e)}")

    def prompt_name(self):
        try:
            logging.debug("Showing name prompt")
            name, ok = QInputDialog.getText(self, "Enter Name", "Enter the name for this face:")
            logging.debug(f"Name prompt result: ok={ok}, name={name}")
            if ok and name.strip():
                try:
                    # Sanitize name to avoid invalid file names
                    name = ''.join(c for c in name.strip() if c.isalnum() or c in (' ', '_')).replace(' ', '_')
                    if not name:
                        raise ValueError("Invalid name after sanitization")
                    if not os.path.exists(self.dataset_path):
                        os.makedirs(self.dataset_path)
                        logging.info(f"Created dataset directory: {self.dataset_path}")
                    np.save(os.path.join(self.dataset_path, f"{name}.npy"), np.array(self.registered_data))
                    self.load_dataset()  # Reload dataset
                    self.status_label.setText(f"Status: Registered {name} successfully")
                    self.result_label.setText("Waiting for faces...")
                    logging.info(f"Registered face: {name}")
                except Exception as e:
                    self.status_label.setText("Status: Failed to save face data")
                    logging.error(f"Failed to save face data: {str(e)}")
            else:
                self.status_label.setText("Status: Registration cancelled")
                logging.info("Registration cancelled by user")
        except Exception as e:
            self.status_label.setText("Status: Failed to prompt name")
            logging.error(f"Error in prompt_name: {str(e)}")

    def start_register_face(self):
        try:
            if self.registering:
                self.status_label.setText("Status: Already registering a face")
                logging.warning("Attempted to start registration while already registering")
                return
            if self.paused:
                self.toggle_pause()  # Resume video if paused
            if self.live_running:
                self.toggle_live_detection()  # Stop live detection
            self.register_thread = threading.Thread(target=self.register_face, daemon=True)
            self.register_thread.start()
            logging.debug("Started face registration thread")
        except Exception as e:
            logging.error(f"Error in start_register_face: {str(e)}")
            self.status_label.setText("Status: Failed to start registration")

    def detect_face(self):
        try:
            if self.trainset is None:
                self.status_label.setText("Status: No dataset available")
                logging.warning("No dataset available for detection")
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
                logging.debug(f"Single detection completed: {num_faces} faces")
            else:
                self.status_label.setText("Status: No frame captured")
                logging.error("No frame captured for detection")
        except Exception as e:
            logging.error(f"Error in detect_face: {str(e)}")
            self.status_label.setText("Status: Detection failed")

    def toggle_live_detection(self):
        try:
            if self.trainset is None:
                self.status_label.setText("Status: No dataset available")
                logging.warning("No dataset available for live detection")
                return

            if self.registering:
                self.status_label.setText("Status: Cannot start live detection while registering")
                logging.warning("Attempted live detection while registering")
                return

            self.live_running = not self.live_running
            if self.live_running:
                self.live_button.setText("Stop Live Detection")
                self.recognition_running = True
                self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
                self.recognition_thread.start()
                self.status_label.setText("Status: Live detection running")
                logging.debug("Started live detection")
            else:
                self.live_button.setText("Live Detection")
                self.recognition_running = False
                if self.recognition_thread:
                    self.recognition_thread.join()
                self.result_label.setText("Waiting for faces...")
                self.status_label.setText("Status: Live detection stopped")
                with self.lock:
                    self.results = []
                logging.debug("Stopped live detection")
        except Exception as e:
            logging.error(f"Error in toggle_live_detection: {str(e)}")
            self.status_label.setText("Status: Live detection failed")

    def toggle_pause(self):
        try:
            self.paused = not self.paused
            self.pause_button.setText("Resume Video" if self.paused else "Pause Video")
            self.status_label.setText(f"Status: {'Paused' if self.paused else 'Video resumed'}")
            logging.debug(f"Video {'paused' if self.paused else 'resumed'}")
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
        except Exception as e:
            logging.error(f"Error in toggle_pause: {str(e)}")
            self.status_label.setText("Status: Pause/resume failed")

    def show_about(self):
        try:
            self.result_label.setText("About: Face Recognition App v1.3\nDeveloped by Alireza Karimi\nGitHub: alirezaalka\nTelegram: @alirezaalka")
            self.status_label.setText("Status: About page displayed")
            logging.debug("Displayed About page")
        except Exception as e:
            logging.error(f"Error in show_about: {str(e)}")
            self.status_label.setText("Status: Failed to display About page")

    def closeEvent(self, event):
        try:
            self.recognition_running = False
            self.registering = False
            if self.recognition_thread:
                self.recognition_thread.join()
            if self.register_thread:
                self.register_thread.join()
            self.cap.release()
            logging.debug("Application closed, resources released")
            event.accept()
        except Exception as e:
            logging.error(f"Error in closeEvent: {str(e)}")
            event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = FaceRecognitionGUI()
        window.show()
        logging.debug("Application started")
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")