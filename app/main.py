import os
import sys
import time
import cv2
import numpy as np
import speech_recognition as sr
import torch
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# === SETTINGS / НАСТРОЙКИ ===
STANDARD_CLASSES = [0, 24, 26, 32, 63, 67, 73] 

# Названия для отображения в интерфейсе
CLASS_NAMES = {
    'ru': {
        'cell phone': '📱 Телефон', 'laptop': '💻 Ноутбук', 'person': '👤 Человек',
        'backpack': '🎒 Рюкзак', 'handbag': '👜 Сумка', 'book': '📖 Книга', 
        'sports ball': '🎾 Мячик', 'pen': '🖊️ Ручка'
    },
    'en': {
        'cell phone': '📱 Phone', 'laptop': '💻 Laptop', 'person': '👤 Person',
        'backpack': '🎒 Backpack', 'handbag': '👜 Handbag', 'book': '📖 Book', 
        'sports ball': '🎾 Ball', 'pen': '🖊️ Pen'
    }
}

# Единая карта голоса (RU + EN)
VOICE_MAP = {
    "телефон": 67, "mobile": 67, "phone": 67,
    "ноутбук": 63, "laptop": 63, "computer": 63,
    "человек": 0, "person": 0, "human": 0,
    "сумка": 24, "bag": 24, "backpack": 24,
    "книга": 73, "book": 73, "notebook": 73,
    "мяч": 32, "ball": 32,
    "ручка": 418, "pen": 418
}

class VoiceThread(QThread):
    command_sig = pyqtSignal(str, float)
    def __init__(self):
        super().__init__()
        self.running = True
        self.recognizer = sr.Recognizer()

    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    start_t = time.time()
                    # Распознаем RU по умолчанию
                    text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                    self.command_sig.emit(text, (time.time() - start_t) * 1000)
                except: continue

class InferenceThread(QThread):
    frame_sig = pyqtSignal(np.ndarray)
    stats_sig = pyqtSignal(dict)
    det_sig = pyqtSignal(str, float)

    def __init__(self, model_path):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Загрузка модели Lavoris
        self.model = YOLO(model_path).to(self.device)
        self.running = False
        self.filter_classes = None 

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            
            t1 = time.time()
            results = self.model.predict(
                frame, conf=0.35, verbose=False, stream=True,
                classes=self.filter_classes, device=self.device
            )
            
            for r in results:
                res_frame = r.plot()
                fps = 1 / (time.time() - t1)
                cv2.putText(res_frame, f"FPS: {fps:.1f} ({self.device.upper()})", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 65), 3)

                self.stats_sig.emit({'fps': fps, 'count': len(r.boxes)})
                if len(r.boxes) > 0:
                    name = self.model.names[int(r.boxes[0].cls[0])]
                    self.det_sig.emit(name, float(r.boxes[0].conf[0]))
                self.frame_sig.emit(res_frame)

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'): self.cap.release()
        self.wait()

class LavorisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = 'ru' 
        self.setWindowTitle("Lavoris AI Items-Detector v2.5")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet("background-color: #121212; color: #ffffff;")
        
        # Обновленный путь к твоей модели
        model_name = "Lavoris_items-detector.pt"
        path = os.path.join("models", model_name)
        self.yolo = InferenceThread(path if os.path.exists(path) else model_name)
        
        self.yolo.frame_sig.connect(self.update_video)
        self.yolo.stats_sig.connect(self.update_stats)
        self.yolo.det_sig.connect(self.update_detection)

        self.init_ui()

        self.voice = VoiceThread()
        self.voice.command_sig.connect(self.handle_voice)
        self.voice.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)

        # --- LEFT PANEL ---
        left_lay = QVBoxLayout()
        
        header_lay = QHBoxLayout()
        self.title_label = QLabel("⚡ LAVORIS AI SYSTEM ⚡")
        self.title_label.setStyleSheet("font-size: 20px; color: #00ff41; font-weight: bold;")
        
        # УЛУЧШЕННАЯ КНОПКА СМЕНЫ ЯЗЫКА
        self.lang_btn = QPushButton("🇷🇺 RU  |  EN 🇬🇧")
        self.lang_btn.setFixedSize(140, 40)
        self.lang_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2a5c8a, stop: 1 #1e3a5f);
                color: #ffffff;
                border: 2px solid #4a9eff;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3a6c9a, stop: 1 #2e4a6f);
                border: 2px solid #7ab5ff;
                font-size: 14px;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1a4c7a, stop: 1 #0e2a4f);
            }
        """)
        self.lang_btn.clicked.connect(self.toggle_language)
        
        header_lay.addWidget(self.title_label)
        header_lay.addStretch()
        header_lay.addWidget(self.lang_btn)
        left_lay.addLayout(header_lay)

        self.video_label = QLabel("WAITING FOR 'START' COMMAND")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #000; border: 2px solid #00ff41; font-size: 22px; color: #00ff41;")
        left_lay.addWidget(self.video_label, 8)
        
        self.log_box = QTextEdit()
        self.log_box.setFixedHeight(160)
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background: #050505; color: #00ff41; font-family: 'Consolas'; border: 1px solid #333;")
        left_lay.addWidget(self.log_box)
        main_lay.addLayout(left_lay, 3)

        # --- RIGHT PANEL ---
        right_panel = QVBoxLayout()
        
        self.mode_gb = QGroupBox("ACTIVE MODE")
        self.mode_gb.setStyleSheet("color: #00ff41; font-weight: bold;")
        mode_lay = QVBoxLayout()
        self.mode_label = QLabel("ALL OBJECTS")
        self.mode_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.mode_label.setStyleSheet("color: #00d4ff;")
        mode_lay.addWidget(self.mode_label)
        self.mode_gb.setLayout(mode_lay)
        right_panel.addWidget(self.mode_gb)

        self.table = QTableWidget(12, 2)
        self.table.setHorizontalHeaderLabels(["Object", "Conf"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("background: #1e1e1e; color: white;")
        right_panel.addWidget(self.table)

        self.fps_lab = QLabel("System FPS: 0.0")
        right_panel.addWidget(self.fps_lab)

        self.btn_main = QPushButton("🚀 START")
        self.btn_main.setFixedHeight(60)
        self.btn_main.setStyleSheet("background: #00ff41; color: black; font-weight: bold; font-size: 18px; border-radius: 10px;")
        self.btn_main.clicked.connect(self.toggle_system)
        right_panel.addWidget(self.btn_main)

        main_lay.addLayout(right_panel, 1)
        self.update_ui_text()

    def toggle_language(self):
        self.lang = 'en' if self.lang == 'ru' else 'ru'
        # Обновляем текст на кнопке
        if self.lang == 'ru':
            self.lang_btn.setText("🇷🇺 RU  |  EN 🇬🇧")
        else:
            self.lang_btn.setText("🇬🇧 EN  |  RU 🇷🇺")
        self.update_ui_text()
        self.log("Language changed" if self.lang == 'en' else "Язык изменен")

    def update_ui_text(self):
        t = {
            'ru': ["ВСЕ ОБЪЕКТЫ", "АКТИВНЫЙ РЕЖИМ", "Объект", "СТАРТ", "СТОП", "Жду команду 'На СТАРТ'"],
            'en': ["ALL OBJECTS", "ACTIVE MODE", "Object", "START", "STOP", "Waiting for 'START'"]
        }[self.lang]
        
        self.mode_label.setText(t[0])
        self.mode_gb.setTitle(t[1])
        self.table.setHorizontalHeaderLabels([t[2], "Conf"])
        self.btn_main.setText(t[3] if not self.yolo.isRunning() else t[4])
        if not self.yolo.isRunning():
            self.video_label.setText(t[5])

    def log(self, text):
        self.log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def handle_voice(self, cmd, speed):
        self.log(f"🎤 Voice ({speed:.0f}ms): {cmd}")
        if any(w in cmd for w in ["старт", "пуск", "start", "run"]): 
            self.control_start()
        elif any(w in cmd for w in ["стоп", "пауза", "stop", "pause", "wait"]): 
            self.control_stop()
        elif any(w in cmd for w in ["стандарт", "standard", "default"]):
            self.yolo.filter_classes = STANDARD_CLASSES
            self.mode_label.setText("MODE: STANDARD" if self.lang == 'en' else "РЕЖИМ: СТАНДАРТ")
        elif any(w in cmd for w in ["clear", "очистить", "сброс", "reset"]):
            self.yolo.filter_classes = None
            self.mode_label.setText("MODE: ALL" if self.lang == 'en' else "РЕЖИМ: ВСЕ ОБЪЕКТЫ")
        else:
            for word, class_id in VOICE_MAP.items():
                if word in cmd:
                    self.yolo.filter_classes = [class_id]
                    self.mode_label.setText(f"FOCUS: {word.upper()}")
                    break

    def toggle_system(self):
        if self.yolo.isRunning(): self.control_stop()
        else: self.control_start()

    def control_start(self):
        if not self.yolo.isRunning():
            self.yolo.start()
            self.btn_main.setText("⏹ STOP" if self.lang == 'en' else "⏹ СТОП")
            self.btn_main.setStyleSheet("background: #ff4141; color: white; font-weight: bold; font-size: 18px; border-radius: 10px;")

    def control_stop(self):
        if self.yolo.isRunning():
            self.yolo.stop()
            self.update_ui_text()
            self.btn_main.setStyleSheet("background: #00ff41; color: black; font-weight: bold; font-size: 18px; border-radius: 10px;")

    def update_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def update_stats(self, s):
        self.fps_lab.setText(f"System FPS: {s['fps']:.1f} | Found: {s['count']}")

    def update_detection(self, name, conf):
        display_name = CLASS_NAMES[self.lang].get(name, name)
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(display_name))
        self.table.setItem(0, 1, QTableWidgetItem(f"{conf:.2f}"))
        if self.table.rowCount() > 12: self.table.removeRow(12)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LavorisApp()
    win.show()
    sys.exit(app.exec_())