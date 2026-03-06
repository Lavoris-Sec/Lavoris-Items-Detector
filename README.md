# 🔍 Lavoris Items Detector (v2.5)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://www.python.org)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-00FF41?style=flat)](https://ultralytics.com)
[![Voice AI](https://img.shields.io/badge/Control-Voice-blueviolet?style=flat)](https://pypi.org/project/SpeechRecognition/)

## 🇷🇺 Описание / 🇺🇸 Description

**RU:** Интеллектуальная система детекции объектов на базе **YOLOv11** с уникальным **голосовым управлением**. Программа позволяет искать предметы (телефон, ноутбук, человек и др.) с помощью микрофона и моментально переключать язык интерфейса (RU/EN). Разработано в рамках экосистемы **Lavoris**.

**EN:** An intelligent object detection system powered by **YOLOv11** with unique **voice control** capabilities. The software allows searching for items (phone, laptop, person, etc.) via microphone and features a real-time language toggle (RU/EN). Developed as part of the **Lavoris** ecosystem.

---

### 🎙️ Voice Commands / Голосовые команды:
| Команда (RU) | Command (EN) | Действие / Action |
|:--- |:--- |:--- |
| **Старт / Пуск** | **Start / Run** | Запуск детекции / Start detection |
| **Стоп / Хватит** | **Stop / Pause** | Остановка / Stop stream |
| **Телефон / Ноутбук** | **Phone / Laptop** | Фокус на предмете / Focus on item |
| **Сброс / Очистить** | **Clear / Reset** | Показать всё / Show all objects |

---

### 📂 Project Structure / Структура проекта:
- `main.py` — Core UI & Logic (Ядро и интерфейс).
- `models/Lavoris_items-detector.pt` — AI Weights (Веса нейросети).
- `scripts/check_gpu.py` — GPU Diagnostics (Проверка видеокарты).
- `scripts/test_camera.py` — Camera Diagnostics (Проверка камеры).

---

### 🚀 Installation / Установка:
1. Clone the repository / Клонируйте репозиторий.
2. Install dependencies / Установите библиотеки:
   ```bash
   pip install -r requirements.txt

### Check your GPU before start
  python scripts/check_gpu.py

  python app/main.py
