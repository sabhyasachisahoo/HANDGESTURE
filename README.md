# 🖐️ Hand Gesture Media Controller

This project uses **OpenCV**, **MediaPipe**, and **PyAutoGUI** to control media playback and system volume using hand gestures via your webcam.  
You can **Play/Pause**, **Skip**, and **Control Volume** without touching the keyboard or mouse.

---

## ✨ Features
- **Play/Pause Control** using left-hand index finger.
- **Volume Control** by pinching with your right-hand thumb and index finger.
- **Double-Tap Gestures**:
  - Right-hand double tap → Skip forward 10 seconds.
  - Left-hand double tap → Skip backward 10 seconds.
- **Gesture Lock**: Show all fingers to lock gestures, close fist to unlock.
- **Visual Feedback** on screen (Volume bar, Lock status).
- Works with media players like **VLC** (auto-focus for skip).

---

## 🛠️ Requirements
Install dependencies before running:
```bash
pip install opencv-python mediapipe numpy pyautogui pygetwindow comtypes pycaw
```
▶️ How to Run

1. Clone this repository:
```bash
git clone https://github.com/sabhyasachisahoo/HANDGESTURE.git
```

2. Navigate to the project folder:
```bash
cd HANDGESTURE
```

3. Run the application:
```bash
python app.py
```
