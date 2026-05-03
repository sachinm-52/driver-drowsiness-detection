# 🚗 Driver Drowsiness Detection System

A real-time drowsiness detection system using OpenCV and Python that monitors a driver's eyes through a webcam and triggers alarms when drowsiness is detected.

## ✨ Features

- **Real-time face & eye detection** using Haar Cascade classifiers
- **5-second drowsiness alarm** — triggers when eyes stay closed for 5 seconds
- **Emergency mode** — activates after 3 consecutive alarms
- **Visual status indicators** — color-coded alerts (Green → Orange → Red)
- **Cross-platform alarm** — works on Windows, macOS, and Linux
- **Minimal, clean UI** with FPS counter and status bar

## 📋 Requirements

- Python 3.8+
- Webcam

## 🚀 Setup & Run

```bash
# Clone the repository
git clone https://github.com/sachinm-52/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python 1.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `r` | Reset all alarms |
| `t` | Test the alarm sound |

## 🔧 How It Works

1. Captures video from webcam
2. Detects face using Haar Cascade classifier
3. Detects eyes within the face region
4. If eyes are closed for **5 seconds**, an alarm sounds
5. After **3 alarms**, the system enters **Emergency Mode**

## 📸 Status Indicators

- 🟢 **AWAKE** — Eyes open, driver is alert
- 🟡 **ALERT** — Eyes partially closed
- 🟠 **SLEEPY** — Eyes closing, alarm approaching
- 🔴 **DROWSY!** — Alarm triggered!

## 📄 License

This project is open source and available for educational purposes.
