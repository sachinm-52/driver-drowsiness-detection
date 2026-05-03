# 🚗 Driver Drowsiness Detection System

A real-time drowsiness detection web app using OpenCV and Streamlit that monitors a driver's eyes through a webcam and triggers alarms when drowsiness is detected.

## 🌐 Live Demo

**[▶️ Try it here on Streamlit Cloud](https://driver-drowsiness-detection.streamlit.app)**

## ✨ Features

- **Real-time face & eye detection** using Haar Cascade classifiers
- **Browser-based webcam access** — no installation needed for end users
- **5-second drowsiness alarm** — triggers when eyes stay closed for 5 seconds
- **Emergency mode** — activates after 3 consecutive alarms
- **Visual status indicators** — color-coded alerts (Green → Orange → Red)
- **Premium dark UI** with live status cards and progress bars
- **Adjustable settings** via sidebar controls

## 📋 Requirements

- Python 3.8+
- Webcam
- Modern browser (Chrome, Firefox, Edge)

## 🚀 Local Setup

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

# Run the web app
streamlit run app.py
```

## 🎮 How to Use

1. Click **START** to enable your webcam
2. Position your face in the center of the frame
3. The system monitors your eyes in real-time
4. Close your eyes for 5 seconds to test the alarm
5. Adjust settings in the sidebar

## 📸 Status Indicators

- 🟢 **AWAKE** — Eyes open, driver is alert
- 🟡 **ALERT** — Eyes partially closed
- 🟠 **SLEEPY** — Eyes closing, alarm approaching
- 🔴 **DROWSY!** — Alarm triggered!

## 🔧 How It Works

1. Captures video from webcam via WebRTC
2. Detects face using Haar Cascade classifier
3. Detects eyes within the face region
4. If eyes are closed for **5 seconds**, an alarm triggers
5. After **3 alarms**, the system enters **Emergency Mode**

## 🖥️ Desktop Version

To run the original desktop version (with OpenCV window):

```bash
python 1.py
```

## 📄 License

This project is open source and available for educational purposes.
