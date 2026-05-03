import streamlit as st
import cv2
import numpy as np
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }
    
    .status-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    .status-awake {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.2), rgba(0, 150, 60, 0.1));
        border-color: rgba(0, 200, 83, 0.3);
    }
    
    .status-sleepy {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.2), rgba(200, 130, 0, 0.1));
        border-color: rgba(255, 165, 0, 0.3);
    }
    
    .status-drowsy {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.2), rgba(200, 30, 30, 0.1));
        border-color: rgba(255, 50, 50, 0.3);
        animation: pulse-red 1s infinite;
    }
    
    .status-noface {
        background: linear-gradient(135deg, rgba(150, 150, 150, 0.2), rgba(100, 100, 100, 0.1));
        border-color: rgba(150, 150, 150, 0.3);
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 50, 50, 0.3); }
        50% { box-shadow: 0 0 25px rgba(255, 50, 50, 0.6); }
    }
    
    .status-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .status-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .status-awake .status-value { color: #00c853; }
    .status-sleepy .status-value { color: #ffa500; }
    .status-drowsy .status-value { color: #ff3232; }
    .status-noface .status-value { color: #999; }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #a8b4ff;
        margin: 0.2rem 0 0 0;
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .info-box h4 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
    }
    
    .info-box p, .info-box li {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
    }
    
    .emergency-banner {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.3), rgba(180, 0, 0, 0.2));
        border: 2px solid rgba(255, 0, 0, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        animation: pulse-red 0.5s infinite;
        margin-bottom: 1rem;
    }
    
    .emergency-banner h2 {
        color: #ff3232;
        margin: 0;
    }
    
    /* Fix streamlit dark theme issues */
    .stSidebar { background: rgba(15, 12, 41, 0.95); }
    
    div[data-testid="stSidebarContent"] {
        background: rgba(15, 12, 41, 0.95);
    }
    
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


class DrowsinessDetector(VideoProcessorBase):
    """Video processor for drowsiness detection using WebRTC."""
    
    def __init__(self):
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Detection state
        self.closed_frames = 0
        self.alarm_seconds = 5
        self.fps_estimate = 15  # WebRTC typically runs ~15fps
        self.alarm_frames = self.alarm_seconds * self.fps_estimate
        self.alarm_count = 0
        self.max_alarms = 3
        self.emergency_mode = False
        self.emergency_start_time = None
        
        # Status tracking (thread-safe)
        self._lock = threading.Lock()
        self._status = "NO FACE"
        self._eyes_status = "N/A"
        self._time_until_alarm = self.alarm_seconds
        self._alarm_count = 0
        self._emergency = False
        self._fps = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
    
    def get_status(self):
        """Get current detection status (thread-safe)."""
        with self._lock:
            return {
                "status": self._status,
                "eyes": self._eyes_status,
                "timer": self._time_until_alarm,
                "alarms": self._alarm_count,
                "emergency": self._emergency,
                "fps": self._fps,
                "frames": self._frame_count
            }
    
    def _enhance_image(self, frame):
        """Enhance image for better detection."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _detect_eyes(self, roi_gray, roi_color):
        """Detect eyes in face ROI."""
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(20, 15)
        )
        
        count = 0
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), 1)
            count += 1
        
        return count
    
    def recv(self, frame):
        """Process each video frame."""
        img = frame.to_ndarray(format="bgr24")
        
        # Calculate FPS
        now = time.time()
        dt = now - self._last_frame_time
        fps = int(1 / dt) if dt > 0 else 0
        self._last_frame_time = now
        self._frame_count += 1
        
        # Mirror
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        
        # Enhance and convert to gray
        enhanced = self._enhance_image(img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        if not self.emergency_mode:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100)
            )
            
            eyes_detected = 0
            face_detected = len(faces) > 0
            
            for (x, y, fw, fh) in faces:
                # Face rectangle color based on danger level
                if self.closed_frames > self.alarm_frames * 0.7:
                    face_color = (0, 0, 255)
                elif self.closed_frames > self.alarm_frames * 0.4:
                    face_color = (0, 165, 255)
                else:
                    face_color = (0, 255, 0)
                
                cv2.rectangle(img, (x, y), (x + fw, y + fh), face_color, 2)
                
                # Eye detection in upper half of face
                roi_gray = gray[y:y + int(fh * 0.5), x:x + fw]
                roi_color = img[y:y + int(fh * 0.5), x:x + fw]
                eyes_detected = self._detect_eyes(roi_gray, roi_color)
                
                # Eye status label
                eye_text = "OPEN" if eyes_detected >= 2 else "CLOSED"
                eye_col = (0, 255, 0) if eyes_detected >= 2 else (0, 0, 255)
                cv2.putText(img, eye_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_col, 2)
            
            # Update closed frame counter
            if eyes_detected >= 2:
                self.closed_frames = max(0, self.closed_frames - 2)
            elif face_detected:
                self.closed_frames += 1
            
            time_until_alarm = max(0, self.alarm_seconds - (self.closed_frames / self.fps_estimate))
            
            # Check alarm
            if self.closed_frames >= self.alarm_frames:
                self.alarm_count += 1
                self.closed_frames = self.alarm_frames // 2
                
                if self.alarm_count >= self.max_alarms:
                    self.emergency_mode = True
                    self.emergency_start_time = time.time()
            
            # Determine status
            if face_detected:
                if eyes_detected >= 2:
                    status = "AWAKE"
                    eyes_st = "OPEN"
                elif time_until_alarm < 1.0:
                    status = "DROWSY!"
                    eyes_st = "CLOSED"
                elif time_until_alarm < 3.0:
                    status = "SLEEPY"
                    eyes_st = "CLOSED"
                else:
                    status = "ALERT"
                    eyes_st = "CLOSED"
            else:
                status = "NO FACE"
                eyes_st = "N/A"
                # Show positioning guide
                if self._frame_count % 60 < 30:
                    cx, cy = w // 2, h // 2
                    cv2.putText(img, "Position face in center", (w // 2 - 140, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 1)
                    cv2.rectangle(img, (cx - 50, cy - 60), (cx + 50, cy + 40), (0, 200, 200), 2)
            
            # --- Draw HUD on video frame ---
            
            # Top bar
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Status text
            if status == "AWAKE":
                s_col = (0, 255, 0)
            elif status == "DROWSY!":
                s_col = (0, 0, 255)
            elif status == "SLEEPY":
                s_col = (0, 165, 255)
            elif status == "ALERT":
                s_col = (0, 255, 255)
            else:
                s_col = (150, 150, 150)
            
            cv2.putText(img, f"Status: {status}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_col, 1)
            
            timer_text = f"Alarm: {time_until_alarm:.1f}s"
            timer_col = (0, 255, 0) if time_until_alarm > 3 else (0, 165, 255) if time_until_alarm > 1 else (0, 0, 255)
            t_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(img, timer_text, (w // 2 - t_size[0] // 2, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, timer_col, 1)
            
            alarm_text = f"Alarms: {self.alarm_count}/{self.max_alarms}"
            a_size = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            a_col = (255, 50, 50) if self.alarm_count > 0 else (150, 150, 150)
            cv2.putText(img, alarm_text, (w - a_size[0] - 10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, a_col, 1)
            
            # Progress bar at bottom
            bar_h = 4
            bar_y = h - bar_h
            progress = min(self.closed_frames / self.alarm_frames, 1.0)
            bar_w = int(w * progress)
            
            if progress > 0.7:
                bar_col = (0, 0, 255)
            elif progress > 0.4:
                bar_col = (0, 165, 255)
            else:
                bar_col = (0, 255, 0)
            
            cv2.rectangle(img, (0, bar_y), (bar_w, h), bar_col, -1)
            
        else:
            # EMERGENCY MODE
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
            
            # Emergency overlay
            cv2.rectangle(img, (w//2 - 170, h//2 - 30), (w//2 + 170, h//2 + 10), (0, 0, 100), -1)
            cv2.rectangle(img, (w//2 - 170, h//2 - 30), (w//2 + 170, h//2 + 10), (0, 0, 255), 2)
            
            if int(time.time() * 2) % 2 == 0:
                t_size = cv2.getTextSize("EMERGENCY STOP", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(img, "EMERGENCY STOP", (w//2 - t_size[0]//2, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if self.emergency_start_time:
                elapsed = time.time() - self.emergency_start_time
                progress = min(elapsed / 10.0, 1.0) * 100
                
                cv2.rectangle(img, (w//2 - 150, h//2 + 20), (w//2 + 150, h//2 + 35), (50, 50, 50), -1)
                cv2.rectangle(img, (w//2 - 150, h//2 + 20),
                              (w//2 - 150 + int(300 * progress / 100), h//2 + 35),
                              (0, int(255 * progress / 100), 0), -1)
                
                cv2.putText(img, f"Stopping: {progress:.0f}%", (w//2 - 60, h//2 + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            status = "EMERGENCY"
            eyes_st = "N/A"
            time_until_alarm = 0
        
        # Update shared state
        with self._lock:
            self._status = status
            self._eyes_status = eyes_st
            self._time_until_alarm = time_until_alarm if not self.emergency_mode else 0
            self._alarm_count = self.alarm_count
            self._emergency = self.emergency_mode
            self._fps = fps
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Driver Drowsiness Detection</h1>
        <p>Real-time eye monitoring system • Alarm after 5 seconds of closed eyes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        alarm_seconds = st.slider("Alarm Threshold (seconds)", 3, 10, 5)
        max_alarms = st.slider("Max Alarms before Emergency", 2, 5, 3)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
            <h4>📖 How to Use</h4>
            <ol>
                <li>Click <b>"START"</b> to enable your webcam</li>
                <li>Position your face in the center</li>
                <li>The system monitors your eyes</li>
                <li>Close your eyes for 5s to test the alarm</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Status Indicators</h4>
            <p>🟢 <b>AWAKE</b> — Eyes open, alert</p>
            <p>🟡 <b>ALERT</b> — Eyes partially closed</p>
            <p>🟠 <b>SLEEPY</b> — Alarm approaching</p>
            <p>🔴 <b>DROWSY!</b> — Alarm triggered!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main layout
    col_video, col_status = st.columns([3, 1])
    
    with col_video:
        ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=DrowsinessDetector,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    
    with col_status:
        status_placeholder = st.empty()
        timer_placeholder = st.empty()
        alarm_placeholder = st.empty()
        eyes_placeholder = st.empty()
        fps_placeholder = st.empty()
    
    # Live status updates
    if ctx.state.playing and ctx.video_processor:
        processor = ctx.video_processor
        
        # Update settings from sidebar
        processor.alarm_seconds = alarm_seconds
        processor.alarm_frames = alarm_seconds * processor.fps_estimate
        processor.max_alarms = max_alarms
        
        while ctx.state.playing:
            info = processor.get_status()
            
            # Status card
            status = info["status"]
            if status == "AWAKE":
                css_class = "status-awake"
            elif status in ("SLEEPY", "ALERT"):
                css_class = "status-sleepy"
            elif status in ("DROWSY!", "EMERGENCY"):
                css_class = "status-drowsy"
            else:
                css_class = "status-noface"
            
            with status_placeholder.container():
                st.markdown(f"""
                <div class="status-card {css_class}">
                    <div class="status-label">Driver Status</div>
                    <div class="status-value">{status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with timer_placeholder.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⏱️ Alarm Timer</div>
                    <div class="metric-value">{info['timer']:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)
            
            with alarm_placeholder.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🔔 Alarms</div>
                    <div class="metric-value">{info['alarms']} / {max_alarms}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with eyes_placeholder.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">👁️ Eyes</div>
                    <div class="metric-value">{info['eyes']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with fps_placeholder.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 FPS</div>
                    <div class="metric-value">{info['fps']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Emergency banner
            if info["emergency"]:
                st.markdown("""
                <div class="emergency-banner">
                    <h2>🚨 EMERGENCY MODE ACTIVATED 🚨</h2>
                    <p style="color: rgba(255,255,255,0.7);">Multiple drowsiness alerts detected. Vehicle stopping procedure initiated.</p>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(0.3)
    else:
        # Show placeholder when not streaming
        with col_status:
            st.markdown("""
            <div class="status-card status-noface">
                <div class="status-label">Driver Status</div>
                <div class="status-value">OFFLINE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">⏱️ Alarm Timer</div>
                <div class="metric-value">—</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🔔 Alarms</div>
                <div class="metric-value">0 / 3</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">👁️ Eyes</div>
                <div class="metric-value">—</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("👆 Click **START** above to begin drowsiness detection with your webcam.")


if __name__ == "__main__":
    main()
