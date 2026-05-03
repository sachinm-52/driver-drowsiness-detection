import cv2
import numpy as np
import time
import os
import platform
import threading

print("=" * 50)
print("DROWSINESS DETECTION SYSTEM")
print("Alarm Time: 5 Seconds")
print("=" * 50)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

cap.set(3, 640)
cap.set(4, 480)

# Load detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load detection models!")
    exit()

# Parameters
closed_frames = 0
ALARM_SECONDS = 5  # 5 seconds
FRAMES_PER_SECOND = 30
ALARM_FRAMES = ALARM_SECONDS * FRAMES_PER_SECOND

alarm_count = 0
MAX_ALARMS = 3
emergency_mode = False
emergency_stop_complete = False

# Timers
eye_closed_start_time = None
emergency_start_time = None
last_alarm_time = 0

print(f"\nSystem Ready: Alarm triggers after {ALARM_SECONDS} seconds")
print("Controls: q=Quit, r=Reset, t=Test Alarm")
print("-" * 50)

def play_alarm_sound():
    """Play alarm sound in background"""
    def play():
        try:
            system_platform = platform.system()
            if system_platform == "Windows":
                import winsound
                winsound.Beep(1000, 500)
                winsound.Beep(800, 500)
                winsound.Beep(1000, 500)
            elif system_platform == "Darwin":  # macOS
                os.system('afplay /System/Library/Sounds/Funk.aiff 2>/dev/null')
            else:  # Linux
                for freq in [1000, 800, 1000]:
                    os.system(f'play -n synth 0.5 sin {freq} 2>/dev/null')
        except:
            print('\a\a\a')
    
    thread = threading.Thread(target=play, daemon=True)
    thread.start()

def enhance_image(frame):
    """Simple image enhancement for better detection"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def detect_eyes(roi_gray, roi_color):
    """Simple eye detection"""
    eyes = eye_cascade.detectMultiScale(
        roi_gray, 
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 15)
    )
    
    eyes_detected = 0
    for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 1)
        eyes_detected += 1
    
    return eyes_detected

# Main loop
frame_count = 0
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Enhance image for better detection
    enhanced = enhance_image(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    if not emergency_mode:
        # NORMAL MODE
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100)
        )
        
        eyes_detected = 0
        face_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            # Draw face rectangle with status color
            if closed_frames > ALARM_FRAMES * 0.7:
                face_color = (0, 0, 255)  # Red - Critical
            elif closed_frames > ALARM_FRAMES * 0.4:
                face_color = (0, 165, 255)  # Orange - Warning
            else:
                face_color = (0, 255, 0)  # Green - Normal
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
            
            # Eye detection region
            roi_gray = gray[y:y+int(h*0.5), x:x+w]
            roi_color = frame[y:y+int(h*0.5), x:x+w]
            
            eyes_detected = detect_eyes(roi_gray, roi_color)
            
            # Draw eye status
            eye_status = "OPEN" if eyes_detected >= 2 else "CLOSED"
            eye_color = (0, 255, 0) if eye_status == "OPEN" else (0, 0, 255)
            cv2.putText(frame, eye_status, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        
        # Update eye state
        if eyes_detected >= 2:
            closed_frames = max(0, closed_frames - 2)  # Faster recovery
            eye_closed_start_time = None
        elif face_detected:
            closed_frames += 1
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
        
        # Calculate time until alarm
        time_until_alarm = max(0, ALARM_SECONDS - (closed_frames / FRAMES_PER_SECOND))
        
        # Check for alarm (5 seconds)
        if closed_frames >= ALARM_FRAMES and time.time() - last_alarm_time > 1.5:
            alarm_count += 1
            last_alarm_time = time.time()
            
            print(f"\nALARM #{alarm_count} - Eyes closed for {ALARM_SECONDS} seconds")
            play_alarm_sound()
            
            # Reset after alarm
            closed_frames = ALARM_FRAMES // 2
            
            # Check for emergency mode
            if alarm_count >= MAX_ALARMS and not emergency_mode:
                emergency_mode = True
                emergency_start_time = time.time()
                print("\nEMERGENCY MODE ACTIVATED!")
        
        # MINIMAL UI - Only essential info
        
        # Top status bar (thin - 30px)
        cv2.rectangle(frame, (0, 0), (640, 30), (40, 40, 50), -1)
        
        # Status indicator
        if face_detected:
            if eyes_detected >= 2:
                status = "AWAKE"
                status_color = (0, 255, 0)
            else:
                if time_until_alarm < 1.0:
                    status = "DROWSY!"
                    status_color = (0, 0, 255)
                elif time_until_alarm < 3.0:
                    status = "SLEEPY"
                    status_color = (0, 165, 255)
                else:
                    status = "ALERT"
                    status_color = (0, 255, 255)
        else:
            status = "NO FACE"
            status_color = (150, 150, 150)
        
        # Calculate text widths for proper positioning
        status_text = f"Status: {status}"
        timer_text = f"Alarm: {time_until_alarm:.1f}s"
        alarm_text = f"Alarms: {alarm_count}/{MAX_ALARMS}"
        
        # Get text sizes
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        alarm_size = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Position text properly with margins
        status_x = 10  # Left margin
        timer_x = 320 - (timer_size[0] // 2)  # Center
        alarm_x = 640 - alarm_size[0] - 10  # Right margin with 10px padding
        
        # Draw status text
        cv2.putText(frame, status_text, (status_x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Draw timer with color coding
        timer_color = (0, 255, 0)
        if time_until_alarm < 1.0:
            timer_color = (0, 0, 255)
        elif time_until_alarm < 3.0:
            timer_color = (0, 165, 255)
        
        cv2.putText(frame, timer_text, (timer_x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, timer_color, 1)
        
        # Draw alarm counter - properly positioned
        alarm_color = (255, 50, 50) if alarm_count > 0 else (150, 150, 150)
        cv2.putText(frame, alarm_text, (alarm_x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, alarm_color, 1)
        
        # Bottom info bar (thin - 25px)
        cv2.rectangle(frame, (0, 455), (640, 480), (30, 30, 40), -1)
        
        # FPS counter
        fps = int(1/(time.time() - last_frame_time))
        cv2.putText(frame, f"FPS: {fps}", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1)
        
        # Controls hint
        controls_text = "q=Quit  r=Reset  t=Test"
        controls_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        controls_x = (640 - controls_size[0]) // 2  # Center horizontally
        cv2.putText(frame, controls_text, (controls_x, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Frame counter
        frame_text = f"Frame: {frame_count}"
        frame_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        frame_x = 640 - frame_size[0] - 10  # Right margin with 10px padding
        cv2.putText(frame, frame_text, (frame_x, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
        
        # Show face positioning help if needed
        if not face_detected and frame_count % 60 < 30:
            cv2.putText(frame, "Position face in center", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 1)
            cv2.rectangle(frame, (270, 160), (370, 260), (0, 200, 200), 2)
            cv2.line(frame, (320, 160), (320, 260), (0, 200, 200), 1)
            cv2.line(frame, (270, 210), (370, 210), (0, 200, 200), 1)
            
    else:
        # EMERGENCY MODE - Simple visualization
        # Darken the frame
        frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        
        # Add emergency overlay
        cv2.rectangle(frame, (150, 180), (490, 220), (0, 0, 100), -1)
        cv2.rectangle(frame, (150, 180), (490, 220), (0, 0, 255), 2)
        
        # Blinking text
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, "EMERGENCY STOP", (180, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Calculate progress
        if emergency_start_time:
            elapsed = time.time() - emergency_start_time
            progress = min(elapsed / 10.0, 1.0) * 100  # 10 second procedure
            
            # Progress bar
            cv2.rectangle(frame, (170, 230), (470, 245), (50, 50, 50), -1)
            cv2.rectangle(frame, (170, 230), (170 + int(300 * progress/100), 245), (0, int(255 * progress/100), 0), -1)
            
            # Progress text
            cv2.putText(frame, f"Stopping: {progress:.0f}%", (250, 265), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Check if complete
            if elapsed > 10 and not emergency_stop_complete:
                emergency_stop_complete = True
                print("\nVehicle safely stopped")
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (640, 30), (40, 0, 0), -1)
        emergency_text = "EMERGENCY MODE"
        emergency_size = cv2.getTextSize(emergency_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        emergency_x = (640 - emergency_size[0]) // 2  # Center the text
        cv2.putText(frame, emergency_text, (emergency_x, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Display frame
    cv2.imshow('Drowsiness Detection - Minimal UI', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset system
        closed_frames = 0
        alarm_count = 0
        emergency_mode = False
        emergency_stop_complete = False
        eye_closed_start_time = None
        print("\nSystem Reset")
    elif key == ord('t'):
        # Test alarm
        if not emergency_mode:
            alarm_count = min(alarm_count + 1, MAX_ALARMS)
            print(f"\nTest Alarm #{alarm_count}")
            play_alarm_sound()
            
            if alarm_count >= MAX_ALARMS:
                emergency_mode = True
                emergency_start_time = time.time()
                print("Emergency Mode (Test)")
    
    # Update frame counter
    frame_count += 1
    last_frame_time = time.time()

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("SYSTEM REPORT:")
print(f"Alarms triggered: {alarm_count}/{MAX_ALARMS}")
print(f"Emergency mode: {'Yes' if emergency_mode else 'No'}")
print("=" * 50)