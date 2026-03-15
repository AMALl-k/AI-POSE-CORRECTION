import cv2
import numpy as np
import csv
import time
import pandas as pd
import os
from datetime import datetime
from src.pose_engine import PoseDetector

# --- 1. DATA UTILITIES ---

def get_personal_best(exercise):
    file_path = "workout_log.csv"
    if not os.path.exists(file_path):
        return 0
    try:
        df = pd.read_csv(file_path)
        exercise_data = df[df["Exercise"] == exercise]
        if exercise_data.empty:
            return 0
        return exercise_data['Reps'].max()
    except:
        return 0

def save_workout_data(exercise, reps):
    file_path = "workout_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Exercise", "Reps"])
            writer.writerow([timestamp, exercise, reps])
        print(f"--- Session Logged: {reps} {exercise}s ---")
    except Exception as e:
        print(f"Save Error: {e}")

# --- 2. MAIN APPLICATION ---

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector(detection_con=0.7, track_con=0.7)

    # State Variables
    count = 0
    direction = 0
    current_exercise = "Detecting..."
    pb_value = 0
    new_record = False
    phase_switched = False

    start_time = time.time()
    calibration_seconds = 4

    try:
        while True:
            success, img = cap.read()
            if not success: break

            img = detector.find_pose(img)
            lm_list = detector.get_positions(img)
            elapsed_time = time.time() - start_time

            # PHASE 1: CALIBRATION & AUTO-DETECT
            if elapsed_time < calibration_seconds:
                countdown = max(0, int(calibration_seconds - elapsed_time))

                if len(lm_list) != 0:
                    hip_y = lm_list[24][2]
                    shoulder_y = lm_list[12][2]
                    wrist_y = lm_list[16][2]

                    # SENSITIVITY FIX: If wrists are anywhere above the chest, it's a curl
                    if wrist_y < (shoulder_y + 50): 
                        current_exercise = "bicep_curl"
                    else:
                        current_exercise = "squat"

                    pb_value = get_personal_best(current_exercise)

                cv2.rectangle(img, (150, 150), (490, 350), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, f"READY: {countdown}", (200, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                cv2.putText(img, f"MODE: {current_exercise.upper()}", (170, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # PHASE 2: ACTIVE TRACKING
            else:
                if not phase_switched:
                    phase_switched = True

                if len(lm_list) != 0:
                    # LOGIC BRANCH: SQUAT
                    if current_exercise == "squat":
                        hip_y = lm_list[24][2]
                        ankle_y = lm_list[28][2]
                        ratio = hip_y / ankle_y if ankle_y != 0 else 0
                        
                        if ratio > 0.72 and direction == 0:
                            count += 0.5
                            direction = 1
                        if ratio < 0.62 and direction == 1:
                            count += 0.5
                            direction = 0
                    
                    # LOGIC BRANCH: BICEP CURL (Forgiving Angles)
                    elif current_exercise == "bicep_curl":
                        # We use lm 12 (shoulder), 14 (elbow), 16 (wrist)
                        angle = detector.find_angle(img, 12, 14, 16, draw=True)
                        
                        # FORGIVING RANGE: 40 deg is fully bent, 150 deg is fully straight
                        per = np.interp(angle, (40, 150), (100, 0))
                        
                        if per >= 95 and direction == 0:
                            count += 0.5
                            direction = 1
                        if per <= 5 and direction == 1:
                            count += 0.5
                            direction = 0

                    if int(count) > pb_value and pb_value > 0:
                        new_record = True

                # --- UI DESIGN ---
                cv2.rectangle(img, (0, 0), (640, 50), (30, 30, 30), cv2.FILLED)
                cv2.putText(img, f"EXERCISE: {current_exercise.upper()}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(img, str(int(count)), (530, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)

                if new_record:
                    cv2.rectangle(img, (160, 60), (480, 105), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "NEW RECORD!", (210, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                else:
                    cv2.putText(img, f"PB: {pb_value}", (15, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("AI Fitness Trainer Pro", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if count >= 1:
            save_workout_data(current_exercise, int(count))
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
