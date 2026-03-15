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
        print(f"Logged: {reps} {exercise}s")
    except Exception as e:
        print(f"Save Error: {e}")

# --- 2. MAIN APPLICATION ---


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector(detection_con=0.8, track_con=0.8)

    # State Variables
    count = 0
    direction = 0
    current_exercise = "Detecting..."
    pb_value = 0
    new_record = False

    start_time = time.time()
    calibration_seconds = 4

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.find_pose(img)
            lm_list = detector.get_positions(img)
            elapsed_time = time.time() - start_time

            if elapsed_time < calibration_seconds:
                # PHASE 1: COUNTDOWN & AUTO-DETECT
                countdown = int(calibration_seconds - elapsed_time)

                if len(lm_list) != 0:
                    # HEURISTIC CLASSIFIER:
                    # If hips are significantly lower than shoulders -> Squat
                    # If wrists are moving significantly relative to elbows -> Curl
                    hip_y = lm_list[24][2]
                    shoulder_y = lm_list[12][2]
                    wrist_y = lm_list[16][2]

                    if abs(wrist_y - shoulder_y) < 150:  # Hands are up near shoulders
                        current_exercise = "bicep_curl"
                    else:
                        current_exercise = "squat"

                    pb_value = get_personal_best(current_exercise)

                cv2.rectangle(img, (150, 150), (490, 350),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(img, f"READY: {countdown}", (200, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                cv2.putText(img, f"MODE: {current_exercise.upper()}", (170, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                # PHASE 2: ACTIVE TRACKING
                if len(lm_list) != 0:
                    # Common Landmarks
                    hip_y = lm_list[24][2]
                    ankle_y = lm_list[28][2]
                    elbow_angle = detector.find_angle(
                        img, 12, 14, 16, draw=False)
                    knee_angle = detector.find_angle(
                        img, 24, 26, 28, draw=False)

                    # LOGIC BRANCHES
                    if current_exercise == "squat":
                        ratio = hip_y / ankle_y if ankle_y != 0 else 0
                        # Squat State Machine
                        if ratio > 0.72 and direction == 0:
                            count += 0.5
                            direction = 1
                        if ratio < 0.62 and direction == 1:
                            count += 0.5
                            direction = 0

                    elif current_exercise == "bicep_curl":
                        per = np.interp(elbow_angle, (30, 160), (100, 0))
                        # Curl State Machine
                        if per == 100 and direction == 0:
                            count += 0.5
                            direction = 1
                        if per == 0 and direction == 1:
                            count += 0.5
                            direction = 0

                    # NEW RECORD NOTIFICATION
                    if int(count) > pb_value and pb_value > 0:
                        new_record = True

                # --- UI DESIGN ---
                # Top Header
                cv2.rectangle(img, (0, 0), (640, 50), (40, 40, 40), cv2.FILLED)
                cv2.putText(img, f"EXERCISE: {current_exercise.upper()}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Rep Counter
                cv2.putText(img, str(int(count)), (540, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 7)

                # PB Banner
                if new_record:
                    cv2.rectangle(img, (150, 60), (490, 100),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "!!! NEW RECORD !!!", (180, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                else:
                    cv2.putText(img, f"PB: {pb_value}", (10, 80),
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
