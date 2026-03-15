import cv2
import numpy as np
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detection_con=0.5, track_con=0.5):
        
        # We access the solutions INSIDE the class to give the 
        # system time to map the library paths.
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:
            # This is the 'emergency' direct access
            from mediapipe.python.solutions import pose as mp_p
            from mediapipe.python.solutions import drawing_utils as mp_d
            self.mp_pose = mp_p
            self.mp_draw = mp_d
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
    
    # ... keep find_pose, get_positions, and find_angle below ...

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_positions(self, img):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle at landmark p2 formed by p1-p2-p3.
        Example: p1=Shoulder, p2=Elbow, p3=Wrist
        """
        # 1. Get coordinates (x, y) from our landmark list
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # 2. Calculate the angle using the arctangent function
        # This gives us the angle in radians, which we convert to degrees
        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - 
                           np.arctan2(y1 - y2, x1 - x2))
        
        # 3. Standardize the angle to be between 0 and 180 degrees
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        # 4. Visualization (Professional Feedback)
        if draw:
            # Draw the 'bone' lines
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            # Draw the joints
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            # Display the angle value near the joint
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        return angle