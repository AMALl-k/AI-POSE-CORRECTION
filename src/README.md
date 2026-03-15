# AI-Powered Personal Fitness Trainer 🏋️‍♂️

A real-time computer vision application that tracks workout repetitions, calculates form depth, and automatically logs performance data. Built with Python, OpenCV, and MediaPipe.

## 🌟 Key Features
* **Auto-Exercise Detection:** Uses heuristic pose analysis to differentiate between Squats and Bicep Curls during the setup phase.
* **Intelligent Rep Counting:** Employs a directional state machine to ensure full range of motion and prevent "ghost counting."
* **Personal Best (PB) Tracking:** Automatically compares current session data against `workout_log.csv` to notify users of new records.
* **Visual Depth Gauge:** Real-time feedback showing hip-to-ankle depth ratios for squat validation.

## 🛠️ Technical Challenges & Solutions

### 1. Overcoming UI Blocking
* **Challenge:** Traditional audio and data processing caused the camera feed to lag.
* **Solution:** Optimized the execution flow to ensure a high-performance 30FPS inference rate, prioritizing visual feedback.

### 2. State Machine Hysteresis
* **Challenge:** Signal jitter caused "double counting" when users lingered at the bottom of a rep.
* **Solution:** Implemented a "Buffer Zone" logic. By requiring a specific entry threshold (0.72 ratio) and a distinct exit threshold (0.62 ratio), the system ignores minor movements and only counts intentional reps.

### 3. Perspective-Invariant Tracking
* **Challenge:** Knee angles vary wildly based on camera height.
* **Solution:** Developed a **Depth-Ratio Heuristic** ($hip\_y / ankle\_y$). This coordinate-based approach is more robust across different camera angles than traditional 2D angle calculations.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-Fitness-Trainer.git](https://github.com/YOUR_USERNAME/AI-Fitness-Trainer.git)