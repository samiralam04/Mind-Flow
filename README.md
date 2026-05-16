# 🧠 MindFlow

**MindFlow** is a state-of-the-art, real-time physiological analytics platform that tracks and predicts cognitive load, fatigue, and behavioral states using a standard webcam feed. By combining computer vision (MediaPipe) with deep temporal modeling (BiLSTM + Attention), the system provides high-fidelity insights into mental effort and cognitive performance.

---

## 🚀 Key Features

*   **Real-time Physiological Stream**: Captures EAR (Eye Aspect Ratio), Gaze Pitch/Yaw, Head Pose (Pitch/Yaw/Roll), Eyebrow Tension, and Eye Openness at 30 FPS.
*   **Deep Temporal Inference**: A Bidirectional LSTM with Attention mechanism analyzes 5-second temporal windows to predict cognitive load with high sensitivity.
*   **Behavioral Intelligence**: Interprets raw scores into meaningful states like `Focused`, `Elevated Load`, `Fatigued`, or `Distracted` based on cognitive science models.
*   **Personalized Calibration**: A built-in system to establish a unique "Relaxed" baseline for each user, compensating for physiological variance.
*   **Research-Grade Analytics**: Includes tools for dataset quality analysis, outlier detection, feature correlation, and sequence quality scoring.
*   **High-Performance Dashboard**: A futuristic, dark-themed UI with real-time telemetry charts, state indicators, and historical session replay.

---

## 🛠 Tech Stack

### Backend
*   **Python 3.10+** & **FastAPI** (Asynchronous WebSocket & REST API)
*   **PyTorch**: BiLSTM + Attention temporal model.
*   **MediaPipe**: Face landmarker and blendshape extraction.
*   **OpenCV**: Image decoding and head pose estimation (solvePnP).
*   **Pandas/NumPy**: Data processing and sequence generation.

### Frontend
*   **React 19** & **Vite**
*   **Zustand**: Global state management.
*   **Recharts**: Real-time telemetry visualization.
*   **Framer Motion**: Micro-animations and fluid UI transitions.
*   **Tailwind CSS**: Modern utility-first styling.

---

## 🧠 Machine Learning Engine

The project utilizes a multi-stage ML pipeline designed for robustness and interpretability.

### 1. Behavioral States
The **Behavioral Intelligence Engine** (`behavioral_state.py`) maps scores to states using a hysteresis-aware state machine:
*   🟢 **Relaxed**: 0–30 score. Baseline mental state.
*   🔵 **Focused**: 30–65 score. High engagement, productive flow.
*   🟡 **Elevated Load**: 65–80 score. Significant mental demand.
*   🔴 **Overloaded**: 80+ score. Sustained high demand; performance may degrade.
*   🟣 **Fatigued**: High load + sustained physiological markers of tiredness.
*   🩵 **Recovering**: Post-overload state where load is actively decreasing.
*   🟠 **Distracted**: Detected via extreme gaze deviation or head pose instability.

### 2. Multi-Factor Labeling
The **Pseudo-Labeling Engine** (`label_engine.py`) generates high-quality training labels by synthesizing:
*   **Task Difficulty**: Prior knowledge of the activity (e.g., Coding vs. Watching Video).
*   **Self-Reports**: Optional user-provided difficulty scores.
*   **Behavioral Signals**: Direct EAR, gaze, and head pose patterns.
*   **Fatigue Accumulation**: Modeling cognitive decay over session duration.

---

## 📊 Data Pipeline

### Step 1: Collection
Webcam frames are processed by `vision.py` and streamed via WebSocket. The `session_manager.py` logs raw telemetry to `backend/dataset/`.

### Step 2: Preprocessing
Run the preprocessing script to generate sequence tensors for training:
```bash
python preprocess.py
```
This produces `X_sequences.npy`, `Y_labels.npy`, and `Y_confidence.npy` in the `processed_data/` directory.

### Step 3: Analysis
Before training, run the diagnostic suite to verify dataset health:
```bash
python ml/analyze_dataset.py
```
This generates a full report on label distribution, feature correlation, and sequence quality.

### Step 4: Training
Train the BiLSTM Attention model:
```bash
python ml/train.py
```
Includes early stopping, mixed-precision training, and TensorBoard logging.

### Step 5: Evaluation
Generate performance metrics (MAE, RMSE, Pearson r) and visualization plots:
```bash
# Triggered via API or running:
python ml/evaluate.py
```

---

## ⚙️ Setup & Installation

### 1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements_ml.txt
pip install fastapi uvicorn mediapipe opencv-python aiofiles
```

### 2. Frontend
```bash
cd frontend
npm install
```

---

## 📖 Usage Guide

1.  **Start Backend**: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
2.  **Start Frontend**: `npm run dev`
3.  **Calibrate**: Open the app, enter a User ID, and run the 10-second calibration.
4.  **Monitor**: Click "Start Monitoring" to see live load and behavioral states.
5.  **Record**: Toggle "Recording" to save data for future model training.

---

## 📡 API Reference

*   `GET /`: Health check and ML engine status.
*   `GET /model/info`: Current checkpoint metrics and epoch.
*   `POST /model/eval`: Trigger a background evaluation report.
*   `POST /preprocess`: Trigger the data preprocessing pipeline.
*   `POST /calibration/start`: Begin user-specific baseline capture.
*   `GET /sessions/{user}/{session}/replay`: Load full session timeline for scrubbing.
*   `WS /ws/stream`: Bi-directional stream for frames and live predictions.

---

## 📁 Project Structure

```text
ML-project/
├── backend/                # FastAPI application
│   ├── ml/                 # ML Engine core
│   │   ├── behavioral_state.py  # Interpretive state machine
│   │   ├── calibration.py       # Personalization system
│   │   ├── inference.py         # Optimized inference engine
│   │   ├── label_engine.py      # Multi-factor labeling
│   │   ├── train.py             # Model trainer
│   │   └── evaluate.py          # Metrics & Reporting
│   ├── dataset/            # Raw session data (CSVs)
│   ├── processed_data/     # Training tensors (NPYs)
│   ├── user_profiles/      # Saved user baselines
│   ├── main.py             # API Entry point
│   ├── vision.py           # Feature extraction (MediaPipe)
│   └── preprocess.py       # Preprocessing pipeline
└── frontend/               # React application
    ├── src/
    │   ├── components/     # UI & Visualization
    │   └── store/          # Zustand state management
```
